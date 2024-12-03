import os
import time
import random
import datetime
import numpy as np

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

import utils
import training
from model import builder
from config import get_parser
from util.data import get_dataset
from util.misc import is_distributed, get_criterion, get_transform, batch_evaluate
from util.optimizer import get_optimizer


def main(args, distributed):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built val dataset.")

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks,
                                                                        rank=global_rank,
                                                                        shuffle=True, drop_last=True)
        # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False, drop_last=False)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem,
        drop_last=True, collate_fn=utils.collate_func)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    criterion = get_criterion(args.model)()
    single_model = builder.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                                args=args, criterion=criterion)
    single_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model)
    print(single_model)
    single_model.cuda()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(single_model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(single_model)

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])

    # optimizer
    optimizer = get_optimizer(single_model, args)
    loss_scaler = utils.NativeScalerWithGradNormCount()
    clip_grad = args.clip_value if args.clip_grads else None

    total_iters = (len(data_loader) * args.epochs)
    lr_scheduler = utils.WarmUpPolyLRScheduler(optimizer, total_iters, power=0.9, min_lr=args.min_lr,
                                               warmup=args.warmup, warmup_iters=args.warmup_iters,
                                               warmup_ratio=args.warmup_ratio)
    # housekeeping
    start_time = time.time()
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    trainer = training.train_one_epoch

    # training loops
    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        if distributed:
            data_loader.sampler.set_epoch(epoch)

        trainer(model, optimizer, data_loader, lr_scheduler, epoch, args.print_freq, loss_scaler, clip_grad, args)

        if epoch % 10 == 0 or epoch >= args.epochs - 16:
            iou, overallIoU = batch_evaluate(model, data_loader_test)

            print('Average object IoU {}'.format(iou))
            print('Overall IoU {}'.format(overallIoU))
            save_checkpoint = (best_oIoU < overallIoU)
            if save_checkpoint:
                print('Better epoch: {}\n'.format(epoch))
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict(), 'scaler': loss_scaler.state_dict()}

                utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                                'model_best_{}.pth'.format(args.model_id)))
                best_oIoU = overallIoU

        dict_to_save = {'model': single_model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                        'lr_scheduler': lr_scheduler.state_dict(), 'scaler': loss_scaler.state_dict()}

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    deterministic = args.deterministic

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    # set up distributed learning
    distributed = is_distributed()
    if distributed:
        utils.init_distributed_mode(args)
    print(f'SEED: {args.seed}')
    print('Image size: {}'.format(str(args.img_size)))
    main(args, distributed)
