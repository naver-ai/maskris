import os
import itertools

import torch
import transforms as T
import utils


def is_distributed():
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    return distributed


def maybe_add_full_model_gradient_clipping(optim, args):  # optim: the optimizer class
    clip_norm_val = args.clip_value
    enable = args.clip_grads

    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
            super().step(closure=closure)

    return FullModelGradientClippingOptimizer if enable else optim


def get_criterion(model):
    from criterion import criterion_dict
    return criterion_dict[model]


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return T.Compose(transforms)


def batch_IoU(pred, gt):
    intersection = torch.sum(torch.mul(pred, gt), dim=1)
    union = torch.sum(torch.add(pred, gt), dim=1) - intersection

    iou = intersection.float() / union.float()

    return iou, intersection, union


def batch_evaluate(model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_num = len(data_loader.dataset)
    acc_ious = torch.zeros(1).cuda()

    # evaluation variables
    # cum_I, cum_U = 0, 0
    cum_I = torch.zeros(1).cuda()
    cum_U = torch.zeros(1).cuda()
    eval_seg_iou_list = [.5, .7, .9]
    seg_correct = torch.zeros(len(eval_seg_iou_list)).cuda()

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):

            image, targets, sentences, attentions = data
            image, sentences, attentions = image.cuda(non_blocking=True), \
                sentences.cuda(non_blocking=True), \
                attentions.cuda(non_blocking=True)
            target = targets['mask'].cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            with torch.cuda.amp.autocast():
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = batch_IoU(output.flatten(1), target.flatten(1))
            acc_ious += iou.sum()
            cum_I += I.sum()
            cum_U += U.sum()
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou).sum()

    torch.cuda.synchronize()
    if is_distributed():
        cum_I = utils.all_reduce_tensor(cum_I, norm=False).cpu().numpy()
        cum_U = utils.all_reduce_tensor(cum_U, norm=False).cpu().numpy()
        acc_ious = utils.all_reduce_tensor(acc_ious, norm=False).cpu().numpy()
        seg_correct = utils.all_reduce_tensor(seg_correct, norm=False).cpu().numpy()
    else:
        cum_I = cum_I.cpu().numpy()
        cum_U = cum_U.cpu().numpy()
        acc_ious = acc_ious.cpu().numpy()
        seg_correct = seg_correct.cpu().numpy()

    mIoU = acc_ious / total_num
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / total_num)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * mIoU, 100 * cum_I / cum_U
