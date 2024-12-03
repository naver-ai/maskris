import random

import torch
import torch.nn.functional as F

import utils
from util.aug_mask import patchify, unpatchify, random_masking


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, print_freq, loss_scaler,
                    clip_grad, args):
    model.train()
    optimizer.zero_grad()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4e}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        image, targets, sentences, sentences_masked, attentions = data
        image, sentences, attentions = image.cuda(non_blocking=True), \
            sentences.cuda(non_blocking=True), \
            attentions.cuda(non_blocking=True)
        sentences_masked = sentences_masked.cuda(non_blocking=True)

        # Image masking for MaskRIS
        image_masked = patchify(image, args.img_patch_size)
        image_masked = random_masking(image_masked, args.img_mask_ratio)
        image_masked = unpatchify(image_masked, args.img_patch_size)

        for k, v in targets.items():
            if isinstance(v, list):
                targets[k] = [m.cuda(non_blocking=True) for m in v]
            else:
                targets[k] = v.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)  # [B, N_l]
        attentions = attentions.squeeze(1)  # [B, N_l]
        sentences_masked = sentences_masked.squeeze(1)  # [B, N_l]

        # first forward pass
        with torch.cuda.amp.autocast():
            loss_dict, logit = model(image, sentences, l_mask=attentions, targets=targets, return_probs=True)
        loss_scaler(loss_dict['total_loss'] * 0.5, optimizer, clip_grad=clip_grad,
                    parameters=model.parameters(), update_grad=False)

        # second forward pass
        with torch.cuda.amp.autocast():
            targets_masked = {'mask': F.softmax(logit.detach(), dim=1)}
            loss_dict_masked = model(image_masked, sentences_masked, l_mask=attentions, targets=targets_masked)
        grad_norm = loss_scaler(loss_dict_masked['total_loss'] * 0.5, optimizer, clip_grad=clip_grad,
                                parameters=model.parameters(), update_grad=True)

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        lr_scheduler.step()

        torch.cuda.synchronize()
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(org_loss=loss_dict['total_loss'].item())
        metric_logger.update(sub_loss=loss_dict_masked['total_loss'].item())
