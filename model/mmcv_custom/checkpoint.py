# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import torch
from torch.nn import functional as F
from torch import distributed as dist
from collections import OrderedDict

# import mmcv
# from mmcv.fileio import FileClient
# from mmcv.fileio import load as load_file
# from mmcv.parallel import is_module_wrapper
# from mmcv.utils import mkdir_or_exist
# from mmcv.runner import get_dist_info

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will NOT be shown if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, in_channel // 4, 4)
        x = x[:, :, [0, 2, 1, 3]].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(in_channel // 4, 4)
        x = x[:, [0, 2, 1, 3]].transpose(0, 1).reshape(in_channel)
        return x

    for _k, v in ckpt.items():
        if not _k.startswith('backbone.'):
            continue
        else:
            k = _k[9:]
            if k.startswith('head'):
                continue
            elif k.startswith('stages'):
                new_v = v
                if 'attn.w_msa.' in k:
                    new_k = k.replace('attn.w_msa.', 'attn.')
                elif 'ffn.' in k:
                    if 'ffn.layers.0.0.' in k:
                        new_k = k.replace('ffn.layers.0.0.', 'mlp.fc1.')
                    elif 'ffn.layers.1.' in k:
                        new_k = k.replace('ffn.layers.1.', 'mlp.fc2.')
                    else:
                        new_k = k.replace('ffn.', 'mlp.')
                elif 'downsample' in k:
                    new_k = k
                    if 'reduction.' in k:
                        new_v = correct_unfold_reduction_order(v)
                    elif 'norm.' in k:
                        new_v = correct_unfold_norm_order(v)
                else:
                    new_k = k
                new_k = new_k.replace('stages', 'layers', 1)
            elif k.startswith('patch_embed'):
                new_v = v
                if 'projection' in k:
                    new_k = k.replace('projection', 'proj')
                else:
                    new_k = k
            else:
                new_v = v
                new_k = k

            new_ckpt[new_k] = new_v

    return new_ckpt

def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    # if filename.startswith('modelzoo://'):
    #     warnings.warn('The URL scheme of "modelzoo://" is deprecated, please '
    #                   'use "torchvision://" instead')
    #     model_urls = get_torchvision_models()
    #     model_name = filename[11:]
    #     checkpoint = load_url_dist(model_urls[model_name])
    # elif filename.startswith('torchvision://'):
    #     model_urls = get_torchvision_models()
    #     model_name = filename[14:]
    #     checkpoint = load_url_dist(model_urls[model_name])
    # elif filename.startswith('open-mmlab://'):
    #     model_urls = get_external_models()
    #     model_name = filename[13:]
    #     deprecated_urls = get_deprecated_model_names()
    #     if model_name in deprecated_urls:
    #         warnings.warn(f'open-mmlab://{model_name} is deprecated in favor '
    #                       f'of open-mmlab://{deprecated_urls[model_name]}')
    #         model_name = deprecated_urls[model_name]
    #     model_url = model_urls[model_name]
    #     # check if is url
    #     if model_url.startswith(('http://', 'https://')):
    #         checkpoint = load_url_dist(model_url)
    #     else:
    #         filename = osp.join(_get_mmcv_home(), model_url)
    #         if not osp.isfile(filename):
    #             raise IOError(f'{filename} is not a checkpoint file')
    #         checkpoint = torch.load(filename, map_location=map_location)
    # elif filename.startswith('mmcls://'):
    #     model_urls = get_mmcls_models()
    #     model_name = filename[8:]
    #     checkpoint = load_url_dist(model_urls[model_name])
    #     checkpoint = _process_mmcls_checkpoint(checkpoint)
    # elif filename.startswith(('http://', 'https://')):
    #     checkpoint = load_url_dist(filename)
    # elif filename.startswith('pavi://'):
    #     model_path = filename[7:]
    #     checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    # elif filename.startswith('s3://'):
    #     checkpoint = load_fileclient_dist(
    #         filename, backend='ceph', map_location=map_location)
    # else:
    if not osp.isfile(filename):
        raise IOError(f'{filename} is not a checkpoint file')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # # for upper net weights only
    # if list(state_dict.keys())[0].startswith('backbone.'):
    #     print('Start stripping upper net pre-fix and loading backbone weights to our swin encoder')
    #     state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
    # # for MoBY, load model of online branch
    # if sorted(list(state_dict.keys()))[0].startswith('encoder'):
    #     state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    if 'mask2former' in filename:
        print('start converting mmdet Mask2Former checkpoint')
        state_dict = swin_converter(state_dict)

    # reshape absolute position embedding
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H*W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                table_pretrained_resized = F.interpolate(
                     table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                     size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


