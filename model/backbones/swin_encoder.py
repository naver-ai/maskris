import torch
from torch import nn
from torch.nn import functional as F
from .swin import SwinTransformer
from timm.models.layers import trunc_normal_


class MultiStageSwinTransformer(SwinTransformer):
    '''
        SwinTransformer that supports multi-stage encoding for multimodal learning.
    '''

    def __init__(self, mask_patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.n_stages = self.num_layers
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, std=.02)

        # newly defined parameter for aug-sub
        self.mask_patch_size = mask_patch_size

        assert self.n_stages == 4, 'Only 4-stage index is supported!'

    def forward_embeddings(self, x, mask_ratio=0.0):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # [B, Wh*Ww, C]
        else:
            x = x.flatten(2).transpose(1, 2)

        if mask_ratio > 0:
            x = self.random_masking(x, mask_ratio)

        x = self.pos_drop(x)
        return x, Wh, Ww

    def forward_stages(self, x, Wh, Ww, stage, out_norm=False):
        assert stage in self.out_indices, 'stage {} does not exist in out_indices!'.format(stage)

        layer = self.layers[stage]
        x, Wh, Ww = layer.forward_pre_downsample(x, Wh, Ww)
        if out_norm:
            norm_layer = getattr(self, f'norm{stage}')
            x_out = norm_layer(x)  # output of a Block has shape (B, H*W, dim)
            x_out = x_out.view(-1, Wh, Ww, self.num_features[stage]).permute(0, 3, 1, 2).contiguous()
            return x_out, x, Wh, Ww
        else:
            return x, Wh, Ww

    def forward_downs(self, x, Wh, Ww, stage):
        assert stage in self.out_indices, 'stage {} does not exist in out_indices!'.format(stage)

        layer = self.layers[stage]
        x, Wh, Ww = layer.forward_downsample(x, Wh, Ww)

        return x, Wh, Ww

    def forward_norms(self, x, Wh, Ww, stage):
        assert stage in self.out_indices, 'stage {} does not exist in out_indices!'.format(stage)

        norm_layer = getattr(self, f'norm{stage}')
        x_out = norm_layer(x)  # output of a Block has shape (B, H*W, dim)
        x_out = x_out.view(-1, Wh, Ww, self.num_features[stage]).permute(0, 3, 1, 2).contiguous()  # [B, dim, Wh, Ww]
        return x_out

    def random_masking(self, x, mask_ratio):
        N, L_ori, D = x.shape  # batch, length, dim
        # mask_patch_size = 32 // 4  # Use mask size 32x32 (image) == 8x8 (feature)
        mask_patch_size = self.mask_patch_size // 4

        L = L_ori // (mask_patch_size ** 2)
        len_keep = int(L * (1 - mask_ratio))

        # Normalize mask_noise to [0, 1]
        noise = torch.rand([N, L], device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # expand 7x7 mask to 56x56
        w = int(L ** 0.5)
        mask = mask.view(N, w, w)
        mask = mask.repeat_interleave(mask_patch_size, dim=-1).repeat_interleave(mask_patch_size, dim=-2)
        mask = mask.view(N, L_ori)

        x_masked = x * (1 - mask).unsqueeze(-1) + self.mask_token * mask.unsqueeze(-1)

        return x_masked
