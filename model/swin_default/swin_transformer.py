"""
    borrow from MoBY
"""
from pytest import xfail
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.distributed as dist
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .register import SwinTransformer
import torch.utils.checkpoint as checkpoint


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, resolution, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        x: B N C
        mask: B N N
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # B head N C' @ B head C' N --> B head N N
        if mask is not None:
            NS = mask.shape[0]
            attn = attn.view(B // NS, NS, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B head N N @ B head N C
        # x = self.proj(x)
        x = F.linear(input=x.float(),
                weight=self.proj.weight.float(),
                bias=self.proj.bias.float(),
            )
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return 'num_heads={}, resolution={}'.format(self.num_heads, self.resolution)

    def flops(self, N):
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class RPETableAttention(nn.Module):
    def __init__(self, dim, resolution, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Not supported now, since we have cls_tokens now.....
        """
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.rel_pos_embed_table = nn.Parameter(
            torch.zeros((2 * resolution[0] - 1) * (2 * resolution[1] - 1),
                        num_heads))  # 2*Ph-1 * 2*Pw-1, nH

        coords_h = torch.arange(self.resolution[0])
        coords_w = torch.arange(self.resolution[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2 Ph Pw
        coords_flatten = torch.flatten(coords, 1)  # 2 Ph*Pw
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2 Ph*Pw Ph*Pw
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ph*Pw Ph*Pw 2
        relative_coords[:, :, 0] += self.resolution[0] - 1
        relative_coords[:, :, 1] += self.resolution[1] - 1
        relative_coords[:, :, 0] *= 2 * self.resolution[1] - 1
        relative_coords = relative_coords.sum(-1)  # Ph*Pw, Ph*Pw

        # relative_coords = relative_coords.view(-1, 2)
        self.register_buffer("relative_coords", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.rel_pos_embed_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, mul_mask=None):
        """
        x: B N C
        mask: B N N
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        rel_pos_embed = self.rel_pos_embed_table[self.relative_coords.view(-1)].view(
            self.resolution[0] * self.resolution[1], self.resolution[0] * self.resolution[1], -1)  # Ph*Pw Ph*Pw head

        rel_pos_embed = rel_pos_embed.permute(2, 0, 1).contiguous()  # head Ph*Pw Ph*Pw

        attn = attn + rel_pos_embed.unsqueeze(0)
        if mask is not None:
            NS = mask.shape[0]
            attn = attn.view(B // NS, NS, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        elif mul_mask is not None:
            NS = mul_mask.shape[0]
            mul_mask_expand = mul_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B // NS, NS, self.num_heads, N, N) * mul_mask_expand
            attn = self.softmax(attn)
            attn = attn * mul_mask_expand
            attn = attn.view(-1, self.num_heads, N, N)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B head N N @ B head N C
        # x = self.proj(x)
        x = F.linear(input=x.float(),
                weight=self.proj.weight.float(),
                bias=self.proj.bias.float(),
            )
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return 'num_heads={}, resolution={}'.format(self.num_heads, self.resolution)

    def flops(self, N):
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SlicedBlockFaster(nn.Module):

    def __init__(self, dim, patches_resolution, num_heads,
                 split_size=7, disturb_size=0,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 rpe='none'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = patches_resolution
        self.split_size = split_size
        self.disturb_size = disturb_size
        self.rpe = rpe
        self.mlp_ratio = mlp_ratio
        if min(self.patches_resolution) <= self.split_size:
            self.disturb_size = 0
            disturb_size = 0
        assert 0 <= self.disturb_size < self.split_size, "disturb_size must in 0-split_size"

        self.norm1 = norm_layer(dim)
        if self.rpe == 'table':
            self.attn = RPETableAttention(
                dim, resolution=to_2tuple(self.split_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif self.rpe == 'none':
            self.attn = Attention(
                dim, resolution=to_2tuple(self.split_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            raise KeyError
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.disturb_size > 0:
            H, W = self.patches_resolution

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            img_mask[:, :-self.split_size, :-self.split_size, :] = 0
            img_mask[:, -self.disturb_size:, -self.disturb_size:, :] = 1
            img_mask[:, -self.disturb_size:, :-self.split_size, :] = 2
            img_mask[:, -self.disturb_size:, -self.split_size:-self.disturb_size, :] = 3
            img_mask[:, :-self.split_size, -self.disturb_size:, :] = 4
            img_mask[:, :-self.split_size, -self.split_size:-self.disturb_size, :] = 5
            img_mask[:, -self.split_size:-self.disturb_size, -self.disturb_size:, :] = 6
            img_mask[:, -self.split_size:-self.disturb_size, :-self.split_size, :] = 7
            img_mask[:, -self.split_size:-self.disturb_size, -self.split_size:-self.disturb_size, :] = 8

            img_mask_splits_hw = faster_img2splits(img_mask, self.split_size)  # B' split_size split_size 1
            img_mask_splits = img_mask_splits_hw.view(-1, self.split_size * self.split_size)  # B' split_size*split_size
            # B' split_size*split_size split_size*split_size
            atten_mask_matrix = img_mask_splits.unsqueeze(1) - img_mask_splits.unsqueeze(2)
            atten_mask_matrix = atten_mask_matrix.masked_fill(atten_mask_matrix != 0, float(-100.0)).masked_fill(
                atten_mask_matrix == 0, float(0.0))
        else:
            atten_mask_matrix = None

        self.register_buffer("atten_mask_matrix", atten_mask_matrix)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)

        img = img.view(B, H, W, C)
        if self.disturb_size > 0:
            disturbed_img = torch.roll(img, shifts=(-self.disturb_size, -self.disturb_size), dims=(1, 2))

        else:
            disturbed_img = img

        # split image
        img_splits_hw = faster_img2splits(disturbed_img, self.split_size)  # B' split_size split_size C

        # attention
        atten_feature = img_splits_hw.view(-1, self.split_size * self.split_size, C)  # B' split_size*split_size C

        atten_feature = self.attn(atten_feature, mask=self.atten_mask_matrix)  # B' split_size*split_size C

        # get img_tokens
        atten_feature = atten_feature.view(-1, self.split_size, self.split_size, C)
        attened_padded_img = faster_splits2img(atten_feature, self.split_size, H, W)  # B H' W' C

        if self.disturb_size > 0:
            attened_img = torch.roll(attened_padded_img, shifts=(self.disturb_size, self.disturb_size), dims=(1, 2))
        else:
            attened_img = attened_padded_img

        attened_img = attened_img.view(B, H * W, C)

        # FFN
        x = x + self.drop_path(attened_img)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return 'patches_resolution={}, split_size={}, disturb_size={}'.format(self.patches_resolution, self.split_size,
                                                                              self.disturb_size)

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        # norm1
        flops += self.dim * H * W

        # attention
        Hp = H
        Wp = W
        B_ = Hp * Wp / self.split_size / self.split_size
        flops += B_ * self.attn.flops(self.split_size * self.split_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


def faster_img2splits(img, split_size):
    """
    img: B H W C
    """
    B, H, W, C = img.shape
    img_reshape = img.view(B, H // split_size, split_size, W // split_size, split_size, C)
    # img_perm = img_reshape.permute(2, 4, 0, 1, 3, 5).contiguous().view(split_size, split_size, -1, C)
    img_perm = img_reshape.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, split_size, split_size, C)
    return img_perm


def faster_splits2img(img_splits_hw, split_size, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / split_size / split_size))
    # img = img_splits_hw.view(split_size, split_size, B, H // split_size, W // split_size, -1)
    # img = img.permute(2, 3, 0, 4, 1, 5).contiguous().view(B, H, W, -1)

    img = img_splits_hw.view(B, H // split_size, W // split_size, split_size, split_size, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class ConCatDownSample(nn.Module):
    def __init__(self, patches_resolution, channel, reduce_channel=True, norm_layer=None, act_layer=None):
        super().__init__()
        self.patches_resolution = patches_resolution
        self.reduce_channel = reduce_channel
        self.channel = channel
        self.channel_reduction = nn.Linear(4 * channel, int(reduce_channel * channel), bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(4 * channel)
        else:
            self.norm = None
        if act_layer is not None:
            self.act = act_layer()
        else:
            self.act = None

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        img = x.view(B, H, W, C)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            img = F.pad(img, (0, 0, 0, W % 2, 0, H % 2))
        img0 = img[:, 0::2, 0::2, :]  # B H/2 W/2 C
        img1 = img[:, 1::2, 0::2, :]  # B H/2 W/2 C
        img2 = img[:, 0::2, 1::2, :]  # B H/2 W/2 C
        img3 = img[:, 1::2, 1::2, :]  # B H/2 W/2 C
        img = torch.cat([img0, img1, img2, img3], -1)  # B H/2 W/2 4*C
        x = img.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        if self.norm is not None:
            x = self.norm(x)
        x = self.channel_reduction(x)
        if self.act is not None:
            x = self.act(x)

        return x

    def flops(self):
        H, W = self.patches_resolution
        flops = (H // 2) * (W // 2) * 4 * self.channel * self.reduce_channel * self.channel
        return flops


class BasicLayer(nn.Module):

    def __init__(self, dim, patches_resolution, depth, num_heads,
                 split_size=7, disturb=True, mlp_ratio=4., reduce_channel=2,
                 reduce_norm_layer=None, reduce_act_layer=None,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 rpe='none', rpe_separate=True, additional_dpe=False, split_version='v1',
                 use_checkpoint=False):
        super().__init__()
        self.patches_resolution = patches_resolution
        self.disturb = disturb
        self.split_size = split_size
        self.reduce_channel = reduce_channel
        self.reduce_norm_layer = reduce_norm_layer
        self.reduce_act_layer = reduce_act_layer
        self.rpe = rpe
        self.rpe_separate = rpe_separate
        self.additional_dpe = additional_dpe
        self.split_version = split_version
        self.use_checkpoint = use_checkpoint

        if self.additional_dpe:
            self.dpe_pos_embed = nn.Parameter(torch.zeros(1, patches_resolution[0] * patches_resolution[1], dim))
            trunc_normal_(self.dpe_pos_embed, std=.02)

        if split_version == 'faster':
            SlicedBlock = SlicedBlockFaster
        else:
            raise KeyError
        self.blocks = nn.ModuleList([
            SlicedBlock(dim=dim, patches_resolution=patches_resolution, num_heads=num_heads,
                        split_size=split_size,
                        disturb_size=0 if (i % 2 == 0 or not self.disturb) else self.split_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        rpe=rpe if (self.rpe_separate or i == 0) else 'none')
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(patches_resolution, channel=dim, reduce_channel=reduce_channel,
                                         norm_layer=reduce_norm_layer, act_layer=reduce_act_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """
        x: B H*W+1 C
        """
        if self.additional_dpe:
            x = x + self.dpe_pos_embed
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return 'patches_resolution={}, disturb={}, ' \
               'split_size={}, reduce_channel={}, ' \
               'reduce_norm_layer={}, reduce_act_layer={}, ' \
               'split_version={}'.format(self.patches_resolution, self.disturb, self.split_size,
                                         self.reduce_channel, self.reduce_norm_layer, self.reduce_act_layer,
                                         self.split_version)

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        return flops

@SwinTransformer.register_module
class SwinUnsup(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, num_classes=1000, img_size=224, in_chans=3,
                 patch_size=4, embed_dim=48, depths=[3, 3, 3, 3], num_heads=[3, 3, 6, 12],
                 split_size=7, disturb=True, mlp_ratio=4., reduce_channels=[2, 2, 2, 2],
                 reduce_norm=False, reduce_act=False,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., head_drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 rpe='none', dpe=True, init_func='trunc_normal',
                 rpe_stages=[0, 1, 2, 3], rpe_separate_stages=[0, 1, 2, 3],
                 additional_dpe_stages=[], split_version='v1',
                 use_checkpoint=False, linear_expand=None, patch_norm=False,
                 patch_fix=False, no_weight_decay_keys=[], linear_eval_head=False,
                 return_no_pool=False,
                 **kwargs):
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.reduce_channels = reduce_channels
        self.reduce_norm = reduce_norm
        self.reduce_act = reduce_act
        self.rpe = rpe
        self.rpe_stages = rpe_stages
        self.rpe_separate_stages = rpe_separate_stages
        self.additional_dpe_stages = additional_dpe_stages
        self.dpe = dpe
        self.init_func = init_func
        self.split_version = split_version
        self.use_checkpoint = use_checkpoint
        self.linear_expand = linear_expand
        self.patch_norm = patch_norm
        self.patch_fix = patch_fix
        self.no_weight_decay_keys = no_weight_decay_keys
        self.num_features = int(embed_dim * np.prod(reduce_channels[:-1]))
        self.depths = depths

        # dev
        self.linear_eval_head=linear_eval_head
        self.return_no_pool = return_no_pool

        if isinstance(mlp_ratio, list):
            assert len(mlp_ratio) == self.num_layers
            self.mlp_ratios = mlp_ratio
        elif isinstance(mlp_ratio, (float, int)):
            self.mlp_ratios = [mlp_ratio for _ in range(self.num_layers)]
        else:
            raise NotImplementedError

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        if self.patch_fix:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.dpe:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.head_drop = nn.Dropout(p=head_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        last_patches_resolution = patches_resolution
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * np.prod(reduce_channels[:i_layer])),
                patches_resolution=[last_patches_resolution[0],
                                    last_patches_resolution[1]],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                split_size=split_size,
                disturb=disturb,
                mlp_ratio=self.mlp_ratios[i_layer],
                reduce_channel=reduce_channels[i_layer],
                reduce_norm_layer=norm_layer if self.reduce_norm else None,
                reduce_act_layer=act_layer if self.reduce_act else None,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=ConCatDownSample if (i_layer < self.num_layers - 1) else None,
                rpe=self.rpe if i_layer in self.rpe_stages else 'none',
                rpe_separate=True if i_layer in self.rpe_separate_stages else False,
                additional_dpe=True if i_layer in self.additional_dpe_stages else False,
                split_version=split_version,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            last_patches_resolution[0] = int(np.ceil(last_patches_resolution[0] / 2))
            last_patches_resolution[1] = int(np.ceil(last_patches_resolution[1] / 2))
        self.norm = norm_layer(self.num_features)

        # Classifier head
        # if self.linear_expand is not None:
        #     self.linear_expand_head = nn.Linear(self.num_features, self.num_features * self.linear_expand)
        #     self.head = nn.Linear(self.num_features * self.linear_expand, num_classes) if num_classes > 0 \
        #         else nn.Identity()
        # else:
        #     self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if self.linear_eval_head:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_func == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif self.init_func == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif self.init_func == 'trunc_normal':
                trunc_normal_(m.weight, std=.02)
            else:
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        ret = set()
        for key in self.no_weight_decay_keys:
            ret.add(key)
        ret.add('pos_embed')
        return ret

    @torch.jit.ignore
    def no_weight_decay_pattern(self):
        return 'pos_embed'

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rel_pos_embed_table'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.dpe:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C

        if self.return_no_pool:
            x = x.transpose(1, 2)  # B C L
            B, C, _ = x.shape
            x = x.reshape(B, C, 7, 7)  # todo: fix hack
            return x

        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # if self.linear_expand is not None:
        #     x = self.linear_expand_head(x)
        # x = self.head_drop(x)
        if self.linear_eval_head:
            x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        if dist.get_rank() == 0:
            print(f"GFLOPs patch_embed: {self.patch_embed.flops() / 1e9}")
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
            if dist.get_rank() == 0:
                print(f"GFLOPs layer_{i}: {layer.flops() / 1e9}")
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        if dist.get_rank() == 0:
            print(f"GFLOPs MLP: {self.num_features * self.num_classes / 1e9}")
        return flops
