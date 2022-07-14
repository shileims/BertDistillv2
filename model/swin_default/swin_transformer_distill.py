# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.distributed as dist

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class PastMlpBlock(nn.Module):
    # def __init__(self, in_features, hidden_features=None, layers_per_block=2, act_layer=nn.GELU,
    #              norm_layer=nn.SyncBatchNorm):
    def __init__(self, in_features, hidden_features=None, layers_per_block=2, act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm1d, is_dist=False):
        super().__init__()
        # if is_dist:
        #     norm_layer = nn.SyncBatchNorm
        self.layers_per_block = layers_per_block
        # out_features = in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fcList = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features) for _ in range(int(layers_per_block - 1))])
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bnList = nn.ModuleList([norm_layer(hidden_features) for _ in range(int(layers_per_block))])
        # self.bn1 = norm_layer(hidden_features)
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        # self.bn2 = norm_layer(out_features)

    def forward(self, x):
        short_cut = x
        x = self.fc1(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.bnList[0](x)
        x = x.permute(0, 2, 1)
        for i in range(int(self.layers_per_block - 1)):
            x = self.fcList[i](x)
            x = self.act(x)
            x = x.permute(0, 2, 1)
            x = self.bnList[i + 1](x)
            x = x.permute(0, 2, 1)
            # x = self.fc2(x)
        # x = x + short_cut
        # x = torch.cat((x,short_cut), dim = 2)
        return x


class PastMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm1d,
                 num_blocks=1, layers_per_block=2, is_dist=False):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([PastMlpBlock(
            in_features=in_features,
            hidden_features=hidden_features,
            layers_per_block=layers_per_block,
            act_layer=act_layer,
            norm_layer=norm_layer,
            is_dist=is_dist
        ) for i in range(num_blocks)])

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
        return x


class Mlp(nn.Module):
    # def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
    #              norm_layer=nn.SyncBatchNorm):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=nn.BatchNorm1d, is_dist=False):
        super().__init__()
        if is_dist:
            norm_layer = nn.SyncBatchNorm
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = norm_layer(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.bn2 = norm_layer(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        # self.local_init_bias()
        self.softmax = nn.Softmax(dim=-1)

    def local_init_bias(self):
        l = self.relative_position_bias_table.shape[0]
        for x, k in enumerate(self.relative_position_bias_table):
            stdd = np.sin(np.pi / (l - 1) * x) * (np.cos(2 * np.pi / 7 * (x + 1)) + 1) / 20
            trunc_normal_(k, std=stdd + 1e-5)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
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


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_dist=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.is_dist = is_dist
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, is_dist=self.is_dist)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        output_dim (int): Output dimention
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, output_dim=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.o_dim = 2 * dim if output_dim is None else output_dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, self.o_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * self.o_dim
        return flops


class ChannelRaiser(nn.Module):
    r""" Channel raising Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        output_dim (int): Output dimention
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, output_dim=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.o_dim = 2 * dim if output_dim is None else output_dim
        self.dim = dim
        self.raiser = nn.Linear(dim, self.o_dim, bias=False)
        self.norm = norm_layer(self.dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = self.norm(x)
        x = self.raiser(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim * self.o_dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        output_dim (int): output dimention of the basic layer, by default it will be 2*dim
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, output_dim=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, act=nn.GELU, is_dist=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.is_dist = is_dist
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 act_layer=act,
                                 is_dist=is_dist)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, output_dim=output_dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
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
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int | tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        output_dim (tuple | None): the output dimension of the layers. By default (None) it will be 2*dim
        change_channel (tuple | None): tuple of bool
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, output_dim=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, double_output=False, act=None,
                 pastmlp_blocks=1, pastmlp_layers=2, pastmlp_ratio=4.0, is_dist=False, **kwargs):
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) if output_dim is None else output_dim[-1]

        self.double_output = double_output
        assert isinstance(window_size, int) or isinstance(window_size, list) or isinstance(window_size,
                                                                                           tuple), "window size should be int or turple"
        assert isinstance(mlp_ratio, float) or isinstance(mlp_ratio, list) or isinstance(mlp_ratio,
                                                                                         tuple), "window size should be float or turple"
        self.window_size = [window_size for _ in range(self.num_layers)] if isinstance(window_size,
                                                                                       int) else window_size
        self.mlp_ratio = [mlp_ratio for _ in range(self.num_layers)] if isinstance(mlp_ratio, float) else mlp_ratio
        assert output_dim is None or len(
            output_dim) == self.num_layers, F"number of output_dim should be the same as the layers number: {self.num_layers}"

        change_channel = [(224 / patch_size) % (2 ** (k + 1)) == 0 for k in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.act = [nn.GELU for _ in range(self.num_layers)] if act is None else act

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dsample = None
            if output_dim is None:
                dsample = PatchMerging if change_channel[i_layer] else None
            elif i_layer == 0 or not output_dim[i_layer - 1] == output_dim[i_layer]:
                dsample = PatchMerging if change_channel[i_layer] else ChannelRaiser
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer) if i_layer == 0 or output_dim is None else output_dim[i_layer - 1],
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size[i_layer],
                output_dim=None if output_dim is None else output_dim[i_layer],
                mlp_ratio=self.mlp_ratio[i_layer],
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=dsample,
                use_checkpoint=use_checkpoint,
                act=self.act[i_layer],
                is_dist=is_dist)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.PastMLPs = PastMLP(self.num_features, hidden_features=int(pastmlp_ratio * self.num_features),
                                num_blocks=pastmlp_blocks, layers_per_block=pastmlp_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.pastmlp_ratio = pastmlp_ratio
        self.head = nn.Linear(int(pastmlp_ratio * self.num_features), num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        B, L, C = x.shape
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = x.view(B, C*L, 1)
        x = x.transpose(1, 2)
        x = self.PastMLPs(x)
        x = x.transpose(1, 2)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        if self.double_output and self.training:
            return x, x
        else:
            return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


"""
Latency (ms)
mini1 23.46
mini2 19.12
mini3 14.97
mini4 13.84
mini5 11.69
mini6 11.18
mini7 8.50
"""


def build_mini_model(model_type, num_classes, use_checkpoint=True, is_dist=False):
    if model_type == 'swin_mini1':
        model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=24,
                                depths=[2, 2, 2, 6],
                                num_heads=[2, 4, 8, 16],
                                window_size=[7, 7, 7, 7],
                                output_dim=[48, 96, 192, 480],
                                mlp_ratio=[2, 2, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=use_checkpoint,
                                double_output=False,
                                half_mlp=False,
                                low_rank_mlp=None,
                                pastmlp_blocks=1,
                                pastmlp_layers=1,
                                pastmlp_ratio=2.5,
                                is_dist=is_dist)
    elif model_type == 'swin_mini2':
        model = SwinTransformer(img_size=192,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=24,
                                depths=[2, 2, 2, 6],
                                num_heads=[2, 4, 8, 16],
                                window_size=[6, 6, 6, 6],
                                output_dim=[48, 96, 192, 480],
                                mlp_ratio=[2, 2, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=use_checkpoint,
                                double_output=False,
                                half_mlp=False,
                                low_rank_mlp=None,
                                pastmlp_blocks=1,
                                pastmlp_layers=1,
                                pastmlp_ratio=2.5,
                                is_dist=is_dist)
    elif model_type == 'swin_mini3':
        model = SwinTransformer(img_size=192,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=24,
                                depths=[1, 1, 2, 4],
                                num_heads=[2, 4, 8, 16],
                                window_size=[6, 6, 6, 6],
                                output_dim=[48, 96, 192, 480],
                                mlp_ratio=[2, 2, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=use_checkpoint,
                                double_output=False,
                                half_mlp=False,
                                low_rank_mlp=None,
                                pastmlp_blocks=1,
                                pastmlp_layers=1,
                                pastmlp_ratio=2.5,
                                is_dist=is_dist)
    elif model_type == 'swin_mini4':
        model = SwinTransformer(img_size=192,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=24,
                                depths=[1, 1, 2, 2],
                                num_heads=[2, 4, 8, 16],
                                window_size=[6, 6, 6, 6],
                                output_dim=[48, 96, 192, 480],
                                mlp_ratio=[2, 2, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=use_checkpoint,
                                double_output=False,
                                half_mlp=False,
                                low_rank_mlp=None,
                                pastmlp_blocks=1,
                                pastmlp_layers=1,
                                pastmlp_ratio=2.5,
                                is_dist=is_dist)
    elif model_type == 'swin_mini5':
        model = SwinTransformer(img_size=160,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=24,
                                depths=[1, 1, 2, 4],
                                num_heads=[2, 4, 8, 16],
                                window_size=[5, 5, 5, 5],
                                output_dim=[48, 96, 192, 480],
                                mlp_ratio=[2, 2, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=use_checkpoint,
                                double_output=False,
                                half_mlp=False,
                                low_rank_mlp=None,
                                pastmlp_blocks=1,
                                pastmlp_layers=1,
                                pastmlp_ratio=2.5,
                                is_dist=is_dist)
    elif model_type == 'swin_mini6':
        model = SwinTransformer(img_size=160,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=24,
                                depths=[1, 1, 2, 2],
                                num_heads=[2, 4, 8, 16],
                                window_size=[5, 5, 5, 5],
                                output_dim=[48, 96, 192, 480],
                                mlp_ratio=[2, 2, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=use_checkpoint,
                                double_output=False,
                                half_mlp=False,
                                low_rank_mlp=None,
                                pastmlp_blocks=1,
                                pastmlp_layers=1,
                                pastmlp_ratio=2.5,
                                is_dist=is_dist)
    elif model_type == 'swin_mini7':
        model = SwinTransformer(img_size=160,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=16,
                                depths=[1, 1, 2, 2],
                                num_heads=[2, 4, 8, 16],
                                window_size=[5, 5, 5, 5],
                                output_dim=[32, 48, 96, 320],
                                mlp_ratio=[2, 2, 4, 4],
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=use_checkpoint,
                                double_output=False,
                                half_mlp=False,
                                low_rank_mlp=None,
                                pastmlp_blocks=1,
                                pastmlp_layers=1,
                                pastmlp_ratio=2.5,
                                is_dist=is_dist)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model