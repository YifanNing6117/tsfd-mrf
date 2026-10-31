# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from scipy import ndimage
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from . import transformer_configs as configs


logger = logging.getLogger(__name__)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-L_16': configs.get_l16_config(),
    'Res-ViT-B_16': configs.get_resvit_b16_config(),
    'Res-ViT-L_16': configs.get_resvit_l16_config(),
    'Swin_16': configs.get_swin_16_config(),  ### wyz
}



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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # print(x.shape)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  # (B, H, W, C) -> (B, H//P, P, W//P, P, C)
    
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)  # (B, H//P, P, W//P, P, C) -> (B, H//P, W//P, P, P, C) -> (B*(H*W)//(P*P), P, P, C)
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
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)  # (B*(H*W)//(P*P), P, P, C) -> (B, H//P, W//P, P, P, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  # (B, H//P, W//P, P, P, C) -> (B, H//P, P, W//P, P, C) -> (B, H, W, C)
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
        self.dim = dim                  # embedding dims = 180
        self.window_size = window_size  # Wh, Ww = P, P
        self.num_heads = num_heads      # num heads = 6
        head_dim = dim // num_heads     # head dims = 180//6 = 30
        self.scale = qk_scale or head_dim ** -0.5
        

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # print(self.relative_position_bias_table.shape)
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
        # print(relative_position_index.shape)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_c = nn.Linear(dim*2, dim * 6, bias=qkv_bias)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2 , bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.proj_x = nn.Linear(dim, dim, bias=qkv_bias)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # B_, N, C = x.shape  # B_: 576 number of Windows * Batch_size in a GPU  N: 64 patch number in a window  C: 180 embedding channel
#        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#        q,k,v (576,6,64,30) (number of Windows * Batch_size in a GPU, number of head, patch number in a window, head_dim)

        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # print("y.shape = ", y.shape)
        # x.shape =  torch.Size([256, 16, 256])
        # y.shape =  torch.Size([256, 16, 256])

        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("self.dim = ", self.dim)
        # self.dim =  256

       
        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("x_y.shape = ", x.shape) 
        # 1024 64 64
        # x_y.shape =  torch.Size([256, 16, 512])

        B_c, N_c, C_c = x.shape

        # qkv_x_y = self.qkv_c(x_y)
        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("qkv_x_y.shape = ", qkv_x_y.shape)
        # qkv_x_y.shape =  torch.Size([256, 16, 1536])

        qkv = self.qkv(x).reshape(B_c, N_c, 3, self.num_heads, C_c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # print("q.shape = ", qkv.shape)
        # print("k.shape = ", k.shape)
        # print("v.shape = ", v.shape)
        # q.shape =  torch.Size([1024, 4, 64, 16])
        # k.shape =  torch.Size([1024, 4, 64, 16])
        # v.shape =  torch.Size([1024, 4, 64, 16])


        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("attn.shape = ", attn.shape)
        # attn.shape =  torch.Size([256, 4, 16, 16])


        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("relative_position_bias.shape = ", relative_position_bias.shape)         
        # relative_position_bias.shape =  torch.Size([16, 16, 4])


        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("relative_position_bias.shape = ", relative_position_bias.shape)    
        # relative_position_bias.shape =  torch.Size([4, 16, 16])

        attn = attn + relative_position_bias.unsqueeze(0)

        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("attn.shape = ", attn.shape)
        # attn.shape =  torch.Size([256, 4, 16, 16])

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_c, N_c, C_c)
        
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([256, 16, 512])

        x = self.proj_x(x)
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("x.shape = ", x.shape)        
        # x.shape =  torch.Size([256, 16, 256])

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

    def params(self):
        # calculate params for 1 window with token length of N
        params = 0
        # qkv = self.qkv(x)
        params += self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        params += 0
        #  x = (attn @ v)
        params += 0
        # x = self.proj(x)
        params += self.dim * self.dim
        return params

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # print(shift_size)
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
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # print(dim)
        # print(mlp_hidden_dim)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
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

        return attn_mask

    def forward(self, x, x_size):
        
        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 256, 1024])

        H, W = x_size
        
        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("H = ", H)
        # print("W = ", W)
        # H =  16
        # W =  16

        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("self.input_resolution = ", self.input_resolution)
        # self.input_resolution =  (1, 1)

        
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut_x = x
        # shortcut_y = y

#########################################################################################################
#########################################################################################################
        # x = torch.cat([x, y], 1)
        
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 512, 1024])

        x = self.norm1(x)
        x = x.view(B, H, W, C) 

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("shifted_x.shape = ", shifted_x.shape)
        # shifted_x.shape =  torch.Size([1, 256 ,256 ,32])


        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("self.window_size = ", self.window_size)
        # self.window_size =  1

#########################################################################################################
        # y = self.norm1(y)
        # y = y.view(B, H, W, C) 

        # # cyclic shift
        # if self.shift_size > 0:
        #     shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # else:
        #     shifted_y = y

#########################################################################################################

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)

        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("x_windows.shape = ", x_windows.shape)
        # print(shifted_x.shape)
        # x_windows.shape =  torch.Size([1024, 8, 8, 32])

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("x_windows.shape = ", x_windows.shape) 
        # x_windows.shape =  torch.Size([1024, 64, 32])


        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("self.input_resolution = ", self.input_resolution)
        # print("x_size = ", x_size)
        # self.input_resolution =  (1, 1)
        # x_size =  (16, 16)

#########################################################################################################

        # y_windows = window_partition(shifted_y, self.window_size)
        # y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("y_windows.shape = ", y_windows.shape)
        # y_windows.shape =  torch.Size([256, 1, 1024])

#########################################################################################################

        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("self.input_resolution = ", self.input_resolution)
        # print("x_size = ", x_size)
        # self.input_resolution =  (1, 1)
        # x_size =  (16, 16)


        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("self.attn_mask = ", self.attn_mask.size)

        # orginal_1, original_2, original_3 = x_windows.shape

        # x_windows = torch.cat([x_windows, y_windows], 0)        
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("x_windows = ", x_windows.shape)
        # x_windows =  torch.Size([512, 1, 1024])


        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # 这条语句没执行
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))  # attn_windows.shape = (576,64,180) = (nW*B, window_size*window_size, C) 

        
        
        # cat后
        # attn_windows.shape =  torch.Size([1024, 64, 32])
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("attn_windows.shape = ", attn_windows.shape)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # attn_windows.shape = (576,8,8,180) = (nW*B, window_size, window_size, C)  nW:number of Windows
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B, H', W', C shifted_x (4,96,96,180) (batch_in_each_GPU, embedding_channel, H, W)  ???????

         # cat后
        # attn_windows.shape =  torch.Size([512, 1, 1024])
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("attn_windows.shape = ", attn_windows.shape)1024 8 8 32
        # print(shifted_x.shape)1 256 256 32 


        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)  # x (4,9216,180) (batch_in_each_GPU, H*W, embedding_channel) x_size (96,96)

#########################################################################################################
#########################################################################################################
        
        # if self.input_resolution == x_size:
        #     attn_windows_y = self.attn(y_windows, x_windows, mask=self.attn_mask)  # 这条语句没执行
        # else:
        #     attn_windows_y = self.attn(y_windows, x_windows, mask=self.calculate_mask(x_size).to(x.device))  # attn_windows.shape = (576,64,180) = (nW*B, window_size*window_size, C) 

        # cat前
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("attn_windows.shape = ", attn_windows.shape)
        # attn_windows.shape =  torch.Size([256, 1, 1024])
        
        # cat后
        # attn_windows.shape =  torch.Size([512, 1, 1024])

        # attn_windows = attn_windows.reshape(orginal_1, original_2, original_3)
        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("attn_windows.shape = ", attn_windows.shape)

        # merge windows
        # attn_windows_y = attn_windows_y.view(-1, self.window_size, self.window_size, C)  # attn_windows.shape = (576,8,8,180) = (nW*B, window_size, window_size, C)  nW:number of Windows
        # shifted_y = window_reverse(attn_windows_y, self.window_size, H, W)  # B, H', W', C shifted_x (4,96,96,180) (batch_in_each_GPU, embedding_channel, H, W)  ???????

        # reverse cyclic shift
        # if self.shift_size > 0:
        #     y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # else:
        #     y = shifted_y
        # y = y.view(B, H * W, C)  # x (4,9216,180) (batch_in_each_GPU, H*W, embedding_channel) x_size (96,96)



#########################################################################################################
#########################################################################################################


        # FFN
        # print(x.shape)
        x = shortcut_x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print(x.shape)
        # y = shortcut_y + self.drop_path(y)
        # y = y + self.drop_path(self.mlp(self.norm2(y)))


        return x  # x.shape = (4,9216,180) = (batch_in_each_GPU, H*W, embedding_channel)   96*96=9216

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

    def params(self):
        params = 0
        # norm1
        params += self.dim * 2
        # W-MSA/SW-MSA
        params += self.attn.params()
        # mlp
        params += 2 * self.dim * self.dim * self.mlp_ratio
        # norm2
        params += self.dim * 2
        return params

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # operate in the last embedding dim (embedding_channel) by default
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape                    # B     L      C
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)               # B  H    W    C

        x0 = x[:, 0::2, 0::2, :]             # B  H/2  W/2  C
        x1 = x[:, 1::2, 0::2, :]             # B  H/2  W/2  C
        x2 = x[:, 0::2, 1::2, :]             # B  H/2  W/2  C
        x3 = x[:, 1::2, 1::2, :]             # B  H/2  W/2  C
        x = torch.cat([x0, x1, x2, x3], -1)  # B  H/2  W/2  C*4
        x = x.view(B, -1, 4 * C)             # B  H/2*W/2   C*4

        x = self.norm(x)
        x = self.reduction(x)                # B  H/2*W/2   C*2

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

    def params(self):
        params = 2 * self.dim
        params += 4 * self.dim * 2 * self.dim
        return params

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
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

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.window_size_basiclayer = window_size

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        
        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("self.window_size_basiclayer = ", self.window_size_basiclayer)
        # self.window_size_basiclayer =  8


        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("self.input_resolution = ", self.input_resolution)
        # self.input_resolution =  (1, 1)


        for blk in self.blocks:
            if self.use_checkpoint:
                x= checkpoint.checkpoint(blk, x, x_size)
            else:
                x= blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
            # y = self.downsample(y)
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

    def params(self):
        params = 0
        for blk in self.blocks:
            params += blk.params()
        if self.downsample is not None:
            params += self.downsample.params()
        return params

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.window_size_rstb = window_size

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':  # =============================================================== default to '1conv'
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):

        # print('\033[1;31m---residual_swin_transformers_multi_modal.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 256, 1024])

        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("self.window_size_rstb = ", self.window_size_rstb)
        # self.window_size_rstb =  8

        # print('\033[1;31m---residual_swin_transformers.py---\033[0m')
        # print("self.input_resolution = ", self.input_resolution)
        # self.input_resolution =  (1, 1)
        x_rstb = self.residual_group(x, x_size)
        # print(x_rstb.shape)

        x_rstb = self.patch_embed(self.conv(self.patch_unembed(x_rstb, x_size))) + x
        # y_rstb = self.patch_embed(self.conv(self.patch_unembed(y_rstb, x_size))) + y
        return x_rstb

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

    def params(self):
        params = 0
        params += self.residual_group.params()
        params += self.dim * self.dim * 9
        params += self.patch_embed.params()
        params += self.patch_unembed.params()

        return params

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

    def params(self):
        params = 0
        if self.norm is not None:
            params += 2 * self.embed_dim
        return params

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

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
        # print(self.num_patches)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

    def params(self):
        params = 0
        return params

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

    def params(self):
        params = self.num_feat * 3 * 9
        return params

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Swin_multi_modal_block(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=128, depths=[4], num_heads=[4], window_size=8, 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 ape=True, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv', **kwargs):
        super(Swin_multi_modal_block, self).__init__()

        num_in_ch = in_chans  # 1 (gray)
        num_out_ch = in_chans  # 1 (gray)
        num_feat = 512
        self.img_range = img_range  # 1.0

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)


        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean_y = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean_y = torch.zeros(1, 1, 1, 1)


        self.upscale = upscale  # 1 (without pixelshuffle)
        self.upsampler = upsampler  # None
        self.window_size = window_size  # 8

        # 1. shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)  # num_in_ch = 1, embed_dim = 180

        # 2. deep feature extraction 
        self.num_layers = len(depths)  # 6, depths = [6,6,6,6,6,6]
        self.embed_dim = embed_dim  # 180
        self.ape = ape  # False
        self.patch_norm = patch_norm  # True
        self.num_features = embed_dim  # 180
        self.mlp_ratio = mlp_ratio  # 2

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  # num_patcher = 65536
        patches_resolution = self.patch_embed.patches_resolution  # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding (ape)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # print(num_patches)
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # dropout after ape
        self.pos_drop = nn.Dropout(p=drop_rate)
        # print(drop_rate)

        # stochastic depth - randomly drop some layers when training - depths = [6,6,6,6,6,6]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # self.num_layers = 6
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],  # num_heads = [6,6,6,6,6,6]
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, 
                         qk_scale=qk_scale,
                         drop=drop_rate, 
                         attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        # ======================================================================================================================================= default to '1conv'
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        # =======================================================================================================================================
        elif resi_connection == '3conv': 
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))


        ######################################################################################################
        ################################ 3, high quality image reconstruction ################################
        ######################################################################################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # ======================================================================================================================================= default to None
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)  
        # =======================================================================================================================================
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

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

#######################################################################################################
#######################################################################################################

        # x.shape =  torch.Size([1, 256, 64, 64])
        # x.shape =  torch.Size([1, 4096, 256])

#######################################################################################################
#######################################################################################################

        
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 32, 256, 256])
        
        # 1. image to patch embedding
        x = self.patch_embed(x)

        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 65536, 32])

        # y = self.patch_embed(y)
        # print(111111111111111)
        # 2. absolute position embedding (ape) 
        # print(self.absolute_pos_embed.shape)
        if self.ape:  
            x = x + self.absolute_pos_embed
            # print(1111111111111111111111)

        x = x + self.absolute_pos_embed
        # y = y + self.absolute_pos_embed
        # if self.ape:  
        #     y = y + self.absolute_pos_embed
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 65536, 32])

        # 3. dropout
        x = self.pos_drop(x)

        # y = self.pos_drop(y)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 65536, 32])

        # 4. RSTBs (main params and computes) ####################
        for layer in self.layers:
            x= layer(x, x_size)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 65536, 32])
        # 5. norm
        x = self.norm(x)

        # y = self.norm(y)
        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # 6. patch embedding to image
        x = self.patch_unembed(x, x_size)

        # y = self.patch_unembed(y, x_size)

        # print('\033[1;31m---residual_transformers.py---\033[0m')
        # print("x.shape = ", x.shape)
        # x.shape =  torch.Size([1, 32, 256, 256])
        return x

    def forward(self, x):
        H, W = x.shape[2:]                 # x (4,1,96,96) (batch_size_in_each_GPU, input_image_channel, H (random-crop 96 in traning and 256 in testing), W) 

        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # y = self.check_image_size(y)
        # self.mean_y = self.mean_y.type_as(y)
        # y = (y - self.mean_y) * self.img_range
        
        x= self.forward_features(x)

        
            # x = self.conv_after_body(self.forward_features(x)) + x
            # x = x + self.conv_last(res)
        # =======================================================================================================================================

        x = x / self.img_range + self.mean
        # y = y / self.img_range + self.mean_y

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 1 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * self.embed_dim * self.embed_dim * 9
        flops += H * W * self.embed_dim * 1 * 9
        return flops

    def params(self):
        params = 0
        params += 1 * self.embed_dim * 9
        params += self.patch_embed.params()
        for i, layer in enumerate(self.layers):
            params += layer.params()
        params += self.embed_dim * self.embed_dim * 9
        params += self.embed_dim * 1 * 9
        return params