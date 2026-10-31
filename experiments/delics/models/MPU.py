import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from functools import partial
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

# 核心依赖：从本地 mamba.py 导入
from .mamba import Mamba

# ==============================================================================
# Section 1: 复用的核心组件
# ==============================================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, to_2tuple(window_size), num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        coords_h, coords_w = torch.arange(self.window_size[0]), torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop, self.proj, self.proj_drop = nn.Dropout(attn_drop), nn.Linear(dim, dim), nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]; attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0); attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C); x = self.proj(x); x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim, self.num_heads, self.window_size, self.shift_size = dim, num_heads, window_size, shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp_hidden_dim, dim), nn.Dropout(drop))

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_mask = None
        if self.shift_size > 0:
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device); cnt = 0
            h_slices, w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)), (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            for h in h_slices:
                for w in w_slices: img_mask[:, h, w, :] = cnt; cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C), self.window_size, Hp, Wp)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        if pad_r > 0 or pad_b > 0: x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SwinLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop_path=[0., 0.]):
        super().__init__()
        self.block1 = SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path[0])
        self.block2 = SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=drop_path[1])
    def forward(self, x):
        B, C, H, W = x.shape; x_3d = x.flatten(2).transpose(1, 2)
        x_3d = self.block1(x_3d, H, W); x_3d = self.block2(x_3d, H, W)
        return x_3d.transpose(1, 2).view(B, C, H, W)

class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if layer_norm_type == 'BiasFree': self.body = self._BiasFree_LayerNorm(dim)
        else: self.body = self._WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]; return self._to_4d(self.body(self._to_3d(x)), h, w)
    def _to_3d(self, x): return rearrange(x, 'b c h w -> b (h w) c')
    def _to_4d(self, x, h, w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    class _BiasFree_LayerNorm(nn.Module):
        def __init__(self, normalized_shape): super().__init__(); self.weight = nn.Parameter(torch.ones(normalized_shape))
        def forward(self, x): sigma = x.var(-1, keepdim=True, unbiased=False); return x / torch.sqrt(sigma + 1e-5) * self.weight
    class _WithBias_LayerNorm(nn.Module):
        def __init__(self, normalized_shape): super().__init__(); self.weight = nn.Parameter(torch.ones(normalized_shape)); self.bias = nn.Parameter(torch.zeros(normalized_shape))
        def forward(self, x): mu = x.mean(-1, keepdim=True); sigma = x.var(-1, keepdim=True, unbiased=False); return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class TemporalMambaBlock(nn.Module):
    def __init__(self, dim, num_groups=6, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.dim, self.num_groups = dim, num_groups
        if dim % num_groups != 0: raise ValueError(f"维度 ({dim}) 必须能被组数 ({num_groups}) 整除")
        self.group_len = dim // num_groups
        self.mamba_scanner = Mamba(d_model=self.group_len, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = LayerNorm(dim, 'WithBias')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        B, D, H, W = x.shape; residual = x
        x_norm = self.norm(x)
        x_permuted = x_norm.permute(0, 2, 3, 1); x_pixel_batches = x_permuted.reshape(B * H * W, D)
        x_grouped = x_pixel_batches.view(B * H * W, self.num_groups, self.group_len)
        x_mamba_input = x_grouped.transpose(1, 2).unsqueeze(-1)
        mamba_out_4d = self.mamba_scanner(x_mamba_input)
        mamba_out_squeezed = mamba_out_4d.squeeze(-1); mamba_out_ungrouped = mamba_out_squeezed.transpose(1, 2)
        x_un_reshaped = mamba_out_ungrouped.reshape(B * H * W, D); x_restored_permuted = x_un_reshaped.view(B, H, W, D)
        out = x_restored_permuted.permute(0, 3, 1, 2)
        return residual + self.drop_path(out)

class SimpleResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        # 修复：将 GroupNorm 设为 1 组（即 LayerNorm 效果），解决维度整除问题
        self.norm = nn.GroupNorm(1, dim) 
    def forward(self, x):
        return x + self.conv2(self.act(self.norm(self.conv1(x))))

# ==============================================================================
# Section 2: 新架构核心模块
# ==============================================================================

class ManifoldProjectionUnit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in * 3, dim_out, kernel_size=1, bias=False),
            nn.GroupNorm(1, dim_out), # 修复：使用更通用的 GroupNorm
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x_mamba, x_swin, x_cnn):
        x_cat = torch.cat([x_mamba, x_swin, x_cnn], dim=1)
        return self.proj(x_cat)

class CascadeStage(nn.Module):
    def __init__(self, dim_in, dim_out, swin_heads=6, mamba_groups=6, window_size=8, d_state=16, drop_path=0.):
        super().__init__()
        self.branch_swin = SwinLayer(dim=dim_in, num_heads=swin_heads, window_size=window_size, drop_path=[drop_path, drop_path])
        self.branch_mamba = TemporalMambaBlock(dim=dim_in, num_groups=mamba_groups, d_state=d_state, drop_path=drop_path)
        self.branch_cnn = SimpleResBlock(dim=dim_in)
        self.mpu = ManifoldProjectionUnit(dim_in, dim_out)
    def forward(self, x):
        f_swin = self.branch_swin(x)
        f_mamba = self.branch_mamba(x)
        f_cnn = self.branch_cnn(x)
        return self.mpu(f_mamba, f_swin, f_cnn)

# ==============================================================================
# Section 3: 主网络 CSBNet
# ==============================================================================

class CSBNet(nn.Module):
    def __init__(self,
                 in_chans=400,
                 num_out_chans=2,
                 stage_dims=[180, 60, 20, 10], 
                 stage_heads=[6, 6, 4, 2],
                 stage_groups=[6, 6, 4, 2],
                 window_size=8,
                 d_state=16,
                 drop_path_rate=0.1,
                 use_checkpoint=True,
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv_first = nn.Conv2d(in_chans, stage_dims[0], 3, 1, 1)
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(stage_dims))]
        
        for i in range(len(stage_dims)):
            dim_in = stage_dims[i]
            # 确定降维目标
            dim_out = stage_dims[i+1] if i < len(stage_dims)-1 else stage_dims[-1]
            
            self.stages.append(
                CascadeStage(
                    dim_in=dim_in, dim_out=dim_out,
                    swin_heads=stage_heads[i], mamba_groups=stage_groups[i],
                    window_size=window_size, d_state=d_state, drop_path=dpr[i]
                )
            )
            
        self.head = nn.Sequential(
            nn.Conv2d(stage_dims[-1], stage_dims[-1], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(stage_dims[-1], num_out_chans, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv_first(x)
        for stage in self.stages:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return self.head(x)
