# ==============================================================================
# InterleavedMambaMRF.py - 最终完整版
#
# 这份代码是经过最终审查的生产版本，包含了所有类的完整实现，
# 并确保了设计的灵活性、正确性和严谨性。
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from functools import partial
from typing import Callable

from einops import rearrange, repeat
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

# 核心依赖
from .mamba import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# ==============================================================================
# Section 1: 核心构建单元 (完整实现)
# ==============================================================================

# --- 组件 1.1: 空间 Mamba 构建单元 ---

class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2., dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, d_state, d_conv, expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner, bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()
        self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs))
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        self.dt_projs = (self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs), self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs), self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs), self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs))
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs, self.Ds = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True), self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan, self.out_norm, self.out_proj = selective_scan_fn, nn.LayerNorm(self.d_inner), nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant": nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random": nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad(): dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge: A_log = A_log.flatten(0, 1)
        return nn.Parameter(A_log)
    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge: D = D.flatten(0, 1)
        return nn.Parameter(D)
    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape; L = H * W; K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, 2, 3).contiguous().view(B, -1, L)], 1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], 1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs, dts, Bs, Cs, Ds, As, dt_projs_bias = xs.float().view(B, -1, L), dts.contiguous().float().view(B, -1, L), Bs.float().view(B, K, -1, L), Cs.float().view(B, K, -1, L), self.Ds.float().view(-1), -torch.exp(self.A_logs.float()).view(-1, self.d_state), self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, K, -1, L)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y, invwh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L), torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = self.out_norm(y.transpose(1, 2).contiguous().view(B, H, W, -1))
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None: out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim, drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), d_state=16):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x_seq, x_size):
        B, L, C = x_seq.shape
        shortcut = x_seq
        x_img_like = x_seq.view(B, *x_size, C)
        x = self.ln_1(x_img_like)
        attended_x = self.self_attention(x)
        attended_x = attended_x.view(B, L, C)
        return shortcut + self.drop_path(attended_x)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=1, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x_seq, x_size):
        B, L, C = x_seq.shape
        H, W = x_size
        return x_seq.transpose(1, 2).view(B, C, H, W)

class SpatialMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, drop_path=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=dim, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.vss_block = VSSBlock(hidden_dim=dim, d_state=d_state, drop_path=drop_path)
    def forward(self, x):
        B, C, H, W = x.shape
        x_size = (H, W)
        x_seq = self.patch_embed(x)
        x_seq = self.vss_block(x_seq, x_size)
        out = self.patch_unembed(x_seq, x_size)
        return out

# --- 组件 1.2: 时间 Mamba 构建单元 ---
class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if layer_norm_type == 'BiasFree': self.body = self._BiasFree_LayerNorm(dim)
        else: self.body = self._WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return self._to_4d(self.body(self._to_3d(x)), h, w)
    def _to_3d(self, x): return rearrange(x, 'b c h w -> b (h w) c')
    def _to_4d(self, x, h, w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    class _BiasFree_LayerNorm(nn.Module):
        def __init__(self, normalized_shape):
            super().__init__(); self.weight = nn.Parameter(torch.ones(normalized_shape))
        def forward(self, x):
            sigma = x.var(-1, keepdim=True, unbiased=False); return x / torch.sqrt(sigma + 1e-5) * self.weight
    class _WithBias_LayerNorm(nn.Module):
        def __init__(self, normalized_shape):
            super().__init__(); self.weight = nn.Parameter(torch.ones(normalized_shape)); self.bias = nn.Parameter(torch.zeros(normalized_shape))
        def forward(self, x):
            mu = x.mean(-1, keepdim=True); sigma = x.var(-1, keepdim=True, unbiased=False); return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

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
        B, D, H, W = x.shape
        residual = x
        x_norm = self.norm(x)
        x_permuted = x_norm.permute(0, 2, 3, 1)
        x_pixel_batches = x_permuted.reshape(B * H * W, D)
        x_grouped = x_pixel_batches.view(B * H * W, self.num_groups, self.group_len)
        x_mamba_input = x_grouped.transpose(1, 2).unsqueeze(-1)
        mamba_out_4d = self.mamba_scanner(x_mamba_input)
        mamba_out_squeezed = mamba_out_4d.squeeze(-1)
        mamba_out_ungrouped = mamba_out_squeezed.transpose(1, 2)
        x_un_reshaped = mamba_out_ungrouped.reshape(B * H * W, D)
        x_restored_permuted = x_un_reshaped.view(B, H, W, D)
        out = x_restored_permuted.permute(0, 3, 1, 2)
        return residual + self.drop_path(out)

# ==============================================================================
# Section 2: 主网络 InterleavedMambaMRF
# ==============================================================================

class InterleavedMambaMRF(nn.Module):
    def __init__(self,
                 in_chans=400,
                 num_out_chans=2,
                 embed_dim=180,
                 spatial_depth=12,
                 temporal_depth=12,
                 num_groups_temporal=6,
                 d_state=16,
                 drop_path_rate=0.1,
                 use_checkpoint=True,
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 统一计算DropPath rates
        total_module_calls = spatial_depth + temporal_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_module_calls)] if total_module_calls > 0 else []
        
        # 创建独立的模块列表
        self.spatial_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        
        # 按照最终的交错顺序来构建列表并分配dpr
        s_idx, t_idx, dpr_idx = 0, 0, 0
        while s_idx < spatial_depth or t_idx < temporal_depth:
            if s_idx < spatial_depth:
                self.spatial_blocks.append(SpatialMambaBlock(
                    dim=embed_dim, d_state=d_state, drop_path=dpr[dpr_idx] if dpr_idx < len(dpr) else 0.0
                ))
                s_idx += 1
                dpr_idx += 1
            if t_idx < temporal_depth:
                self.temporal_blocks.append(TemporalMambaBlock(
                    dim=embed_dim, num_groups=num_groups_temporal, d_state=d_state,
                    drop_path=dpr[dpr_idx] if dpr_idx < len(dpr) else 0.0
                ))
                t_idx += 1
                dpr_idx += 1
        
        self.conv_last = nn.Conv2d(embed_dim, num_out_chans, 3, 1, 1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x_first = self.conv_first(x)
        
        num_iterations = max(len(self.spatial_blocks), len(self.temporal_blocks))
        
        x = x_first
        for i in range(num_iterations):
            if i < len(self.spatial_blocks):
                blk = self.spatial_blocks[i]
                if self.use_checkpoint and self.training:
                    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)

            if i < len(self.temporal_blocks):
                blk = self.temporal_blocks[i]
                if self.use_checkpoint and self.training:
                    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
                
        output = self.conv_last(x + x_first)
        
        return output
