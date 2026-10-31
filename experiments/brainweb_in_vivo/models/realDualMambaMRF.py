# ==============================================================================
# DualMambaMRF.py - 最终正确版
#
# 核心修正:
# 1. 恢复了 U-Net 代码中原始的 TemporalMambaBlock，它包含一个
#    关键的“维度戏法”(transpose + unsqueeze)，能让一个4D视觉Mamba
#    在功能上实现对3D时间序列的分组扫描。
# 2. 确保这个 TemporalMambaBlock 调用您提供的、为视觉优化的 mamba.py。
# 3. 移除了之前版本中所有错误的简化和假设。
#
# 前提:
# 将您提供的 mamba.py 文件与此 DualMambaMRF.py 文件放在同一个 models 目录下。
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from typing import Callable

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 核心依赖：从本地 mamba.py 导入所有需要的组件
from .mamba import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# ==============================================================================
# Section 1: 核心组件与辅助模块
# ==============================================================================

# --- 组件 1.1: 空间 Mamba 分支所需模块 (保持不变，确保其独立性) ---
class SS2D(nn.Module):
    # ... (此处粘贴完整的 SS2D 代码, 和之前一样) ...
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
    def forward(self, x: torch.Tensor, **kwargs):
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
    # ... (此处粘贴完整的 VSSBlock 代码, 和之前一样) ...
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, input_seq, x_size):
        B, L, C = input_seq.shape
        input_img_like = input_seq.view(B, *x_size, C).contiguous()
        x = self.ln_1(input_img_like)
        attended_x = self.self_attention(x)
        x = input_seq + self.drop_path(attended_x.view(B, L, C))
        return x

class PatchEmbed(nn.Module):
    # ... (此处粘贴完整的 PatchEmbed 代码, 和之前一样) ...
    def __init__(self, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None: x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    # ... (此处粘贴完整的 PatchUnEmbed 代码, 和之前一样) ...
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x_seq, x_size):
        B, L, C = x_seq.shape
        H, W = x_size
        return x_seq.transpose(1, 2).view(B, C, H, W)
    
class BasicLayer(nn.Module):
    # ... (此处粘贴完整的 BasicLayer 代码, 和之前一样) ...
    def __init__(self, dim, input_resolution, depth, drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, d_state: int = 16):
        super().__init__()
        self.dim, self.input_resolution, self.depth, self.use_checkpoint = dim, input_resolution, depth, use_checkpoint
        self.blocks = nn.ModuleList([VSSBlock(hidden_dim=dim, drop_path=dp, norm_layer=norm_layer, d_state=d_state, input_resolution=input_resolution) for dp in (drop_path if isinstance(drop_path, list) else [drop_path]*depth)])
    def forward(self, x_seq, x_size):
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x_seq = checkpoint.checkpoint(blk, x_seq, x_size, use_reentrant=False)
            else:
                x_seq = blk(x_seq, x_size)
        return x_seq

class ResidualGroup(nn.Module):
    # ... (此处粘贴完整的 ResidualGroup 代码, 和之前一样) ...
    def __init__(self, dim, input_resolution, depth, drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, resi_connection='1conv', d_state: int = 16):
        super(ResidualGroup, self).__init__()
        self.dim, self.input_resolution = dim, input_resolution
        self.residual_group_basic_layer = BasicLayer(dim=dim, input_resolution=input_resolution, depth=depth, drop_path=drop_path, norm_layer=norm_layer, use_checkpoint=use_checkpoint, d_state=d_state)
        if resi_connection == '1conv': self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv': self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(dim//4, dim//4, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(dim//4, dim, 3, 1, 1))
        self.patch_embed_rg = PatchEmbed(patch_size=1, in_chans=dim, embed_dim=dim)
        self.patch_unembed_rg = PatchUnEmbed(embed_dim=dim)
    def forward(self, x_seq, x_size):
        processed_seq = self.residual_group_basic_layer(x_seq, x_size)
        processed_img = self.patch_unembed_rg(processed_seq, x_size)
        conv_out_img = self.conv(processed_img)
        conv_out_seq = self.patch_embed_rg(conv_out_img)
        return conv_out_seq + x_seq

# --- 组件 1.2: 时间 Mamba 分支所需模块 (恢复U-Net中的正确版本) ---
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

# <<< 核心修正: 恢复U-Net中原始的、带有“维度戏法”的TemporalMambaBlock >>>
class TemporalMambaBlock(nn.Module):
    def __init__(self, dim, num_groups=6, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        if dim % num_groups != 0:
            raise ValueError(f"维度 ({dim}) 必须能被组数 ({num_groups}) 整除")
        self.group_len = dim // num_groups
        
        # 这个Mamba实例来自您提供的、能处理4D张量的mamba.py
        self.mamba_scanner = Mamba(
            d_model=self.group_len, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.norm = LayerNorm(dim, 'WithBias')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, D, H, W = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # 准备分组扫描
        x_permuted = x_norm.permute(0, 2, 3, 1)
        x_pixel_batches = x_permuted.reshape(B * H * W, D)
        x_grouped = x_pixel_batches.view(B * H * W, self.num_groups, self.group_len)
        
        # 【维度戏法】: 将3D序列伪装成4D图像 (B', C', H', W')
        # B' -> B*H*W, C' -> group_len, H' -> num_groups, W' -> 1
        x_mamba_input = x_grouped.transpose(1, 2).unsqueeze(-1)
        
        mamba_out_4d = self.mamba_scanner(x_mamba_input)
        
        # 恢复原始形状
        mamba_out_squeezed = mamba_out_4d.squeeze(-1)
        mamba_out_ungrouped = mamba_out_squeezed.transpose(1, 2)
        x_un_reshaped = mamba_out_ungrouped.reshape(B * H * W, D)
        x_restored_permuted = x_un_reshaped.view(B, H, W, D)
        out = x_restored_permuted.permute(0, 3, 1, 2)
        
        return residual + self.drop_path(out)

# ==============================================================================
# Section 2: 主网络 DualMambaMRF
# ==============================================================================

class DualMambaMRF(nn.Module):
    def __init__(self,
                 img_size=256,
                 patch_size=1,
                 in_chans=400,
                 num_out_chans=2,
                 embed_dim=180, 
                 mamba_group_depths=(6, 6, 6, 6),
                 temporal_depth=12,
                 num_groups_temporal=4, # <--- 现在这个参数是有意义的了
                 d_state=16,
                 drop_path_rate=0.1,
                 resi_connection='1conv',
                 use_checkpoint=True,
                 **kwargs):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embed_dim = embed_dim
        
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # --- 空间 Mamba 分支 (保持原样) ---
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        
        num_total_vss_blocks = sum(mamba_group_depths)
        spatial_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_total_vss_blocks)]
        self.spatial_branch = nn.ModuleList([
            ResidualGroup(
                dim=embed_dim, input_resolution=self.patches_resolution, depth=depth, 
                drop_path=spatial_dpr[sum(mamba_group_depths[:i]):sum(mamba_group_depths[:i+1])], 
                use_checkpoint=use_checkpoint, d_state=d_state, resi_connection=resi_connection
            ) for i, depth in enumerate(mamba_group_depths)
        ])
        self.norm_spatial = nn.LayerNorm(embed_dim)

        # --- 时间 Mamba 分支 (使用正确的 TemporalMambaBlock) ---
        temporal_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, temporal_depth)]
        self.temporal_branch = nn.ModuleList([
            TemporalMambaBlock(
                dim=embed_dim, num_groups=num_groups_temporal, d_state=d_state,
                drop_path=temporal_dpr[i]
            ) for i in range(temporal_depth)
        ])

        # --- 融合与重建模块 ---
        self.fusion_conv = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        self.conv_last = nn.Conv2d(embed_dim, num_out_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
            
    def _forward_spatial_branch(self, x, x_size):
        x_seq = self.patch_embed(x)
        for layer_group in self.spatial_branch:
            x_seq = layer_group(x_seq, x_size)
        x_norm_seq = self.norm_spatial(x_seq)
        return self.patch_unembed(x_norm_seq, x_size)
    
    def _forward_temporal_branch(self, x):
        for blk in self.temporal_branch:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        x_first = self.conv_first(x)
        
        x_spatial_img = self._forward_spatial_branch(x_first, x_size)
        x_temporal_img = self._forward_temporal_branch(x_first)
        
        x_fused_cat = torch.cat([x_spatial_img, x_temporal_img], dim=1)
        x_merged_img = self.fusion_conv(x_fused_cat)
        
        res_features = self.conv_after_body(x_merged_img) + x_first
        output = self.conv_last(res_features)
        
        return output
