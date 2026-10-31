import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from typing import Optional, Callable, List

# 核心依赖：mamba_ssm
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("Warning: mamba_ssm not found. SpatialMambaBlock will fail.")
    selective_scan_fn = None

# ==============================================================================
# Section 1: MambaIR 核心组件
# ==============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        mid_feat = max(1, num_feat // squeeze_factor)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, mid_feat, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_feat, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        mid_feat = max(1, num_feat // compress_ratio)
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, mid_feat, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))
    def forward(self, x):
        return self.cab(x)

# 1. 基础组件: SimpleMamba1D (手搓版，用于时序扫描)
# --------------------------------------------------
class SimpleMamba1D(nn.Module):
    """
    自包含的 1D Mamba，基于 selective_scan_fn。
    专门用于处理 [Batch, Length, Dim] 的序列数据。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = self.act(x)
        x = x.transpose(1, 2)

        x_dbl = self.x_proj(x) 
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt)
        
        x = x.transpose(1, 2).contiguous()
        dt = dt.transpose(1, 2).contiguous()
        B_ssm = B_ssm.transpose(1, 2).contiguous()
        C_ssm = C_ssm.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())
        
        y = selective_scan_fn(
            x, dt, A, B_ssm, C_ssm, self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        
        y = y.transpose(1, 2)
        y = y * self.act(z)
        out = self.out_proj(y)
        
        return out

# 2. 封装组件: BiTemporalMambaLayer (双向扫描)
# --------------------------------------------------
class BiTemporalMambaLayer(nn.Module):
    # ... (init 部分保持不变) ...
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.d_inner = 4 # 恢复到 16，因为 Chunking 已经解决了显存问题
        self.chunk_size = 4096
        self.in_proj = nn.Linear(1, self.d_inner)
        self.mamba = SimpleMamba1D(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(self.d_inner * 2, 1)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # [CRITICAL FIX] 不再做内部残差
        
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(-1, C, 1)
        total_pixels = x_seq.shape[0]
        x_embed = self.in_proj(x_seq)
        
        output_list = []
        for i in range(0, total_pixels, self.chunk_size):
            chunk = x_embed[i:i + self.chunk_size]
            x_fwd, x_bwd = chunk, torch.flip(chunk, [1])
            x_in_chunk = torch.cat([x_fwd, x_bwd], dim=0)
            x_out_chunk = self.mamba(x_in_chunk)
            out_fwd, out_bwd = x_out_chunk.chunk(2, dim=0)
            out_bwd = torch.flip(out_bwd, [1])
            out_merged = torch.cat([out_fwd, out_bwd], dim=-1)
            output_list.append(out_merged)
            
        full_out = torch.cat(output_list, dim=0)
        res = self.out_proj(full_out)
        res = res.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # [CRITICAL FIX] 归一化后直接返回处理结果
        res = self.norm(res.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return self.drop_path(res)

# [我写的] 频域分支 (一字不改)

# 1. 删掉 DynamicSpectralFilter，替换为 GlobalFilterLayer
# --------------------------------------------------
class GlobalFilterLayer(nn.Module):
    """
    [Minimal & Effective] - 极简全局滤波器 (GFNet-Lite)
    逻辑: FFT -> 乘以一个可学习的全局权重 -> IFFT
    """
    def __init__(self, dim, h=128, w=128):
        super().__init__()
        # 权重形状: [C, H, W//2+1, 2] (对应复数)
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_fft_filtered = x_fft * weight
        x_out = torch.fft.irfft2(x_fft_filtered, s=(H, W), norm='ortho')
        return x_out

class SpatialMambaBlock(nn.Module):
    """
    [消融实验 3: w/o GFNet]
    彻底去除了底部的 GlobalFilterLayer 频域分支和 alpha 融合机制。
    此时，该模块完美退化为纯粹的 Mamba 时序处理块。
    """
    def __init__(self, dim, depth=1, d_state=16, d_conv=3, expand=2, drop_path=0., img_size=256):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # [消融]: 删除 alpha 参数
        # self.alphas = nn.ParameterList()
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        # h = w = img_size (不再需要传给 frequency 层了)
        
        for i in range(depth):
            self.layers.append(
                nn.ModuleDict({
                    # 仅保留纯粹的 Bi-Temporal Mamba
                    'temporal': BiTemporalMambaLayer(
                        dim=dim, d_state=d_state, d_conv=d_conv, 
                        expand=expand, drop_path=dp_rates[i]
                    ),
                    # [消融]: 彻底删除 frequency 分支
                    # 'frequency': GlobalFilterLayer(dim=dim, h=h, w=w),
                })
            )
            # [消融]: 不再初始化 alpha
            # self.alphas.append(nn.Parameter(torch.tensor(0.5)))
            
        self.norm_temporal = nn.LayerNorm(dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = x
            
            # --- 仅保留上路: 时序 Mamba ---
            x_norm_for_temporal = self.norm_temporal(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if self.training:
                x_temporal_res = checkpoint.checkpoint(layer['temporal'], x_norm_for_temporal, use_reentrant=False)
            else:
                x_temporal_res = layer['temporal'](x_norm_for_temporal)
            
            # ========================================================
            # 🛡️ 核心消融：彻底跳过频域处理和融合计算！
            # ========================================================
            # x_freq_res = layer['frequency'](x)
            # alpha = torch.sigmoid(self.alphas[i])
            # x_fused = alpha * x_temporal_res + (1 - alpha) * x_freq_res
            
            # 直接使用纯 Mamba 特征
            x_fused = x_temporal_res
            
            # --- 残差相加 ---
            x = residual + x_fused
            
        return x
# 3. 基础组件: SS2D (原有的空间扫描)
# --------------------------------------------------
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
        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
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
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, K, -1, L)
        
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, 1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None: out = self.dropout(out)
        return out

# 4. 封装组件: VSSBlock (空间 Mamba Block)
# --------------------------------------------------
class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), attn_drop_rate: float = 0, d_state: int = 16, mlp_ratio: float = 2., is_light_sr: bool = False, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr=is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, input_seq, x_size):
        B, L, C = input_seq.shape
        H, W = x_size
        input_img_like = input_seq.view(B, H, W, C).contiguous()
        
        x = self.ln_1(input_img_like)
        attended_x = self.self_attention(x)
        x = input_img_like * self.skip_scale + self.drop_path(attended_x)
        
        x_for_cab_ln = self.ln_2(x)
        x_for_cab_permuted = x_for_cab_ln.permute(0, 3, 1, 2).contiguous()
        cab_out_permuted = self.conv_blk(x_for_cab_permuted)
        cab_out = cab_out_permuted.permute(0, 2, 3, 1).contiguous()
        
        x = x * self.skip_scale2 + cab_out
        x = x.view(B, -1, C).contiguous()
        return x

#=============================
# Section 2: Swin & Wavelet 注意力组件 (Dual-Branch)
# ==============================================================================
# ==============================================================================
# 新增: DWT 和 IDWT 基础算子 (Haar Wavelet)
# ==============================================================================
class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义 Haar 小波核 (不可学习，保持物理意义)
        # LL: 低频 (平均)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        # LH: 垂直边缘 (Vertical)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        # HL: 水平边缘 (Horizontal)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        # HH: 对角边缘 (Diagonal)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        # 堆叠核: (4, 1, 2, 2)
        filts = torch.stack([ll, lh, hl, hh]).unsqueeze(1)

        # 注册为 buffer，不是 Parameter (不参与梯度更新，只做变换)
        self.register_buffer('filters', filts)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 将 C 个通道的 filters 重复，适配 input channels
        # groups=C 保证每个通道独立进行 DWT，不混合通道
        filters = self.filters.repeat(C, 1, 1, 1)

        # stride=2 实现下采样
        out = F.conv2d(x, filters, stride=2, groups=C)
        # Output shape: (B, 4*C, H/2, W/2)
        # 通道排列顺序: C个LL, C个LH, C个HL, C个HH (取决于 PyTorch group conv 的实现)
        # 实际上 PyTorch group conv 输出是: [LL_c1, LH_c1, HL_c1, HH_c1, LL_c2, ...]
        return out


class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        # 逆变换核 (对应上面的正变换，Haar是正交的，转置即可)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        filts = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer('filters', filts)

    def forward(self, x):
        # x: (B, 4*C, H/2, W/2)
        B, TotalC, H_half, W_half = x.shape
        C = TotalC // 4

        filters = self.filters.repeat(C, 1, 1, 1)

        # 逆变换使用转置卷积 (Transposed Conv)
        # groups=C 保证通道独立
        out = F.conv_transpose2d(x, filters, stride=2, groups=C)
        return out



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
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
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


class TrueWaveletBlock(nn.Module):
    def __init__(self, dim, num_heads=None):
        # num_heads 参数保留是为了兼容原来的接口调用，这里实际上不用
        super().__init__()

        self.dwt = DWT()
        self.idwt = IDWT()

        # 高频处理分支: 处理 LH, HL, HH 三个分量
        # 输入通道是 3 * dim，因为有三个高频子带
        self.high_freq_processor = nn.Sequential(
            # 1. Depthwise Conv: 扩大感受野，提取高频特征
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3),
            nn.GELU(),
            # 2. Pointwise Conv (1x1): 通道融合，学习纹理相关性
            nn.Conv2d(dim * 3, dim * 3, kernel_size=1),
            # 3. Channel Attention: 你的代码里有这个类，加上它强化“关键高频筛选”
            ChannelAttention(dim * 3)
        )

        # 低频处理分支: LL 分量
        # 既然并行的 Swin 分支已经在处理全局信息，这里我们对 LL 做轻量处理或不做处理
        # 这里的 1x1 卷积是为了让网络有机会调整低频的幅度，以便和处理后的高频完美重组
        self.low_freq_processor = nn.Conv2d(dim, dim, 1)

    def forward(self, x, H, W):
        # x: (B, L, C) -> 来自 Swin 的输入通常是 Flatten 的
        B, L, C = x.shape

        # 1. 恢复图像维度
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 2. DWT 分解 -> (B, 4*C, H/2, W/2)
        # 此时通道排列是混合的: [LL1, LH1, HL1, HH1, LL2, LH2, ...]
        x_dwt = self.dwt(x_img)

        # 3. 重排通道以分离频率子带
        # View 成 (B, C, 4, ...) 方便切片
        x_reshaped = x_dwt.view(B, C, 4, H // 2, W // 2)

        # 提取子带
        x_ll = x_reshaped[:, :, 0, :, :]  # (B, C, H/2, W/2)
        x_high = x_reshaped[:, :, 1:, :, :]  # (B, C, 3, H/2, W/2)

        # 合并高频通道以便卷积: (B, 3*C, H/2, W/2)
        x_high = x_high.reshape(B, C * 3, H // 2, W // 2)

        # 4. 独立处理
        x_ll_out = self.low_freq_processor(x_ll)
        x_high_out = self.high_freq_processor(x_high)

        # 5. 准备重组 (Reconstruct)
        # 先把处理后的高频 reshape 回去: (B, C, 3, H/2, W/2)
        x_high_out = x_high_out.view(B, C, 3, H // 2, W // 2)

        # 拼接回 (B, C, 4, ...): 顺序必须是 LL, LH, HL, HH
        x_combined = torch.cat([
            x_ll_out.unsqueeze(2),
            x_high_out
        ], dim=2)

        # Flatten 回 (B, 4*C, ...) 供 IDWT 使用
        x_combined_flat = x_combined.view(B, C * 4, H // 2, W // 2)

        # 6. IDWT 逆变换
        x_recon = self.idwt(x_combined_flat)  # (B, C, H, W)

        # 7. 变回 Transformer 格式 (B, L, C)
        x_out = x_recon.permute(0, 2, 3, 1).reshape(B, L, C)

        return x_out
class DualSwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 input_resolution=(256, 256)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = to_2tuple(input_resolution)
        
        self.norm1_a = nn.LayerNorm(dim)
        self.attn_a = WindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm1_b = nn.LayerNorm(dim)
        # 替换为新的 TrueWaveletBlock
        self.attn_b = TrueWaveletBlock(dim, num_heads) 
        
        self.fusion = nn.Linear(dim * 2, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        
        x_a = self.norm1_a(x).view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_a = F.pad(x_a, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x_a.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x_a, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device); cnt = 0
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            for h in h_slices:
                for w in w_slices: 
                    img_mask[:, h, w, :] = cnt; cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x_a
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn_a(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        if self.shift_size > 0:
            x_a_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_a_out = shifted_x
        if pad_r > 0 or pad_b > 0:
            x_a_out = x_a_out[:, :H, :W, :].contiguous()
        x_a_out = x_a_out.view(B, L, C)

        x_b = self.norm1_b(x)
        x_b_out = self.attn_b(x_b, H, W) 
        
        x_cat = torch.cat([x_a_out, x_b_out], dim=-1)
        x_fused = self.fusion(x_cat)

        x = shortcut + self.drop_path(x_fused)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SwinLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop_path=[0., 0.], img_size=256):
        super().__init__()
        self.block1 = DualSwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, 
                                    shift_size=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                    drop_path=drop_path[0], input_resolution=(img_size, img_size))
        
        self.block2 = DualSwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, 
                                    shift_size=window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                    drop_path=drop_path[1], input_resolution=(img_size, img_size))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.block1(x_flat, H, W)
        x_flat = self.block2(x_flat, H, W)
        return x_flat.transpose(1, 2).view(B, C, H, W)

# ==============================================================================
# Section 3: 其他组件 (ResBlock, MPU, CSBNet)
# ==============================================================================

# ==============================================================================
# [FIXED] Section 3: CNN Branch (HAPM - 3D Topology with GroupNorm)
# 修正说明: 将所有的 BatchNorm3d 替换为 GroupNorm(1, C)，解决 BatchSize=1 时的训练崩溃问题。
# ==============================================================================

# ==============================================================================
# [FINAL 2D VERSION] Section 3: CNN Branch (HAPM - Pure 2D)
# 逻辑：将 Channel 维度视为“像素向量”，利用 2D 卷积的高效性进行处理。
# 显存占用极低，不再会有 OOM 问题。
# ==============================================================================

class LinearAttention(nn.Module):
    """ O(N) 线性注意力 """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.elu = nn.ELU()
        self.eps = eps

    def forward(self, queries, keys, values):
        # phi(x) = ELU(x) + 1
        Q = self.elu(queries) + 1.0
        K = self.elu(keys) + 1.0

        # KV: [B, H, D, D]
        KV = torch.einsum("bhnd,bhne->bhde", K, values)
        
        # Norm Z
        K_sum = K.sum(dim=2, keepdim=True)
        Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", Q, K_sum.squeeze(2)) + self.eps)
        
        # Aggregation
        V_agg = torch.einsum("bhnd,bhde->bhne", Q, KV)
        return V_agg * Z.unsqueeze(-1)

# -------------------------------------------------------------------------
# MPU V3: 方案 C (Tri-Perspective Retrieval + Concat Fusion)
# -------------------------------------------------------------------------
class ManifoldProjectionUnit(nn.Module):
    """
    [MPU V8] 双流共享注入总线
    逻辑: 
    1. 交互得到 Attn_M, Attn_S
    2. 拼接压缩得到 Shared_Feature
    3. M = M + Shared, S = S + Shared (残差注入)
    4. 各自卷积降维，各奔东西
    """
    def __init__(self, dim_in, dim_out, num_heads=4):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        
        # 1. 投影层
        self.to_q_m = nn.Conv2d(dim_in, dim_in, 1)
        self.to_kv_m = nn.Conv2d(dim_in, dim_in * 2, 1)
        self.to_q_s = nn.Conv2d(dim_in, dim_in, 1)
        self.to_kv_s = nn.Conv2d(dim_in, dim_in * 2, 1)
        
        self.attn = LinearAttention()
        
        # 2. 共享特征生成层 (Shared Feature Generator)
        # Input: [Attn_M, Attn_S] -> 2*dim_in
        # Output: dim_in (为了和原始输入做残差)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(dim_in * 2, dim_in, kernel_size=1, bias=False),
            nn.GroupNorm(1, dim_in),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. 独立降维层 (Separate Projection)
        # 负责将 dim_in 降维到 dim_out，传给下一个 Stage
        self.proj_m = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.proj_s = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

    def _run_attn(self, q, k, v, B, H, W):
        N = H * W
        q = q.flatten(2).transpose(1, 2).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.flatten(2).transpose(1, 2).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.flatten(2).transpose(1, 2).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        out = self.attn(q, k, v)
        out = out.permute(0, 2, 1, 3).flatten(2).transpose(1, 2).reshape(B, -1, H, W)
        return out

    def forward(self, x_mamba, x_swin):
        """
        Input: [B, dim_in, H, W] x 2
        Output: [B, dim_out, H', W'] x 2 (返回两个独立分支)
        """
        B, C, H, W = x_mamba.shape
        
        # --- Step 1: 准备 Q, K, V ---
        q_m = self.to_q_m(x_mamba)
        k_m, v_m = self.to_kv_m(x_mamba).chunk(2, dim=1)
        
        q_s = self.to_q_s(x_swin)
        k_s, v_s = self.to_kv_s(x_swin).chunk(2, dim=1)
        
        # --- Step 2: 交叉检索 ---
        info_m_from_s = self._run_attn(q_m, k_s, v_s, B, H, W)
        info_s_from_m = self._run_attn(q_s, k_m, v_m, B, H, W)
        
        # --- Step 3: 生成共享特征 (Shared Feature) ---
        # "把交换来的信息拼在一起，提炼出共识"
        concat_feat = torch.cat([info_m_from_s, info_s_from_m], dim=1)
        shared_feat = self.shared_conv(concat_feat) # -> [B, dim_in, H, W]
        
        # --- Step 4: 残差注入 ---
        # "把共识加回到各自的原始特征上"
        x_m_refined = x_mamba + shared_feat
        x_s_refined = x_swin + shared_feat
        
        # --- Step 5: 独立降维/输出 ---
        # "各自收拾行李，去往下一站"
        out_m = self.proj_m(x_m_refined)
        out_s = self.proj_s(x_s_refined)
        
        return out_m, out_s
class CascadeStage(nn.Module):
    def __init__(self, dim_in, dim_out, swin_heads=6, mamba_groups=6, window_size=8, d_state=16, drop_path=0., img_size=256):
        super().__init__()
        
        # Mamba 分支
        self.branch_mamba = SpatialMambaBlock(
            dim=dim_in, depth=mamba_groups, d_state=d_state, 
            drop_path=drop_path, img_size=img_size
        )
        
        # Swin 分支
        self.branch_swin = SwinLayer(
            dim=dim_in, num_heads=swin_heads, window_size=window_size, 
            drop_path=[drop_path, drop_path], img_size=img_size
        )
        
        # MPU (返回双流)
        self.mpu = ManifoldProjectionUnit(dim_in, dim_out, num_heads=swin_heads)

    def forward(self, x):
        # x 现在是一个元组 (x_mamba, x_swin)
        # 如果是第一层，可能只传来一个 x，需要做一下兼容
        if isinstance(x, tuple):
            x_m_in, x_s_in = x
        else:
            x_m_in = x_s_in = x

        # 1. 各自跑特征提取
        f_mamba = self.branch_mamba(x_m_in)
        f_swin = self.branch_swin(x_s_in)
        
        # 2. 进入 MPU 交互并降维
        # MPU 返回 (out_m, out_s)
        return self.mpu(f_mamba, f_swin)


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
                 img_size=256
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # 初始卷积
        self.conv_first = nn.Conv2d(400, stage_dims[0], 3, 1, 1)
        
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(stage_dims))]
        
        for i in range(len(stage_dims)):
            dim_in = stage_dims[i]
            # 最后一个 Stage 的输出维度不需要降太多，或者根据你的需求定
            # 这里逻辑保持不变，由 MPU 负责降维
            dim_out = stage_dims[i+1] if i < len(stage_dims)-1 else stage_dims[-1]
            
            self.stages.append(
                CascadeStage(
                    dim_in=dim_in, dim_out=dim_out,
                    swin_heads=stage_heads[i], mamba_groups=stage_groups[i],
                    window_size=window_size, d_state=d_state, drop_path=dpr[i],
                    img_size=img_size
                )
            )
            
        # 最终输出头
        # 此时我们有两个分支的输出，需要把它们合起来
        self.final_fusion = nn.Conv2d(stage_dims[-1] * 2, stage_dims[-1], 1)
        
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
        # 1. 初始特征提取
        x = self.conv_first(x)
        
        # 2. 准备双流输入
        # 刚开始，两个分支的输入是相同的
        inputs = (x, x) 
        
        # 3. 级联 Stage (双流传递)
        for stage in self.stages:
            if self.use_checkpoint and self.training:
                # Checkpoint 需要稍微特殊处理 tuple 输入，但通常只要 inputs 是 Tensor 就行
                # 这里为了简单，我们假设 stage 内部处理了
                # 注意：torch checkpoint 对 tuple 输入支持可能有限，
                # 如果报错，需要把 stage 改写成接受两个参数 forward(x_m, x_s)
                # 这里我们先直接调用
                inputs = stage(inputs)
            else:
                inputs = stage(inputs)
        
        # 4. 最终汇聚
        x_m_final, x_s_final = inputs
        x_cat = torch.cat([x_m_final, x_s_final], dim=1)
        x_fused = self.final_fusion(x_cat)
        
        return self.head(x_fused)
