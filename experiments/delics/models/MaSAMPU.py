import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

# 核心依赖：从本地 mamba.py 导入
try:
    from .mamba import Mamba
except ImportError:
    # 兼容调试模式，如果没有 mamba 环境可以跳过或报错
    pass

# ==============================================================================
# Section 1: 核心组件 (Swin Attention & Manhattan Attention)
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
    """ [分支 A] 标准 Swin Window Attention (负责局部精细纹理) """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        ).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecomposedMaSA(nn.Module):
    """
    [分支 B] 分解版曼哈顿自注意力 (Decomposed Manhattan Self-Attention)
    物理意义：基于曼哈顿距离的指数衰减，提供显式的全局空间先验 (Explicit Spatial Prior)。
    特点：将 2D 全局注意力分解为 H 和 W 两个方向，计算量由 O((HW)^2) 降为 O(H*W*(H+W))。
    """
    def __init__(self, dim, num_heads, resolution=(256, 256), decay_base=0.9):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = to_2tuple(resolution) # (H, W)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 可学习的衰减率 gamma (Initial decay rate)
        # 形状: [num_heads, 1, 1] 方便广播
        self.gamma_h = nn.Parameter(torch.ones(num_heads, 1, 1) * decay_base)
        self.gamma_w = nn.Parameter(torch.ones(num_heads, 1, 1) * decay_base)

        # QKV Projections
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # 预计算距离矩阵索引 (Distance Index) - 极大地加速 Forward
        h_idx = torch.arange(self.resolution[0])
        self.dist_h = torch.abs(h_idx.unsqueeze(0) - h_idx.unsqueeze(1)) # [H, H]
        
        w_idx = torch.arange(self.resolution[1])
        self.dist_w = torch.abs(w_idx.unsqueeze(0) - w_idx.unsqueeze(1)) # [W, W]
        
        # 注册为 buffer (随模型保存但不更新梯度)
        self.register_buffer('distance_h', self.dist_h)
        self.register_buffer('distance_w', self.dist_w)

    def get_decay_mask(self, dist, gamma):
        # Mask = gamma ^ distance
        # dist: [L, L], gamma: [heads, 1, 1] -> [1, heads, L, L]
        return torch.pow(gamma, dist).unsqueeze(0)

    def apply_1d_attention(self, x, gamma, dist_matrix):
        # 输入 x: [Batch*Other_Dim, Length, Channels]
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, nH, L, d]

        # Standard Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, nH, L, L]
        
        # --- 核心：注入曼哈顿先验 (Hadamard Product) ---
        # 这里的 mask 广播机制会自动匹配 Batch 维度
        decay_mask = self.get_decay_mask(dist_matrix, gamma) # [1, nH, L, L]
        attn = attn * decay_mask 
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x

    def forward(self, x):
        # x: [B, H*W, C]
        H, W = self.resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # 1. Horizontal Pass (W-axis)
        # 将 H 并入 Batch，对 W 做注意力
        x_w = x.view(B * H, W, C)
        x_w = self.apply_1d_attention(x_w, self.gamma_w, self.distance_w)
        x_w = x_w.view(B, H, W, C)

        # 2. Vertical Pass (H-axis)
        # 将 W 并入 Batch，对 H 做注意力
        x_h = x_w.permute(0, 2, 1, 3).reshape(B * W, H, C)
        x_h = self.apply_1d_attention(x_h, self.gamma_h, self.distance_h)
        x_h = x_h.view(B, W, H, C).permute(0, 2, 1, 3) # 变回 [B, H, W, C]
        
        return x_h.reshape(B, L, C)

# ==============================================================================
# Section 2: 双分支 Swin Block (Dual-Branch Block)
# ==============================================================================

class DualSwinBlock(nn.Module):
    """
    终极魔改 Block：Swin (Local) + MaSA (Global) 双流并行
    融合方式：Concat + Linear
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 input_resolution=(256, 256)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = to_2tuple(input_resolution)
        
        # --- 分支 A: Swin Window Attention (规则纹理) ---
        self.norm1_a = nn.LayerNorm(dim)
        self.attn_a = WindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # --- 分支 B: Manhattan Self-Attention (空间先验/全局结构) ---
        self.norm1_b = nn.LayerNorm(dim)
        self.attn_b = DecomposedMaSA(dim, num_heads, resolution=input_resolution)
        
        # --- Fusion Module (简单粗暴有效) ---
        self.fusion = nn.Linear(dim * 2, dim)

        # --- Shared MLP ---
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
        # 注意：这里我们假设 H, W 与 input_resolution 一致
        # 如果动态变化，MaSA 部分的 buffer 会报错，所以必须固定分辨率
        B, L, C = x.shape
        shortcut = x
        
        # -------------------------------------------------
        # Branch A: Swin Window Attention Logic
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Branch B: MaSA (Manhattan Global Prior)
        # -------------------------------------------------
        x_b = self.norm1_b(x)
        x_b_out = self.attn_b(x_b) # [B, L, C]
        
        # -------------------------------------------------
        # Feature Fusion
        # -------------------------------------------------
        x_cat = torch.cat([x_a_out, x_b_out], dim=-1) # [B, L, 2C]
        x_fused = self.fusion(x_cat) # [B, L, C]

        # -------------------------------------------------
        # Post-Processing
        # -------------------------------------------------
        x = shortcut + self.drop_path(x_fused)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SwinLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop_path=[0., 0.], img_size=(256, 256)):
        super().__init__()
        # 使用 DualSwinBlock，并传入 img_size
        self.block1 = DualSwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, 
                                    shift_size=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                    drop_path=drop_path[0], input_resolution=img_size)
        
        self.block2 = DualSwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, 
                                    shift_size=window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                    drop_path=drop_path[1], input_resolution=img_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, L, C]
        
        # 传递 H, W
        x_flat = self.block1(x_flat, H, W)
        x_flat = self.block2(x_flat, H, W)
        
        return x_flat.transpose(1, 2).view(B, C, H, W)

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
            nn.GroupNorm(1, dim_out), 
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x_mamba, x_swin, x_cnn):
        x_cat = torch.cat([x_mamba, x_swin, x_cnn], dim=1)
        return self.proj(x_cat)

class CascadeStage(nn.Module):
    def __init__(self, dim_in, dim_out, swin_heads=6, mamba_groups=6, window_size=8, d_state=16, drop_path=0., img_size=256):
        super().__init__()
        # 传递 img_size 到 SwinLayer
        self.branch_swin = SwinLayer(dim=dim_in, num_heads=swin_heads, window_size=window_size, drop_path=[drop_path, drop_path], img_size=img_size)
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
                 img_size=256 # 新增参数：输入图像尺寸 (假设正方形)
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv_first = nn.Conv2d(in_chans, stage_dims[0], 3, 1, 1)
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(stage_dims))]
        
        for i in range(len(stage_dims)):
            dim_in = stage_dims[i]
            dim_out = stage_dims[i+1] if i < len(stage_dims)-1 else stage_dims[-1]
            
            # 传递当前 stage 的分辨率给 MaSA
            # 假设只有第一层是 Conv3x3 (stride=1)，后续都是 Stage 内部处理降维?
            # 等等，原来的代码里 dim_in -> dim_out 是通过 MPU 做的降维
            # 那么 每个 CascadeStage 的输入分辨率其实是一样的？(因为没有 stride conv)
            # 如果是这样，那每一层都是 img_size
            # 如果 MPU 改变了 H,W，那这里需要调整。根据 ManifoldProjectionUnit 的定义，它是 Conv 1x1，不改变 H,W。
            # 所以，所有 Stage 的 H,W 都是 img_size。
            current_resolution = (img_size, img_size)
            
            self.stages.append(
                CascadeStage(
                    dim_in=dim_in, dim_out=dim_out,
                    swin_heads=stage_heads[i], mamba_groups=stage_groups[i],
                    window_size=window_size, d_state=d_state, drop_path=dpr[i],
                    img_size=current_resolution # 传入分辨率
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
