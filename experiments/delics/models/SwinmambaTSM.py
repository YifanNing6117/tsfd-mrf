import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_
import numbers

# 明确依赖同一目录下的 mamba.py
from .mamba import Mamba


##########################################################################
## Section 1: 辅助工具与核心组件 (均为Swinmamba模型所需)
##########################################################################
class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM).
    This module shifts a portion of channels along the channel dimension,
    which in our task corresponds to the pseudo-temporal axis.
    This implementation is a standard, robust, three-part shift.

    Args:
        n_div (int): The denominator for channel division. 1/n_div of channels
                     will be shifted forward, and 1/n_div will be shifted backward.
                     Defaults to 8.
    """
    def __init__(self, n_div=8):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                              C is the dimension to be shifted.
        """
        B, C, H, W = x.shape
        fold = C // self.fold_div

        # If fold is zero (e.g., C is too small), do nothing and return original tensor.
        # This adds robustness for very shallow layers or small dimensions.
        if fold == 0:
            return x

        # Create a new tensor to store the shifted result.
        # This is an out-of-place operation, which is generally more stable and
        # predictable than in-place operations, especially with checkpointing.
        out = torch.zeros_like(x)

        # Part 1: Shift [0, fold-1] channels backward
        # The first `fold` channels of `out` get data from the second `fold` channels of `x`.
        out[:, :fold, :, :] = x[:, fold:2*fold, :, :]

        # Part 2: Shift [fold, 2*fold-1] channels forward
        # The second `fold` channels of `out` get data from the first `fold` channels of `x`.
        out[:, fold:2*fold, :, :] = x[:, :fold, :, :]

        # Part 3: The remaining channels stay in place.
        out[:, 2*fold:, :, :] = x[:, 2*fold:, :, :]

        return out

def to_3d(x):
    """(B, C, H, W) -> (B, H*W, C)"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """(B, H*W, C) -> (B, C, H, W)"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将窗口合并回特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 图像高度
        W (int): 图像宽度
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """
    带相对位置偏置的窗口化多头自注意力。
    (这是被SwinTransformerBlock实际使用的版本)
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))

        # 获取相对位置索引
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
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
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

class SwinTransformerBlock(nn.Module):
    """Swin Transformer基础模块，支持常规窗口和移位窗口。"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias)
        
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.0)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口划分
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 计算注意力mask (仅在移位时需要)
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
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
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 逆移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SwinLayer(nn.Module):
    """一个Swin Layer，包含一个常规窗口和一个移位窗口的SwinTransformerBlock。"""
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.block1 = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
        self.block2 = SwinTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_3d = x.flatten(2).transpose(1, 2)
        
        x_3d = self.block1(x_3d, H, W)
        x_3d = self.block2(x_3d, H, W)

        x_3d = self.norm(x_3d)
        x_4d = x_3d.transpose(1, 2).view(B, C, H, W)
        return x_4d

class DRMamba(nn.Module):
    """带方向控制的Mamba封装，支持前向和反向扫描。"""
    def __init__(self, dim, reverse):
        super(DRMamba, self).__init__()
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.reverse = reverse

    def forward(self, x):
        # Mamba期望(B, L, C)格式，但这里传入的是(B, C, H, W)
        # 假设 .mamba.Mamba 内部处理了维度转换
        if self.reverse:
            x = x.flip(2) # 沿高度维度翻转以模拟反向扫描
        x = self.mamba(x)
        if self.reverse:
            x = x.flip(2)
        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """自定义的LayerNorm，支持2D图像张量输入。"""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class MambaBlock(nn.Module):
    """一个Mamba Block，包含前向和反向两次Mamba扫描。"""
    def __init__(self, dim, LayerNorm_type):
        super(MambaBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.mamba1 = DRMamba(dim, reverse=False)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mamba2 = DRMamba(dim, reverse=True)

    def forward(self, x):
        x = x + self.mamba1(self.norm1(x))
        x = x + self.mamba2(self.norm2(x))
        return x

class SpatialTransMambaBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.trans_block = SwinLayer(
            dim=dim, num_heads=num_heads, window_size=window_size,
            mlp_ratio=ffn_expansion_factor, qkv_bias=bias
        )
        self.mamba_block = MambaBlock(dim, LayerNorm_type)
        self.conv_fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # 【新增代码-第1行】: 实例化TSM模块。
        # 我们遵循奥卡姆剃刀原理，不添加任何其他组件。
        self.tsm = TemporalShift(n_div=8)

    def forward(self, x):
        # 1. Swin和Mamba分支，保持不变
        x1 = x + self.trans_block(x)
        x2 = self.mamba_block(x)
        
        # 2. 融合，保持不变
        out = torch.cat((x1, x2), dim=1)
        out = self.conv_fusion(out)
        
        # 【新增代码-第2行】: 在最后应用TSM。
        # 这是一个干净、独立的步骤，不与任何其他逻辑耦合。
        out_shifted = self.tsm(out)
        
        return out_shifted
##########################################################################
## Section 2: U-Net 基础结构模块
##########################################################################

class OverlapPatchEmbed(nn.Module):
    """使用重叠卷积进行Patch Embedding。"""
    def __init__(self, in_c, embed_dim, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    """下采样模块。"""
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    """上采样模块。"""
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
## Section 3: 主网络模型 - Swinmamba
##########################################################################

class Swinmamba(nn.Module):
    """
    一个基于U-Net的混合架构，其核心块由Swin Transformer和Mamba并行构成。
    此版本代码经过清理，移除了所有未使用的模块。
    """
    def __init__(self,
                 img_size=256,
                 in_chans=400,
                 num_out_chans=2,
                 embed_dim=180,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 window_size=8,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 use_checkpoint=True,
                 ):
        super(Swinmamba, self).__init__()

        self.use_checkpoint = use_checkpoint

        # 1. 浅层特征提取
        self.patch_embed = OverlapPatchEmbed(in_chans, embed_dim, bias=bias)

        # 2. 编码器
        self.encoder_level1 = nn.Sequential(*[
            SpatialTransMambaBlock(dim=embed_dim, num_heads=heads[0], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(embed_dim)
        
        self.encoder_level2 = nn.Sequential(*[
            SpatialTransMambaBlock(dim=int(embed_dim * 2 ** 1), num_heads=heads[1], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(embed_dim * 2 ** 1))
        
        self.encoder_level3 = nn.Sequential(*[
            SpatialTransMambaBlock(dim=int(embed_dim * 2 ** 2), num_heads=heads[2], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(int(embed_dim * 2 ** 2))
        
        self.latent = nn.Sequential(*[
            SpatialTransMambaBlock(dim=int(embed_dim * 2 ** 3), num_heads=heads[3], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        # 3. 解码器
        self.up4_3 = Upsample(int(embed_dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(embed_dim * 2 ** 3), int(embed_dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            SpatialTransMambaBlock(dim=int(embed_dim * 2 ** 2), num_heads=heads[2], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(embed_dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(embed_dim * 2 ** 2), int(embed_dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            SpatialTransMambaBlock(dim=int(embed_dim * 2 ** 1), num_heads=heads[1], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(embed_dim * 2 ** 1))
        self.reduce_chan_level1 = nn.Conv2d(int(embed_dim * 2 ** 1), embed_dim, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            SpatialTransMambaBlock(dim=embed_dim, num_heads=heads[0], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        # 4. 精炼模块
        self.refinement = nn.Sequential(*[
            SpatialTransMambaBlock(dim=embed_dim, num_heads=heads[0], window_size=window_size, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        # 5. 输出层
        self.output = nn.Conv2d(embed_dim, num_out_chans, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # 正向传播逻辑保持不变
        inp_enc_level1 = self.patch_embed(inp_img)

        if self.use_checkpoint:
            out_enc_level1 = checkpoint.checkpoint(self.encoder_level1, inp_enc_level1, use_reentrant=False)
            inp_enc_level2 = self.down1_2(out_enc_level1)
            out_enc_level2 = checkpoint.checkpoint(self.encoder_level2, inp_enc_level2, use_reentrant=False)
            inp_enc_level3 = self.down2_3(out_enc_level2)
            out_enc_level3 = checkpoint.checkpoint(self.encoder_level3, inp_enc_level3, use_reentrant=False)
            inp_enc_level4 = self.down3_4(out_enc_level3)
            latent = checkpoint.checkpoint(self.latent, inp_enc_level4, use_reentrant=False)
            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = checkpoint.checkpoint(self.decoder_level3, inp_dec_level3, use_reentrant=False)
            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = checkpoint.checkpoint(self.decoder_level2, inp_dec_level2, use_reentrant=False)
            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
            out_dec_level1 = checkpoint.checkpoint(self.decoder_level1, inp_dec_level1, use_reentrant=False)
            refined_features = checkpoint.checkpoint(self.refinement, out_dec_level1, use_reentrant=False)
        else:
            out_enc_level1 = self.encoder_level1(inp_enc_level1)
            inp_enc_level2 = self.down1_2(out_enc_level1)
            out_enc_level2 = self.encoder_level2(inp_enc_level2)
            inp_enc_level3 = self.down2_3(out_enc_level2)
            out_enc_level3 = self.encoder_level3(inp_enc_level3)
            inp_enc_level4 = self.down3_4(out_enc_level3)
            latent = self.latent(inp_enc_level4)
            inp_dec_level3 = self.up4_3(latent)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            out_dec_level3 = self.decoder_level3(inp_dec_level3)
            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2)
            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
            out_dec_level1 = self.decoder_level1(inp_dec_level1)
            refined_features = self.refinement(out_dec_level1)

        output = self.output(refined_features)
        return output
