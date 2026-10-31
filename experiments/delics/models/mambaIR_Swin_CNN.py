import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable

# timm and mamba_ssm are required dependencies
# pip install timm mamba-ssm causal-conv1d
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

# =================================================================================
# Helper classes (ChannelAttention, CAB, SS2D, VSSBlock, etc.)
# These are unchanged.
# =================================================================================


# =================================================================================
# NEW: Guided Swin Upsampler Module
# =================================================================================
class GuidedSwinUpsampler(nn.Module):
    def __init__(self, embed_dim=60, swin_final_stage_channels=768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Define upsampling blocks for each stage
        # Stage 1: 8x8 -> 16x16
        self.up_block1 = self.create_up_block(swin_final_stage_channels, 512)
        # The input to the fusion conv will be upsampled_channels + guide_channels
        self.fusion1 = nn.Conv2d(512 + embed_dim, 512, 3, 1, 1)

        # Stage 2: 16x16 -> 32x32
        self.up_block2 = self.create_up_block(512, 256)
        self.fusion2 = nn.Conv2d(256 + embed_dim, 256, 3, 1, 1)

        # Stage 3: 32x32 -> 64x64
        self.up_block3 = self.create_up_block(256, 128)
        self.fusion3 = nn.Conv2d(128 + embed_dim, 128, 3, 1, 1)
        
        # Stage 4: 64x64 -> 128x128
        self.up_block4 = self.create_up_block(128, 64)
        self.fusion4 = nn.Conv2d(64 + embed_dim, 64, 3, 1, 1)
        
        # Stage 5: 128x128 -> 256x256
        self.up_block5 = self.create_up_block(64, embed_dim)
        self.fusion5 = nn.Conv2d(embed_dim + embed_dim, embed_dim, 3, 1, 1)

    def create_up_block(self, in_chans, out_chans):
        return nn.Sequential(
            nn.Conv2d(in_chans, out_chans * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_swin, cnn_guide_features):
        # x_swin: The raw 8x8 output from Swin backbone
        # cnn_guide_features: The full-resolution 256x256 features from CNN branch

        # Stage 1
        x = self.up_block1(x_swin)
        guide = F.interpolate(cnn_guide_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, guide], dim=1)
        x = self.fusion1(x)
        
        # Stage 2
        x = self.up_block2(x)
        guide = F.interpolate(cnn_guide_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, guide], dim=1)
        x = self.fusion2(x)

        # Stage 3
        x = self.up_block3(x)
        guide = F.interpolate(cnn_guide_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, guide], dim=1)
        x = self.fusion3(x)
        
        # Stage 4
        x = self.up_block4(x)
        guide = F.interpolate(cnn_guide_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, guide], dim=1)
        x = self.fusion4(x)
        
        # Stage 5
        x = self.up_block5(x)
        # The final guide is the full-res feature itself
        x = torch.cat([x, cnn_guide_features], dim=1)
        x = self.fusion5(x)
        
        return x
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN."""
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:
             compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )
    def forward(self, x):
        return self.cab(x)

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
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
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
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
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr=is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input_seq, x_size):
        B, L, C = input_seq.shape
        input_img_like = input_seq.view(B, *x_size, C).contiguous()
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

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 d_state: int = 16,
                 is_light_sr: bool = False,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
                d_state=d_state,
                is_light_sr=is_light_sr,
                input_resolution=input_resolution
            ))

    def forward(self, x_seq, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x_seq = checkpoint.checkpoint(blk, x_seq, x_size, use_reentrant=False)
            else:
                x_seq = blk(x_seq, x_size)
        return x_seq

class ResidualGroup(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 d_state: int = 16,
                 is_light_sr: bool = False,
                 ):
        super(ResidualGroup, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group_basic_layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            d_state=d_state,
            is_light_sr=is_light_sr,
        )
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.patch_embed_rg = PatchEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim)
        self.patch_unembed_rg = PatchUnEmbed(img_size=input_resolution, patch_size=1, embed_dim=dim)

    def forward(self, x_seq, x_size):
        processed_seq = self.residual_group_basic_layer(x_seq, x_size)
        processed_img = self.patch_unembed_rg(processed_seq, x_size)
        conv_out_img = self.conv(processed_img)
        conv_out_seq = self.patch_embed_rg(conv_out_img)
        return conv_out_seq + x_seq

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patches_resolution = [to_2tuple(img_size)[0] // to_2tuple(patch_size)[0], to_2tuple(img_size)[1] // to_2tuple(patch_size)[1]]
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96):
        super().__init__()
        self.patches_resolution = [to_2tuple(img_size)[0] // to_2tuple(patch_size)[0], to_2tuple(img_size)[1] // to_2tuple(patch_size)[1]]
        self.embed_dim = embed_dim

    def forward(self, x_seq, x_size):
        x = x_seq.transpose(1, 2).view(x_seq.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

# =================================================================================
# The new MambaSwinCNN Tri-Branch Model
# =================================================================================

class MambaSwinCNN(nn.Module):
    """
    MambaSwinCNN Tri-Branch Hybrid Model for MRF Reconstruction.

    This model features a three-branch parallel architecture:
    1.  Mamba Branch: A deep Mamba-based branch for long-range dependencies.
    2.  Swin Branch: A Swin Transformer based branch for multi-scale spatial features.
    3.  CNN Branch: A light-weight CNN branch for local texture and high-freq details.

    The outputs of the three branches are fused via element-wise addition.
    """
    def __init__(self,
                 img_size=256,
                 patch_size=1,
                 in_chans=400,
                 embed_dim=180,
                 mamba_group_depths=(6, 6, 6, 6),
                 cnn_depth=4,
                 mlp_ratio=2.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=True,
                 img_range=1.,
                 resi_connection='1conv',
                 d_state=16,
                 num_out_chans=2,
                 **kwargs):
        super(MambaSwinCNN, self).__init__()

        # --- Basic Model Parameters ---
        self.use_checkpoint = use_checkpoint
        self.img_range = img_range
        self.embed_dim = embed_dim
        
        # --- Shallow Feature Extraction ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # --- Main Patch Embed/UnEmbed (primarily for Mamba Branch) ---
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.body_patches_resolution = self.patch_embed.patches_resolution

        # --- Stochastic Depth (for Mamba Branch only) ---
        num_total_vss_blocks_in_mamba = sum(mamba_group_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_total_vss_blocks_in_mamba)]

        # --- Branch 1: Deep Mamba Branch ---
        self.mamba_layers = nn.ModuleList()
        dpr_idx_offset = 0
        for i_group in range(len(mamba_group_depths)):
            num_vss_blocks_in_this_group = mamba_group_depths[i_group]
            self.mamba_layers.append(ResidualGroup(
                dim=embed_dim,
                input_resolution=self.body_patches_resolution,
                depth=num_vss_blocks_in_this_group,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[dpr_idx_offset : dpr_idx_offset + num_vss_blocks_in_this_group],
                use_checkpoint=use_checkpoint,
                d_state=d_state,
                resi_connection=resi_connection
            ))
            dpr_idx_offset += num_vss_blocks_in_this_group
        
        self.norm_mamba_branch = norm_layer(embed_dim)

        # --- Branch 2: Swin Transformer Backbone ---
        self.swin_backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            in_chans=embed_dim,
            features_only=True,
            img_size=img_size,
            grad_checkpointing=use_checkpoint
        )
        
        # --- Branch 3: CNN Branch ---
        self.cnn_branch = nn.Sequential(
            *[CAB(num_feat=embed_dim) for _ in range(cnn_depth)]
        )

        # --- NEW: Guided Swin Upsampler ---
        # This module now connects the Swin and CNN branches.
        swin_final_stage_channels = self.swin_backbone.feature_info.channels()[-1]
        self.guided_swin_upsampler = GuidedSwinUpsampler(
            embed_dim=embed_dim,
            swin_final_stage_channels=swin_final_stage_channels
        )

        # --- Post-Fusion Modules ---
        if resi_connection == '1conv':
            # Note: The fusion now happens between Mamba and the enhanced Swin-CNN block.
            # So the input to this layer will be embed_dim * 2 if we concatenate,
            # or embed_dim if we add. Let's stick to addition for now.
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        
        self.conv_last = nn.Conv2d(embed_dim, num_out_chans, 3, 1, 1)

        # --- Mean Initialization ---
        self.mean = torch.zeros(1, in_chans, 1, 1)
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _forward_mamba_branch(self, x_seq, x_size):
        for layer_group in self.mamba_layers:
            if self.use_checkpoint:
                x_seq = checkpoint.checkpoint(layer_group, x_seq, x_size, use_reentrant=False)
            else:
                x_seq = layer_group(x_seq, x_size)
        return x_seq
        
    def _forward_swin_branch(self, x_img):
        swin_features = self.swin_backbone(x_img)
        x_swin_raw = swin_features[-1]
        if x_swin_raw.ndim == 4 and x_swin_raw.shape[1] != self.swin_backbone.feature_info.channels()[-1]:
             x_swin_raw = x_swin_raw.permute(0, 3, 1, 2).contiguous()
        x_swin_img = self.swin_upsampler(x_swin_raw)
        return x_swin_img
        
    def _forward_cnn_branch(self, x_img):
        # Can also add checkpointing here if the CNN branch becomes deep
        return self.cnn_branch(x_img)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x_in_range = (x - self.mean) * self.img_range

        # 1. Shared shallow feature extraction
        x_first = self.conv_first(x_in_range)
        current_x_size = (x_first.shape[2], x_first.shape[3])

        # 2. Parallel Branch Processing
        # --- Mamba Branch Path (Deep Features) ---
        x_seq = self.patch_embed(x_first)
        x_seq = self.pos_drop(x_seq)
        x_mamba_seq = self._forward_mamba_branch(x_seq, current_x_size)
        x_mamba_normed_seq = self.norm_mamba_branch(x_mamba_seq)
        x_mamba_img = self.patch_unembed(x_mamba_normed_seq, current_x_size)
        
        # --- Enhanced Shallow Module (Swin + CNN) Path ---
        # First, get the outputs from both Swin backbone and CNN branch
        swin_features = self.swin_backbone(x_first)
        x_swin_raw = swin_features[-1]
        if x_swin_raw.ndim == 4 and x_swin_raw.shape[1] != self.swin_backbone.feature_info.channels()[-1]:
             x_swin_raw = x_swin_raw.permute(0, 3, 1, 2).contiguous()
        
        x_cnn_img = self._forward_cnn_branch(x_first)
        
        # Now, use the CNN output to guide the Swin upsampling
        x_enhanced_shallow_img = self.guided_swin_upsampler(x_swin_raw, x_cnn_img)
        
        # 3. Final Fusion (Mamba Deep vs Enhanced Shallow)
        # We now fuse the Mamba branch with the output of the enhanced Swin-CNN block.
        x_merged_img = x_mamba_img + x_enhanced_shallow_img
        
        # 4. Post-fusion processing and final reconstruction
        res_features = self.conv_after_body(x_merged_img) + x_first
        output = self.conv_last(res_features)
        
        output = output / self.img_range
        
        return output

