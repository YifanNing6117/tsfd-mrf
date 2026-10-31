import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
# from basicsr.utils.registry import ARCH_REGISTRY # Assuming this might be used elsewhere
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
# from torch.nn import TransformerEncoderLayer # Not used in MambaIR

NEG_INF = -1000000


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

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
    # compress_ratio=3 for classic SR, and 6 for light-SR
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30): # Default compress_ratio to 3
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
             compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module): # Kept for completeness, though not directly used by MambaIR variants here
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
            **kwargs, # Accept and ignore extra_kwargs like input_resolution
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

    def forward(self, x: torch.Tensor, **kwargs): # kwargs to catch H, W if passed by VSSBlock
        B, H, W, C = x.shape # VSSBlock passes x as (B, H, W, C)
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
            attn_drop_rate: float = 0, # Renamed from expand to mlp_ratio in original MambaIR's VSSBlock call
            d_state: int = 16,
            mlp_ratio: float = 2., # This is 'expand' for SS2D
            is_light_sr: bool = False, # Added for CAB
            **kwargs, # Catches input_resolution from BasicLayer
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        # Pass mlp_ratio as expand to SS2D, and remove attn_drop_rate if not used by SS2D
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr=is_light_sr) # Pass is_light_sr
        self.ln_2 = nn.LayerNorm(hidden_dim) # Original MambaIR used nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input_seq, x_size): # input_seq is (B, L, C)
        B, L, C = input_seq.shape
        # Reshape to (B, H, W, C) for SS2D and CAB
        input_img_like = input_seq.view(B, *x_size, C).contiguous()

        # First part: SS2D
        x = self.ln_1(input_img_like)
        # SS2D expects (B, H, W, C)
        attended_x = self.self_attention(x) # Returns (B, H, W, C)
        x = input_img_like * self.skip_scale + self.drop_path(attended_x)

        # Second part: CAB
        # CAB expects (B, C, H, W)
        # ln_2 expects (B, H, W, C) from previous stage
        x_for_cab_ln = self.ln_2(x)
        # Permute for CAB
        x_for_cab_permuted = x_for_cab_ln.permute(0, 3, 1, 2).contiguous()
        cab_out_permuted = self.conv_blk(x_for_cab_permuted)
        # Permute back
        cab_out = cab_out_permuted.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)

        x = x * self.skip_scale2 + cab_out
        x = x.view(B, -1, C).contiguous() # Reshape back to (B, L, C)
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution, # Tuple (H, W)
                 depth,
                 mlp_ratio=2., # For VSSBlock/SS2D
                 drop_path=0.,
                 norm_layer=nn.LayerNorm, # For VSSBlock
                 # downsample=None, # Not used in MambaIR structure
                 use_checkpoint=False,
                 d_state: int = 16, # For VSSBlock/SS2D
                 is_light_sr: bool = False, # For VSSBlock/CAB
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
                mlp_ratio=mlp_ratio, # Passed to VSSBlock, then to SS2D as 'expand'
                d_state=d_state,     # Passed to VSSBlock, then to SS2D
                is_light_sr=is_light_sr, # Passed to VSSBlock, then to CAB
                input_resolution=input_resolution # Pass to VSSBlock for SS2D if it needs it (though SS2D gets H,W from input)
            ))
        # self.downsample = None # Explicitly set as MambaIR doesn't use it here

    def forward(self, x_seq, x_size): # x_seq is (B, L, C), x_size is (H,W)
        for blk in self.blocks:
            if self.use_checkpoint:
                x_seq = checkpoint.checkpoint(blk, x_seq, x_size)
            else:
                x_seq = blk(x_seq, x_size)
        # if self.downsample is not None: # This block is not hit
        #     x = self.downsample(x)
        return x_seq

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    # def flops(self): # Add if needed
    #     flops = 0
    #     for blk in self.blocks:
    #         flops += blk.flops() # VSSBlock would need a flops method
    #     return flops


class ResidualGroup(nn.Module):
    """Simplified ResidualGroup based on MambaIR's structure where it's a list of VSSBlocks.
       The original MambaIR used ResidualGroup to wrap BasicLayer and add a conv + skip.
       This version aligns with that.
    """
    def __init__(self,
                 dim,
                 input_resolution, # Tuple (H,W) of the feature map *before* patch_embed inside forward_features
                 depth, # Number of VSSBlocks inside the BasicLayer of this group
                 mlp_ratio=2.,
                 drop_path=0., # This will be a list for VSSBlocks
                 norm_layer=nn.LayerNorm,
                 # downsample=None, # Not used by MambaIR's ResidualGroup
                 use_checkpoint=False,
                 img_size=None, # Passed to PatchEmbed/UnEmbed within this group
                 patch_size=None, # Passed to PatchEmbed/UnEmbed within this group
                 resi_connection='1conv',
                 d_state: int = 16, # For BasicLayer/VSSBlock
                 is_light_sr: bool = False, # For BasicLayer/VSSBlock/CAB
                 ):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # Spatial res of the *input* to this group's BasicLayer

        self.residual_group_basic_layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution, # This is the H, W for VSSBlocks
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path, # List of drop_path rates for VSSBlocks
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
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        # These are for the residual connection around the BasicLayer + conv
        # They operate on data that is already in (B,L,C) from patch_embed before this group
        # or (B,C,H,W) after patch_unembed within this group.
        # img_size and patch_size here should correspond to the feature map dimensions *within* the body
        self.patch_embed_rg = PatchEmbed( # Renamed to avoid conflict
            img_size=input_resolution, # The H,W of the feature map
            patch_size=1,       # patch_size=1 means no spatial reduction, just reshape
            in_chans=dim,       # These are dummy if patch_size=1
            embed_dim=dim,
            norm_layer=None)    # No norm for these utility embeds/unembeds

        self.patch_unembed_rg = PatchUnEmbed( # Renamed
            img_size=input_resolution,
            patch_size=1,
            in_chans=dim,       # Dummy
            embed_dim=dim)

    def forward(self, x_seq, x_size): # x_seq: (B, L, C), x_size: (H, W)
        # BasicLayer operates on (B, L, C)
        processed_seq = self.residual_group_basic_layer(x_seq, x_size)

        # Convert sequence to image for conv
        # x_size here is the H,W of the feature map corresponding to L
        processed_img = self.patch_unembed_rg(processed_seq, x_size)
        conv_out_img = self.conv(processed_img)

        # Convert image back to sequence for residual connection
        conv_out_seq = self.patch_embed_rg(conv_out_img)

        return conv_out_seq + x_seq # Residual connection for the group

    # def flops(self): # Add if needed
    #     flops = 0
    #     flops += self.residual_group.flops()
    #     h, w = self.input_resolution
    #     flops += h * w * self.dim * self.dim * 9 # For self.conv
    #     flops += self.patch_embed_rg.flops()
    #     flops += self.patch_unembed_rg.flops()
    #     return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size) # patch_size for MambaIR is typically 1
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # self.img_size = img_size # Not strictly needed after calculating patches_resolution
        # self.patch_size = patch_size # Not strictly needed
        # self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] # Not strictly needed

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # The original MambaIR uses patch_size=1.
        # If patch_size=1, no actual patching (conv projection) is done.
        # It's mainly for reshaping and optional LayerNorm.
        # If patch_size > 1, a conv projection would be typical (e.g., SwinTransformer)
        # MambaIR's PatchEmbed with patch_size=1 doesn't have a projection conv.
        # It expects input 'x' to already have 'embed_dim' channels if norm_layer is applied to it.
        # Or, it's used as a utility to reshape.

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x): # x: (B, C, H, W)
        # If patch_size is 1, H_patch = H, W_patch = W. L = H*W.
        # x.flatten(2) -> (B, C, L)
        # .transpose(1,2) -> (B, L, C)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        # h, w = self.img_size # If img_size is needed, ensure it's stored
        # For MambaIR, patches_resolution is more relevant for flops if norm is applied
        if self.norm is not None:
            flops += self.patches_resolution[0] * self.patches_resolution[1] * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size) # MambaIR uses patch_size=1
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans # Not directly used in forward if patch_size=1
        self.embed_dim = embed_dim

    def forward(self, x_seq, x_size): # x_seq: (B, L, C), x_size: (H, W)
        # B, L, C = x_seq.shape
        # x_seq.transpose(1,2) -> (B, C, L)
        # .view(B, C, H, W)
        x = x_seq.transpose(1, 2).view(x_seq.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class MambaIR_Refactored_DualBranch(nn.Module):
    def __init__(self,
                 img_size=256,
                 patch_size=1, # Crucial for MambaIR's PatchEmbed/UnEmbed behavior
                 in_chans=400,   # Changed from 400 to 3 for common SR
                 embed_dim=180, # Original MambaIR embed_dim
                 # depths replaced by dual branch depths
                 shallow_group_depths=(4, 4), # Num VSSBlocks per ResidualGroup in shallow branch
                 deep_group_depths=(6, 6, 6, 6), # Num VSSBlocks per ResidualGroup in deep branch
                 mlp_ratio=2.,   # For SS2D expand and VSSBlock
                 drop_rate=0.,   # Dropout for pos_drop
                 drop_path_rate=0.1, # Stochastic depth for VSSBlocks (overall rate)
                 norm_layer=nn.LayerNorm,
                 patch_norm=True, # Whether to use norm in the main PatchEmbed
                 use_checkpoint=True,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 d_state=16, # d_state for SS2D
                 num_out_chans=2, # If None, defaults to in_chans
                 **kwargs):
        super(MambaIR_Refactored_DualBranch, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.img_range = img_range
        self.use_checkpoint = use_checkpoint
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_norm = patch_norm
        self.upscale = upscale
        self.upsampler = upsampler

        if num_out_chans is None:
            num_out_ch = in_chans
        else:
            num_out_ch = num_out_chans

        # Mean initialization (same as original MambaIR)
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, in_chans, 1, 1) # Adjusted for in_chans

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep Feature Extraction (Dual Branch)
        # Main PatchEmbed and PatchUnEmbed
        # These operate on the output of conv_first (H,W of original image if patch_size=1)
        # and convert between (B,C,H,W) and (B,L,C)
        self.patch_embed = PatchEmbed(
            img_size=img_size, # Original image size
            patch_size=patch_size, # Usually 1 for MambaIR
            in_chans=embed_dim, # Input channels to PatchEmbed = embed_dim
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # patches_resolution of the feature map *after* conv_first (and after patch_embed if patch_size > 1)
        # For MambaIR with patch_size=1, this is effectively (img_size, img_size)
        self.body_patches_resolution = self.patch_embed.patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, # Original image size
            patch_size=patch_size, # Usually 1
            embed_dim=embed_dim) # Output channels from PatchUnEmbed = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth calculation for all VSSBlocks across both branches
        total_vss_blocks = sum(s_depth * len(shallow_group_depths) for s_depth in shallow_group_depths) if isinstance(shallow_group_depths[0], int) else sum(shallow_group_depths)
        total_vss_blocks += sum(d_depth * len(deep_group_depths) for d_depth in deep_group_depths) if isinstance(deep_group_depths[0], int) else sum(deep_group_depths)
        
        # Corrected calculation for dpr if depths are lists/tuples of #VSSBlocks per group
        # Original MambaIR 'depths' was a tuple where each element is the num VSSBlocks for one ResidualGroup.
        # E.g., depths=(6,6,6,6) means 4 ResidualGroups, each with 6 VSSBlocks.
        # We assume shallow_group_depths and deep_group_depths follow this:
        # e.g., shallow_group_depths = (4,4) means 2 RGs in shallow, each with 4 VSSBlocks.

        num_total_vss_blocks_in_shallow = sum(shallow_group_depths)
        num_total_vss_blocks_in_deep = sum(deep_group_depths)
        grand_total_vss_blocks = num_total_vss_blocks_in_shallow + num_total_vss_blocks_in_deep
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, grand_total_vss_blocks)]

        is_light_model = self.upsampler == 'pixelshuffledirect' or embed_dim < 100 # Heuristic for light model

        # --- Shallow Branch ---
        self.shallow_layers = nn.ModuleList()
        dpr_idx_offset = 0
        for i_group in range(len(shallow_group_depths)):
            num_vss_blocks_in_this_group = shallow_group_depths[i_group]
            self.shallow_layers.append(ResidualGroup(
                dim=embed_dim,
                input_resolution=self.body_patches_resolution, # H, W of the feature map for VSSBlock
                depth=num_vss_blocks_in_this_group, # Num VSSBlocks in this RG
                mlp_ratio=mlp_ratio,
                drop_path=dpr[dpr_idx_offset : dpr_idx_offset + num_vss_blocks_in_this_group],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                img_size=self.body_patches_resolution, # For PatchEmbed/UnEmbed inside RG
                patch_size=1,                     # For PatchEmbed/UnEmbed inside RG
                resi_connection=resi_connection,
                d_state=d_state,
                is_light_sr=is_light_model
            ))
            dpr_idx_offset += num_vss_blocks_in_this_group

        # --- Deep Branch ---
        self.deep_layers = nn.ModuleList()
        # dpr_idx_offset is already advanced from shallow branch
        for i_group in range(len(deep_group_depths)):
            num_vss_blocks_in_this_group = deep_group_depths[i_group]
            self.deep_layers.append(ResidualGroup(
                dim=embed_dim,
                input_resolution=self.body_patches_resolution,
                depth=num_vss_blocks_in_this_group,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[dpr_idx_offset : dpr_idx_offset + num_vss_blocks_in_this_group],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                img_size=self.body_patches_resolution,
                patch_size=1,
                resi_connection=resi_connection,
                d_state=d_state,
                is_light_sr=is_light_model
            ))
            dpr_idx_offset += num_vss_blocks_in_this_group

        self.norm_after_branches = norm_layer(embed_dim) # Norm for the merged sequence

        # Convolution after body (operates on image-like tensor)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # 3. Restoration Module (Upsampling)
        num_feat_upsample = 64 # As in original MambaIR
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat_upsample, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat_upsample)
            self.conv_last = nn.Conv2d(num_feat_upsample, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.conv_before_upsample = nn.Identity() # Not used
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
            self.conv_last = nn.Identity() # Not used
        else: # Denoising
            self.conv_before_upsample = nn.Identity()
            self.upsample = nn.Identity()
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

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
    def no_weight_decay(self): # Matches original MambaIR
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self): # Matches original MambaIR
        return {'relative_position_bias_table'}

    def _forward_branch(self, x_seq, x_size, layers_branch):
        """Helper to forward a single branch."""
        for layer_group in layers_branch:
            if self.use_checkpoint:
                x_seq = checkpoint.checkpoint(layer_group, x_seq, x_size)
            else:
                x_seq = layer_group(x_seq, x_size)
        return x_seq

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x_in_range = (x - self.mean) * self.img_range

        # 1. Shallow feature extraction
        x_first = self.conv_first(x_in_range) # (B, embed_dim, H, W)
        
        # Get spatial dimensions for VSSBlocks/PatchUnEmbed
        # This is H,W of the feature map after conv_first
        current_x_size = (x_first.shape[2], x_first.shape[3])

        # 2. Deep feature extraction - Dual Branch
        # Convert to sequence: (B, L, C) where L = H*W
        x_seq = self.patch_embed(x_first)
        x_seq = self.pos_drop(x_seq)

        # Shallow Branch
        x_shallow = self._forward_branch(x_seq, current_x_size, self.shallow_layers)

        # Deep Branch
        x_deep = self._forward_branch(x_seq, current_x_size, self.deep_layers)

        # Merge branches
        x_merged_seq = x_shallow + x_deep
        x_normed_seq = self.norm_after_branches(x_merged_seq)

        # Convert back to image-like tensor: (B, embed_dim, H, W)
        x_body_out_img = self.patch_unembed(x_normed_seq, current_x_size)
        
        # Convolution after body and global residual connection
        # This 'res' is the input to the upsampling block
        res_features = self.conv_after_body(x_body_out_img) + x_first

        # 3. Restoration
        if self.upsampler == 'pixelshuffle':
            up_x = self.conv_before_upsample(res_features)
            up_x = self.upsample(up_x)
            output = self.conv_last(up_x)
        elif self.upsampler == 'pixelshuffledirect':
            output = self.upsample(res_features) # UpsampleOneStep handles conv_last implicitly
        else: # Denoising
            output = self.conv_last(res_features)
        
        output = output / self.img_range 
        return output

    # def flops(self): # Implement if needed, summing flops from all components
    #     flops = 0
    #     # ... add flops for conv_first, patch_embed, branches, norm, patch_unembed, conv_after_body, upsampler parts
    #     return flops


# Example builder functions (similar to original MambaIR)
def buildMambaIR_dual_refactored(upscale=2, img_size=64, in_chans=3, num_out_chans=None):
    if num_out_chans is None:
        num_out_chans = in_chans
    return MambaIR_Refactored_DualBranch(
                   img_size=img_size,
                   patch_size=1,
                   in_chans=in_chans,
                   num_out_chans=num_out_chans,
                   embed_dim=180,
                   shallow_group_depths=(6, 6),       # Example: 2 RGs, 6 VSSBlocks each
                   deep_group_depths=(6, 6, 6, 6), # Example: 4 RGs, 6 VSSBlocks each
                                                      # Total 6 RGs, (2+4)*6 = 36 VSSBlocks
                                                      # Matches original MambaIR depths=(6,6,6,6,6,6)
                   mlp_ratio=2.,
                   drop_rate=0.,
                   drop_path_rate=0.1, # Original MambaIR usually has 0.1 for SwinIR compat.
                   norm_layer=nn.LayerNorm,
                   patch_norm=True,
                   use_checkpoint=False,
                   upscale=upscale,
                   img_range=1.,
                   upsampler='pixelshuffle',
                   resi_connection='1conv',
                   d_state=16)


def buildMambaIR_light_dual_refactored(upscale=2, img_size=64, in_chans=3, num_out_chans=None):
    if num_out_chans is None:
        num_out_chans = in_chans
    return MambaIR_Refactored_DualBranch(
                   img_size=img_size,
                   patch_size=1,
                   in_chans=in_chans,
                   num_out_chans=num_out_chans,
                   embed_dim=60,
                   shallow_group_depths=(6, 6),     # Example: 2 RGs, 6 VSSBlocks each
                   deep_group_depths=(6, 6),      # Example: 2 RGs, 6 VSSBlocks each
                                                    # Total 4 RGs, (2+2)*6 = 24 VSSBlocks
                                                    # Matches original light MambaIR depths=(6,6,6,6)
                   mlp_ratio=1.5, # Matches original light
                   drop_rate=0.,
                   drop_path_rate=0.1,
                   norm_layer=nn.LayerNorm,
                   patch_norm=True,
                   use_checkpoint=False,
                   upscale=upscale,
                   img_range=1.,
                   upsampler='pixelshuffledirect',
                   resi_connection='1conv',
                   d_state=16)
