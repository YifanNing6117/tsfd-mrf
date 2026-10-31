import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat

# ==========================================
# 1. 显卡设置 (强制使用 GPU 2)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只让程序看到 2 号卡，它会变成逻辑上的cuda:0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"�� Running on: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. 核心依赖检查
# ==========================================
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("✅ Success: mamba_ssm.ops.selective_scan_interface found.")
except ImportError:
    print("❌ Error: mamba_ssm not found. Cannot proceed.")
    exit()

# ==========================================
# 3. SimpleMamba1D (手搓版)
# ==========================================
class SimpleMamba1D(nn.Module):
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
        """ x: [Batch, Length, Dim] """
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

# ==========================================
# 4. BiTemporalScan Wrapper (你的目标模块)
# ==========================================
class BiTemporalScanTest(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_inner = 16 
        self.in_proj = nn.Linear(1, self.d_inner)
        self.mamba = SimpleMamba1D(d_model=self.d_inner)
        self.out_proj = nn.Linear(self.d_inner * 2, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(B*H*W, C, 1)
        x_embed = self.in_proj(x_seq)
        
        x_fwd = x_embed
        x_bwd = torch.flip(x_embed, [1])
        
        x_in = torch.cat([x_fwd, x_bwd], dim=0)
        x_out = self.mamba(x_in)
        
        out_fwd, out_bwd = x_out.chunk(2, dim=0)
        out_bwd = torch.flip(out_bwd, [1])
        
        out_merged = torch.cat([out_fwd, out_bwd], dim=-1)
        res = self.out_proj(out_merged)
        res = res.view(B, H, W, C).permute(0, 3, 1, 2)
        return res

# ==========================================
# 5. 执行测试
# ==========================================
if __name__ == "__main__":
    print("\n--- Testing SimpleMamba1D & BiTemporalScan on GPU 2 ---")
    
    # 模拟数据
    x = torch.randn(1, 400, 32, 32).to(DEVICE)
    
    model = BiTemporalScanTest(dim=400).to(DEVICE)
    
    try:
        out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        
        if x.shape == out.shape:
            print("✅ Test Passed: Shape matches!")
        else:
            print("❌ Test Failed: Shape mismatch!")
            
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
