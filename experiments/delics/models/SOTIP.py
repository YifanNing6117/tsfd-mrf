import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# ==============================================================================
# SOTIP Component 1: 3-Level 2D U-Net
# ==============================================================================
class DoubleConv(nn.Module):
    """(Conv -> GroupNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = 64 if out_channels >= 64 and out_channels % 64 == 0 else out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SOTIP_UNet(nn.Module):
    """
    3-level 2D U-Net (no residual connection, GroupNorm 64 groups)
    Input:  [B*Z, K, H, W]
    Output: [B*Z, K, H, W]
    """
    def __init__(self, in_channels=5, base_features=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))

        self.up1 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 4, base_features * 2)
        self.up2 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 2, base_features)

        self.outc = nn.Conv2d(base_features, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        u2 = self.up1(x3)
        d2 = self.dec1(torch.cat([x2, u2], dim=1))

        u1 = self.up2(d2)
        d1 = self.dec2(torch.cat([x1, u1], dim=1))

        return self.outc(d1)


# ==============================================================================
# SOTIP Component 2: Voxel-wise Parameter Estimation Network
# ==============================================================================
class Voxelwise_FCNN(nn.Module):
    """
    3-layer FCNN implemented as 1x1 Conv: 512 -> 512 -> 2
    末层 Sigmoid 将输出约束到 [0, 1]，需配合标签归一化 (T1/4000, T2/1500)
    如果你的标签是原始 ms 值，将 Sigmoid 换成 Softplus
    """
    def __init__(self, in_channels=5, out_channels=2):
        super().__init__()
        self.fcnn = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, kernel_size=1),
            nn.Sigmoid()  # 输出约束到 [0,1]，匹配归一化后的 T1/T2 标签
        )

    def forward(self, x):
        return self.fcnn(x)


# ==============================================================================
# SOTIP Main Network
# ==============================================================================
class SOTIP_Net(nn.Module):
    """
    完整的 SOTIP 等效网络 (图像域版本)

    Pipeline:
        [B, 400, H, W] 
            -> temporal_proj (1x1 Conv3d, 正交初始化) -> [B, K, 1, H, W]
            -> 3x Unrolled { U-Net denoise + Soft DC } 
            -> L2 normalize 
            -> Voxelwise FCNN -> [B, 2, H, W]

    与论文的对应关系:
        - temporal_proj     ↔ PCA/SVD 子空间投影 (论文用物理字典, 这里端到端学习)
        - SOTIP_UNet x3     ↔ 3-Unrolls CNN denoiser (权重不共享)
        - Soft DC           ↔ k-space DC (论文用 NUFFT, 这里用图像域软约束)
        - Voxelwise_FCNN    ↔ 逐体素参数估计网络
    """
    def __init__(self,
                 in_chans=400,
                 num_out_chans=2,
                 num_unrolls=3,
                 K=5,
                 use_checkpoint=True,
                 **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_out_chans = num_out_chans
        self.K = K

        # 1. 时间子空间投影 (400 -> K)
        self.temporal_proj = nn.Conv3d(in_chans, K, kernel_size=1, bias=False)

        # 2. 级联 U-Net (权重不共享，论文原文明确要求)
        self.unrolls = nn.ModuleList([
            SOTIP_UNet(in_channels=K, base_features=64) for _ in range(num_unrolls)
        ])

        # 3. Soft DC 步长 (每个 cascade 独立, 可学习)
        #    初始化 -2.0 -> sigmoid(-2.0) ≈ 0.12
        #    含义: 初始阶段 88% 信任网络输出 + 12% 锚定原始输入
        self.alphas = nn.Parameter(torch.full((num_unrolls,), -2.0))

        # 4. 体素级参数估计
        self.parameter_estimator = Voxelwise_FCNN(in_channels=K, out_channels=num_out_chans)

        # 分层初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """
        分层初始化策略:
        - U-Net 卷积层:     Kaiming (配合 ReLU)
        - GroupNorm:         weight=1, bias=0
        - temporal_proj:     正交初始化 (模拟 PCA 基的正交性先验)
        - FCNN 输出层:       Xavier (配合 Sigmoid)
        """
        # --- U-Net + FCNN 隐层: Kaiming ---
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # --- temporal_proj: 正交初始化 ---
        # weight shape: [K, 400, 1, 1, 1], 需要 reshape 成 [K, 400] 做正交初始化
        with torch.no_grad():
            w = self.temporal_proj.weight.squeeze(-1).squeeze(-1).squeeze(-1)  # [K, 400]
            nn.init.orthogonal_(w)
            self.temporal_proj.weight.copy_(w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

        # --- FCNN 输出层: Xavier (后接 Sigmoid, 不是 ReLU) ---
        with torch.no_grad():
            for module in reversed(list(self.parameter_estimator.fcnn)):
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                    break

    def forward(self, x):
        """
        Args:
            x: [B, 400, H, W] 或 [B, 400, Z, H, W]
               400 个时间点的 MRF 幅值图序列

        Returns:
            [B, 2, H, W] 或 [B, 2, Z, H, W]
            通道 0: T1 估计值, 通道 1: T2 估计值
        """
        # --- 维度兼容: 纯 2D 输入自动补 Z=1 ---
        squeezed = False
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, 400, H, W] -> [B, 400, 1, H, W]
            squeezed = True

        # === Step 1: 时间子空间压缩 ===
        # [B, 400, Z, H, W] -> [B, K, Z, H, W]
        c_3d = self.temporal_proj(x)
        c_init_3d = c_3d.clone()  # 锚点, 用于 Soft DC
        B, K, Z, H, W = c_3d.shape

        # === Step 2: 级联去噪 + Soft DC ===
        for i, unet in enumerate(self.unrolls):
            # 3D -> 2D: [B, K, Z, H, W] -> [B*Z, K, H, W]
            c_2d = c_3d.permute(0, 2, 1, 3, 4).reshape(B * Z, K, H, W)

            # 逐通道强度标准化 (论文要求, 因为不同子空间基底强度差异大)
            mean = c_2d.mean(dim=[2, 3], keepdim=True)
            std = c_2d.std(dim=[2, 3], keepdim=True) + 1e-6
            c_norm = (c_2d - mean) / std

            # U-Net 去噪 (带梯度检查点)
            if self.use_checkpoint and self.training:
                z_2d = checkpoint.checkpoint(unet, c_norm, use_reentrant=False)
            else:
                z_2d = unet(c_norm)

            # 反标准化
            z_2d = z_2d * std + mean

            # 2D -> 3D: [B*Z, K, H, W] -> [B, K, Z, H, W]
            z_3d = z_2d.view(B, Z, K, H, W).permute(0, 2, 1, 3, 4)

            # Soft Data Consistency
            # c_new = (1-alpha) * z_denoised + alpha * c_init
            alpha = torch.sigmoid(self.alphas[i])
            c_3d = z_3d - alpha * (z_3d - c_init_3d)

        # === Step 3: 逐体素参数估计 ===
        # 3D -> 2D
        c_final_2d = c_3d.permute(0, 2, 1, 3, 4).reshape(B * Z, K, H, W)

        # 逐像素 L2 归一化 (消除质子密度等比例因子, 论文 Variant 1 做法)
        l2_norm = torch.norm(c_final_2d, p=2, dim=1, keepdim=True) + 1e-8
        c_final_2d = c_final_2d / l2_norm

        # FCNN: [B*Z, K, H, W] -> [B*Z, 2, H, W]
        out_params_2d = self.parameter_estimator(c_final_2d)

        # 2D -> 3D: [B*Z, 2, H, W] -> [B, 2, Z, H, W]
        out_params_3d = out_params_2d.view(B, Z, self.num_out_chans, H, W).permute(0, 2, 1, 3, 4)

        # 如果原始输入是纯 2D, 去掉补充的 Z 维度
        if squeezed:
            out_params_3d = out_params_3d.squeeze(2)  # [B, 2, 1, H, W] -> [B, 2, H, W]

        return out_params_3d

