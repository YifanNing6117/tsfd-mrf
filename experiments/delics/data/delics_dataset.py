import os
import pandas as pd
import torch
import torchio as tio
from data.base_dataset import BaseDataset

class DelicsDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.df = pd.read_csv(self.root)
        
        self.df = self.df.loc[self.df['fold'] == opt.phase]
        
        if hasattr(self.opt, 'debug') and self.opt.debug == 1:
            self.df = self.df.head(256)

        print(f"---------------------------{opt.phase} images: {len(self.df)} (Mamba MRF Dataset) ---------------------------")

    def __getitem__(self, index):
        row = self.df.iloc[index]
        input_path = row['input_path']
        t1_path = row['t1_path']
        t2_path = row['t2_path']
        mask_path = row['mask_path']
        slice_idx = row['slice_idx']

        # 从命令行动态读取 input_nc (你在命令行配了什么就是什么，比如 400)
        t = self.opt.input_nc 

        # =========================================================
        # �� 修复 5D 怪物错误：干净利落的张量提取
        # =========================================================
        
        # 1. Input A: Torchio 会把 (1, 256, 256, 500) 读成 [500, 1, 256, 256]
        AA = tio.ScalarImage(input_path).data
        # 直接提取前 t 个时间点，并丢弃 W=1 那个维度 -> 得到 [t, 256, 256]
        A = AA[0:t, 0, :, :].clone()

        # 2. Target B & C: Torchio 会把 (1, 256, 256, 1) 读成 [1, 1, 256, 256]
        BB = tio.ScalarImage(t1_path).data
        # 提取第0个Channel，第0个W，并强行扩展第一维 -> 得到 [1, 256, 256]
        B = BB[0, 0, :, :].unsqueeze(0).clone()

        CC = tio.ScalarImage(t2_path).data
        C = CC[0, 0, :, :].unsqueeze(0).clone()

        # 3. 官方 Mask 同样提取 -> 得到 [1, 256, 256]
        MM = tio.ScalarImage(mask_path).data
        mask = MM[0, 0, :, :].unsqueeze(0).clone()

        # =========================================================
        # �� 全局物理极值归一化 (彻底消灭 T2 误差折叠)
        # =========================================================
        # Input A (指纹): 保持单切片最大值归一化，以抵抗 B1 偏置场的影响
        A = A / (A.max() + 1e-8) 
        
        # Target B & C: 严格按照人体物理极限缩放到 0~1
        B = B / 5000.0  # T1 物理极限是 5000 ms
        C = C / 4000.0  # T2 物理极限是 4000 ms
        
        # 安全裁剪，防止有极个别异常噪点超过极限导致越界
        B = torch.clamp(B, 0.0, 1.0)
        C = torch.clamp(C, 0.0, 1.0)


        # 拼接 T1 和 T2 -> 最终形状 [2, 256, 256]
        B = torch.cat((B, C), dim=0)

        return {
            'A': A, 
            'B': B, 
            'mask': mask,
            'A_paths': input_path, 
            'B_paths': t1_path, 
            'slice_idx': int(slice_idx)
        }

    def __len__(self):
        return len(self.df)

    def name(self):
        return 'DelicsDataset'

