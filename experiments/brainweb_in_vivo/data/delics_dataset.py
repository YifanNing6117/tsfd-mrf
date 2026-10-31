import os
import pandas as pd
import torch
import torchio as tio
from data.base_dataset import BaseDataset

class MambamrfDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot  # 此时 dataroot 传入的应该是新生成的 delics_dataset.csv 路径
        self.df = pd.read_csv(self.root)
        
        # 按照新 CSV 的 'fold' 列过滤当前阶段的数据 (train / test)
        self.df = self.df.loc[self.df['fold'] == opt.phase]
        
        # 保留你的 debug 逻辑
        if hasattr(self.opt, 'debug') and self.opt.debug == 1:
            self.df = self.df.head(256)

        print(f"---------------------------{opt.phase} images: {len(self.df)} (Mamba MRF Dataset) ---------------------------")

    def __getitem__(self, index):
        # 1. 解析新版 7列 CSV
        row = self.df.iloc[index]
        fold = row['fold']
        case_id = row['case_id']
        slice_idx = row['slice_idx']
        input_path = row['input_path']
        t1_path = row['t1_path']
        t2_path = row['t2_path']
        mask_path = row['mask_path']

        # 动态读取命令行配置的输入通道数 (例如 t=400 或 500)
        t = self.opt.input_nc 

        # =========================================================
        # 2. 读取 Input A (指纹图)
        # NIfTI 存的形状: (1, 256, 256, 500)
        # =========================================================
        AA = tio.ScalarImage(input_path).data[:, :, :, 0:t]
        AA = torch.squeeze(AA, dim=0)
        A = torch.transpose(AA, 0, 2)
        A = torch.transpose(A, 1, 2) # 完全保留你之前的维度交换逻辑 -> (t, 256, 256)

        # =========================================================
        # 3. 读取 Target B (T1) 和 C (T2)
        # NIfTI 存的形状: (1, 256, 256, 1)
        # =========================================================
        BB = tio.ScalarImage(t1_path).data
        B = torch.squeeze(BB, dim=3) # -> (1, 256, 256)

        CC = tio.ScalarImage(t2_path).data
        C = torch.squeeze(CC, dim=3) # -> (1, 256, 256)

        # =========================================================
        # 4. 读取官方纯大脑 Mask
        # NIfTI 存的形状: (1, 256, 256, 1)
        # =========================================================
        MM = tio.ScalarImage(mask_path).data
        mask = torch.squeeze(MM, dim=3) # -> (1, 256, 256)

        # =========================================================
        # 5. 绝对效忠你的老逻辑：单张切片 Min-Max 归一化
        # (加了 1e-8 防止除以0报错)
        # =========================================================
        A = (A - A.min()) / (A.max() - A.min() + 1e-8)
        B = (B - B.min()) / (B.max() - B.min() + 1e-8)
        C = (C - C.min()) / (C.max() - C.min() + 1e-8)

        # 拼接 T1 和 T2 -> 形状 (2, 256, 256)
        B = torch.cat((B, C), dim=0)

        return {
            'A': A, 
            'B': B, 
            'mask': mask,            # <--- 核心新增：把掩码传给模型！
            'A_paths': input_path, 
            'B_paths': t1_path, 
            'slice_idx': int(slice_idx)
        }

    def __len__(self):
        return len(self.df)

    def name(self):
        return 'DelicsDataset'

