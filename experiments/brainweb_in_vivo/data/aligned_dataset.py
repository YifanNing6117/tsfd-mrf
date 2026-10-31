import glob
import os.path
import random
import numpy as np
import pandas as pd 
import torchio as tio 
import torch
import torchvision.transforms as transforms
from PIL import Image
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torch.nn as nn
# ----------------------------------------------------------------------------------------------------------------------------------

### 自定义 T2-PD 配对 IXI Dataset
class AlignedDataset2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.df = pd.read_csv(self.root)
        self.df = self.df.loc[self.df['fold'] == opt.phase]  # 获取当前阶段 fold 的数据 (train / valid / test) 

        # wyz
        if self.opt.debug == 1:
            self.df = self.df.head(256)  # for debug

        print(f"---------------------------{opt.phase} images: {len(self.df)} ---------------------------")

    def __getitem__(self, index):
        _, slice_indice,  number, tims, t1,t2 , _  = self.df.iloc[index].tolist()
        
        # 1. 动态获取想要的有效序列长度 (如果命令行没传，默认用200，或者通过 opt 传进来)
        # 这里我们利用命令行的 input_nc，如果命令里写了 --input_nc 200，它就是 200
        t_valid = self.opt.input_nc if hasattr(self.opt, 'input_nc') and self.opt.input_nc != 1 else 400
        # (注：防止默认的1干扰，如果是1就强制改回400；您可以随时在命令里加 --input_nc 200)

        a=0
        
        # 2. 从原始图像里只截取前 t_valid 个有效回波
        AA = tio.ScalarImage(tims).data[:,:,:,0:t_valid]
        AA = torch.squeeze(AA,dim=0)
        A = torch.transpose(AA,0,2)
        A = torch.transpose(A,1,2) # 此时 A 的形状是 [t_valid, 256, 256]

        BB = tio.ScalarImage(t1).data[:,a:256-a,a:256-a]
        B = torch.squeeze(BB,dim=3) 

        CC = tio.ScalarImage(t2).data[:,a:256-a,a:256-a]
        C = torch.squeeze(CC,dim=3)

        # 3. 归一化 (保持您原本的 Min-Max)
        A = (A - A.min()) / (A.max() - A.min() + 1e-8) # 加上1e-8防止分母为0
        B = (B - B.min()) / (B.max() - B.min() + 1e-8)
        C = (C - C.min()) / (C.max() - C.min() + 1e-8)

        # =========================================================
        # 4. 核心魔法：补零到 400 维，骗过网络！
        # =========================================================
        if t_valid < 400:
            pad_A = torch.zeros(400, A.shape[1], A.shape[2])
            pad_A[0:t_valid, :, :] = A
            A = pad_A  # 现在 A 重新变成了 400 维，但后 (400 - t_valid) 维全是 0

        B = torch.cat((B,C),dim=0)

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': tims, 'B_paths': t2, 'slice_idx': int(slice_indice)
               }

    def __len__(self):
        return len(self.df)

    def name(self):
        return 'AlignedDataset2'

# ----------------------------------------------------------------------------------------------------------------------------------

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            tmp = A[1, ...] 
            A = tmp.unsqueeze(0)
        elif input_nc == 2:
            tmp = A[0:2, ...] 
            A = tmp

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            tmp = B[1, ...] 
            B = tmp.unsqueeze(0)
        elif output_nc == 2:
            tmp = B[0:2, ...] 
            B = tmp
        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

