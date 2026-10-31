# options/base_options.py
import argparse
import os
from util1 import util # 确保 util1 可导入
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # --- 核心数据和模型参数 ---
        self.parser.add_argument('--dataroot', required=True, help='path to dataset (e.g., a .csv file for MRF)')
        self.parser.add_argument('--name', type=str, default='mrf_mambair_experiment', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # --- 数据加载相关参数 (适配MRF SR) ---
        self.parser.add_argument('--dataset_mode', type=str, default='aligned2', help='chooses how datasets are loaded. e.g., aligned2 for MRF')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=128, help='scale images to this size (for LR input, if no further cropping)')
        self.parser.add_argument('--fineSize', type=int, default=128, help='final size of the LR input image to the network')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset.')
        self.parser.add_argument('--resize_or_crop', type=str, default='none', help='scaling and cropping of images at load time; for MRF usually "none" as handled by loader')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        # --- 模型相关参数 (适配MRF SR 和 MambaIR) ---
        self.parser.add_argument('--model', type=str, default='resvit_one', help='chooses which model_framework to use (e.g., resvit_one)') # 模型封装类
        self.parser.add_argument('--input_nc', type=int, default=400, help='# of input image channels (MRF: 400 timepoints)')
        self.parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels (MRF: 2 parameter maps)')
        self.parser.add_argument('--ngf', type=int, default=96, help='# of gen filters in first conv layer (can map to MambaIR embed_dim)') # 例如默认 embed_dim
        self.parser.add_argument('--which_model_netG', type=str, default='mambaIR', help='selects model to use for netG')
        self.parser.add_argument('--norm', type=str, default='none', help='instance normalization or batch normalization or none. MambaIR uses LayerNorm internally.') # 对于MambaIR，外部norm可能不需要
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator (MambaIR has its own drop_rate)')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]') # MambaIR有自己的初始化
        self.parser.add_argument('--upscale_factor', type=int, default=2, help='upscale factor for super-resolution') # MRF SR 128 -> 256
        
        # --- 移除的判别器 (netD) 相关参数 ---
        # self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        # self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        # self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        
        # --- 其他模型可能用到的参数 (对于MambaIR，其预训练逻辑在其内部或 define_G 中处理) ---
        self.parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='ViT configuration name (if using ViT-based generator)')
        self.parser.add_argument('--pre_trained_path', type=str, default='', help='path to pre-trained weights (if applicable)')
        self.parser.add_argument('--pre_trained_transformer', type=int, default=0, help='flag to use pre-trained transformer weights (if applicable)')
        self.parser.add_argument('--pre_trained_resnet', type=int, default=0, help='flag to use pre-trained resnet weights (if applicable)')

        # --- 显示和调试参数 ---
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display (0 for no visdom)')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--debug', type=int, default=0, help='Debug mode (e.g., use limited samples)')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA (usually AtoB for SR)')


        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
