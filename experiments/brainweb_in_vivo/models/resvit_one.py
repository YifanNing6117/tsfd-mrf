import torch
from collections import OrderedDict
from torch.autograd import Variable # Variable 在较新的PyTorch中已不推荐，可以直接使用tensor
import util1.util as util
# from util1.image_pool import ImagePool # ImagePool不再需要，可以删除这行导入
from .base_model import BaseModel
from . import networks
# from torchvision import models # 如果没用到models.vgg等，可以删除

class ResViT_model(BaseModel):
    def name(self):
        return 'ResViT_model'

    # initialize 方法是和 name 方法同级的
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # --- 在这里添加 self.device 的定义 (如果父类没有定义) ---
        if not hasattr(self, 'device'): # 检查父类是否已经定义了device
            if len(opt.gpu_ids) > 0:
                self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
            else:
                self.device = torch.device('cpu')

        # -------------------- 网络定义 --------------------
        # netG 的定义保持不变，它由 define_G 创建
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG,opt.vit_name,opt.fineSize,opt.pre_trained_path, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            print(f'netG loaded !!!')
            # # if self.isTrain:
            # #     self.load_network(self.netD, 'D', opt.which_epoch)
            # #     print(f'netD loaded !!!')

        # -------------------- 训练相关设置 --------------------
        if self.isTrain:
            # --- 删除 fake_AB_pool ---
            # # self.fake_AB_pool = ImagePool(opt.pool_size)

            # --- 定义像素损失函数 ---
            if opt.pixel_loss_type == 'l1':
                self.criterionPixel = torch.nn.L1Loss()
            elif opt.pixel_loss_type == 'l2':
                self.criterionPixel = torch.nn.MSELoss()
            elif opt.pixel_loss_type == 'smooth_l1':
                self.criterionPixel = torch.nn.SmoothL1Loss()
            else:
                raise NotImplementedError('Pixel loss type [%s] is not recognized for criterionPixel' % opt.pixel_loss_type)
            
            # --- 设置损失名称 (用于日志和可视化) ---
            self.loss_names = ['G_pixel']

            # --- 定义优化器 (只有G的优化器) ---
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G] # 只包含G的优化器

            # --- 定义学习率调度器 ---
            self.schedulers = []
            for optimizer in self.optimizers: # 这个循环现在只作用于 optimizer_G
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # --- 设置模型名称 (用于保存) ---
        self.model_names = ['G'] # 只包含生成器

        print('------------ Networks initialized -------------')
        networks.print_network(self.netG)
        # # if self.isTrain: # --- 移除 netD 打印 ---
        # #     networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input_i_j(self, input,i,j): # 这个方法如果不用，可以考虑删除，以简化代码
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_A = input_A[..., i*64:64*(i+1)+16, j*64:(j+1)*64+16]
        input_B = input['B' if AtoB else 'A']
        input_B = input_B[..., i*64:64*(i+1)+16, j*64:(j+1)*64+16]
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
        self.input_A = input_A.float()
        self.input_B = input_B.float()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.slice_idx = input['slice_idx']

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device).float() # 直接在这里 .to(self.device) 和 .float()
        self.real_B = input['B' if AtoB else 'A'].to(self.device).float()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if 'slice_idx' in input: # 检查是否存在，避免KeyError
            self.slice_idx = input['slice_idx']
        # self.mask = input['mask'].to(self.device).float() # 如果使用带mask的损失，则取消注释并确保数据中有mask

    def forward(self):
        # self.real_A 和 self.real_B 已经在 set_input 中处理过了
        # Variable 也不再需要
        
        # --- 之前的调试打印，确认工作后可以删除或注释 ---
        # print(f"DEBUG: self.real_A.shape = {self.real_A.shape}")
        # ... (其他打印)

        self.fake_B = self.netG(self.real_A)

    def test(self): # 用于验证或测试，不计算梯度
        with torch.no_grad():
            # self.real_A 和 self.real_B 已经在 set_input 中处理过了
            self.fake_B = self.netG(self.real_A)

    def get_image_paths(self):
        return self.image_paths

    def get_slice_idx(self):
        # 确保 self.slice_idx 存在，如果 set_input 可能不总是有 slice_idx
        return getattr(self, 'slice_idx', None)

    # --- backward_D 方法整个删除 ---
    # def backward_D(self):
    #     ...

    def backward_G(self):
        """Calculate pixel loss for the generator"""
        # 使用在 initialize 中定义的 self.criterionPixel 和 opt.lambda_pixel
        # 直接对整个多通道张量计算损失
        self.loss_G_pixel = self.criterionPixel(self.fake_B, self.real_B) * self.opt.lambda_pixel
        
        # 如果需要分通道计算（例如，对于特定任务或调试），可以这样做：
        # self.loss_G_pixel_ch0 = self.criterionPixel(self.fake_B[:,0,...], self.real_B[:,0,...])
        # self.loss_G_pixel_ch1 = self.criterionPixel(self.fake_B[:,1,...], self.real_B[:,1,...])
        # self.loss_G_pixel = (self.loss_G_pixel_ch0 + self.loss_G_pixel_ch1) * (self.opt.lambda_pixel / 2) # 如果分开算再加权平均

        self.loss_G = self.loss_G_pixel # 总损失现在只有像素损失
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # 运行生成器得到 self.fake_B
        # --- G_optimizer ---
        self.optimizer_G.zero_grad()     # 清零G的梯度
        self.backward_G()                # 计算G的损失并反向传播
        self.optimizer_G.step()          # 更新G的权重

    def get_current_errors(self): # 在较新的PyTorch中，.data 不再推荐，直接 .item()
        errors_ret = OrderedDict()
        # 键名应该与 self.loss_names 中的一致
        if hasattr(self, 'loss_G_pixel'): # 确保损失已经计算
            errors_ret[self.loss_names[0]] = self.loss_G_pixel.item() # 假设 loss_names[0] 是 'G_pixel'
        # 如果有分通道的损失记录，也可以添加：
        # if hasattr(self, 'loss_G_pixel_ch0'):
        #     errors_ret['G_pixel_ch0'] = self.loss_G_pixel_ch0.item()
        # if hasattr(self, 'loss_G_pixel_ch1'):
        #     errors_ret['G_pixel_ch1'] = self.loss_G_pixel_ch1.item()
        return errors_ret

    def get_current_visuals(self): # .data 也不再推荐
        real_A_tensor = self.real_A.cpu() # 转换为CPU tensor
        fake_B_tensor = self.fake_B.cpu()
        real_B_tensor = self.real_B.cpu()
        
        # util.tensor2im 可能需要处理 (Batch, 400, H, W) 的 real_A
        # 你可能需要选择一个或几个时间点来可视化 real_A
        # 例如，可视化第一个时间点:
        real_A_display = util.tensor2im(real_A_tensor[:, 0:1, :, :]) # 取第一个时间点并保持通道维度为1
        
        fake_B_display = util.tensor2im(fake_B_tensor)
        real_B_display = util.tensor2im(real_B_tensor)
        return OrderedDict([('real_A_t0', real_A_display), ('fake_B_params', fake_B_display), ('real_B_params', real_B_display)])
        
    # --- 修改后的 train 方法 ---
    def train(self, mode=True):
        """Sets the model to training mode.
        
        Args:
            mode (bool): whether to set training mode (True) or evaluation mode (False).
        """
        # super().train(mode) # <--- 删除或注释掉这一行
        self.isTrain = mode # 你可以保留或添加这一行来显式设置isTrain状态，如果BaseModel或其子类依赖它
        if hasattr(self, 'netG'):
            self.netG.train(mode)
        return self

    # --- 修改后的 eval 方法 ---
    def eval(self):
        """Sets the model to evaluation mode."""
        # super().eval() # <--- 删除或注释掉这一行
        self.isTrain = False # 相应地设置isTrain状态
        if hasattr(self, 'netG'):
            self.netG.eval()
        return self
        

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        # --- 移除 netD 保存 ---
        # # self.save_network(self.netD, 'D', label, self.gpu_ids)
