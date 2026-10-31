# options/train_options_wyz.py
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self) # 调用父类初始化 (已包含修改后的MRF SR默认值)

        # --- 训练流程控制参数 (大部分保留，可以调整默认值) ---
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc') # 通常由脚本逻辑设置，非用户命令行指定
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        
        # --- 学习率和优化器参数 ---
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate (full LR epochs)')
        self.parser.add_argument('--niter_decay', type=int, default=150, help='# of iter to linearly decay learning rate to zero (decay epochs)')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam (beta1 for Adam)') # Adam的beta1通常设为0.9
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters epochs (for step policy)')
        self.parser.add_argument('--use_amp', action='store_true', help='Enable Automatic Mixed Precision (AMP) training')
        # self.parser.add_argument('--trans_lr_coef', type=float, default=1, help='transformer learning rate coefficient (if used)') # 用途不明，若不用可删

        # --- 移除的GAN和CycleGAN相关损失权重 ---
        # self.parser.add_argument('--no_lsgan', ...)
        # self.parser.add_argument('--lambda_A', ...) # 已被 lambda_pixel 替代
        # self.parser.add_argument('--lambda_B', ...)
        # self.parser.add_argument('--lambda_f', ...)
        # self.parser.add_argument('--lambda_identity', ...)
        # self.parser.add_argument('--pool_size', ...)
        # self.parser.add_argument('--lambda_adv', ...)

        # --- 暂时移除的VGG感知损失参数 ---
        # self.parser.add_argument('--lambda_vgg', ...)
        # self.parser.add_argument('--vgg_layer', ...)

        # --- 新增/修改的像素损失参数 ---
        self.parser.add_argument('--pixel_loss_type', type=str, default='l1', choices=['l1', 'l2', 'smooth_l1'], help='type of pixel-wise loss (l1 | l2 | smooth_l1)')
        self.parser.add_argument('--lambda_pixel', type=float, default=1.0, help='weight for the pixel-wise supervision loss (e.g., 1.0, 10.0, 100.0)')

        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        
        # mask选项 (如果你的像素损失计算不再使用mask，这个也可以移除)
        # 如果 ResViT_model.backward_G 中不再使用 self.opt.mask, 则可移除
        self.parser.add_argument('--mask', type=str, default="", help='path to mask file or mask type (if used in loss or data processing)')


        self.isTrain = True
