import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util1.visualizer import Visualizer
from util1 import html
from PIL import Image  
from utils import utils_image as util
import torch
import cv2
import lpips

# python test1.py --dataroot MRF.csv  --name diedai50 --gpu_ids 1 --model resvit_one --which_model_netG MambaIR --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 3 --serial_batches --fineSize 220 --loadSize 220 --results_dir zju1/ --checkpoints_dir EXP_checkpoints --which_epoch latest --pre_trained_resnet 0 --pre_trained_transformer 0


# 后续考虑加入的指标
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
# from pytorch_msssim import ssim, ms_ssim

'''
python3 test_wyz.py --dataroot /home/wyz/Desktop/IXI/IXI_process_scripts/data_ixi_t2_pd.csv --name T2_PD_IXI_resvit_v2 --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 12000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest --pre_trained_path ./checkpoints/T2_PD_IXI_resvit_v2/latest_net_G.pth --pre_trained_resnet 0 --pre_trained_transformer 0

python3 test_wyz.py --dataroot /home/wyz/Desktop/IXI/IXI_process_scripts/data_ixi_t2_pd.csv --name T2_PD_IXI_swinres_v0 --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 12000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest --pre_trained_path ./checkpoints/T2_PD_IXI_swinres_v0/latest_net_G.pth --pre_trained_resnet 0 --pre_trained_transformer 0

'''
n=220

def NMSE(x1,x2):
    sum1=0
    sum2=0
    for i in range(n):
        for j in range(n):
            sum1=(x1[...,i,j]-x2[...,i,j])*(x1[...,i,j]-x2[...,i,j])+sum1
            sum2=x1[...,i,j]*x1[...,i,j]+sum2
    return sum1/sum2
# 后续考虑加入的指标
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
# from pytorch_msssim import ssim, ms_ssim

'''
python3 test_wyz.py --dataroot /home/wyz/Desktop/IXI/IXI_process_scripts/data_ixi_t2_pd.csv --name T2_PD_IXI_resvit_v2 --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 12000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest --pre_trained_path ./checkpoints/T2_PD_IXI_resvit_v2/latest_net_G.pth --pre_trained_resnet 0 --pre_trained_transformer 0

python3 test_wyz.py --dataroot /home/wyz/Desktop/IXI/IXI_process_scripts/data_ixi_t2_pd.csv --name T2_PD_IXI_swinres_v0 --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 12000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest --pre_trained_path ./checkpoints/T2_PD_IXI_swinres_v0/latest_net_G.pth --pre_trained_resnet 0 --pre_trained_transformer 0

'''

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flipS

    opt.phase = 'valid'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
#    visualizer = Visualizer(opt)

    # record
    mae_avg = np.zeros([len(dataset)])      
    psnr_avg = np.zeros([len(dataset)])
    ssim_avg = np.zeros([len(dataset)])
    nmse_avg = np.zeros([len(dataset)])



##################################################################################################
##################################################################################################

    # 验证数据个数是否相等
    dataset_size_test = len(dataset)
 
##################################################################################################
##################################################################################################

    # 设置空的数组

    mae_avg = np.zeros([opt.how_many])
    psnr_avg = np.zeros([opt.how_many])
    ssim_avg = np.zeros([opt.how_many])
    nmse_avg = np.zeros([opt.how_many])
    lpips_avg = np.zeros([opt.how_many])

    another_ssim_avg = np.zeros([opt.how_many])
    lips_rec_avg = np.zeros([opt.how_many])

    zf_mae_avg = np.zeros([opt.how_many])
    zf_psnr_avg = np.zeros([opt.how_many])
    zf_ssim_avg = np.zeros([opt.how_many])
    zf_lpips_avg = np.zeros([opt.how_many])

    another_zf_ssim_avg = np.zeros([opt.how_many])
    lips_zf_avg = np.zeros([opt.how_many])

##################################################################################################
##################################################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

##################################################################################################
##################################################################################################


    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        model.test()


        #########################################################################################

        # fake数据是生成的模态数据
        lpips_fake_im_ = model.fake_B
        # print('lpips_fake_im_.shape=',lpips_fake_im_.shape)
        # real数据是真实的模态数据
        lpips_real_im_ = model.real_B
        # print('lpips_real_im_.shape=',lpips_real_im_.shape)
        # 这里的零填充是输入的模态数据
        # lpips_ZF_im = model.real_A
        # print('lpips_ZF_im.shape=',lpips_ZF_im.shape)

        fake_im_ = model.fake_B.squeeze(0).cpu().data.numpy()
        # print(fake_im_.shape)


        #########################################################################################
        #########################################################################################

        # 重建的图像
        import matplotlib.pyplot as plt
        fake_im_ = model.fake_B.squeeze(0).cpu().data.numpy()
  
        # 真实的ground truth图像
        real_im_ = model.real_B.squeeze(0).cpu().data.numpy()

        #########################################################################################
        # 零填充图像
        # ZF_im = model.real_A.squeeze(0).cpu().data.numpy()

        #########################################################################################

        # ssim_fake_im_ = fake_im_.squeeze(0)
        # ssim_real_im_ = real_im_.squeeze(0)
        # # ssim_ZF_im = ZF_im.squeeze(0)

        # another_ssim_avg[i] = util.calculate_ssim_single(ssim_real_im_, ssim_fake_im_)
        # another_zf_ssim_avg[i] = util.calculate_ssim_single(ssim_real_im_, ssim_ZF_im)

        #########################################################################################

        # 重建图像与真实图像的差值
        print(fake_im_.shape)
        mae_avg[i] = abs(fake_im_-real_im_).mean()
        # 重建图像的PSNR
        psnr_avg[i] = psnr(real_im_, fake_im_, data_range=1)
        # 重建图像SSIM
        # ssim_avg[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_im_, (1, 2, 0)), data_range=1, multichannel=True)
        # print('\033[1;31m---test_wyz.py---\033[0m')
        # print("psnr_avg = ", psnr_avg[i])

        #########################################################################################
    

        #########################################################################################

        # 多模态图像与输入零填充图像

        # 辅助模态图像
        # real_assistant = model.real_A.squeeze(0).cpu().data.numpy()

        # 输入零填重图像
        # input_im_ = model.real_A.squeeze(0).cpu().data.numpy()


        #########################################################################################
        #########################################################################################

        # 获取图像名字以及slice id
        img_path = model.get_image_paths()
        slice_idx = model.get_slice_idx()  # wyz
        volumn_name = img_path[0].split('/')[-1].split('.')[0]  # wyz
        print('%04d: process image... %s-%d' % (i, volumn_name, slice_idx))  # wyz

        #########################################################################################
        #########################################################################################

        # 存储图像
        # print('\033[1;31m---test_wyz_v2.py---\033[0m')
        # print("opt.results_dir = ", opt.results_dir)
        # opt.results_dir = EXP_results/

   
        # GT图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'GT'))
        if not isExists:
            os.makedirs(os.path.join(opt.results_dir, 'GT'))
        # 重建图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'Recon'))
        if not isExists:
            os.makedirs(os.path.join(opt.results_dir, 'Recon'))
    
        # 重建差值图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'Different_rec'))
        if not isExists:
             os.makedirs(os.path.join(opt.results_dir, 'Different_rec'))
      
        #########################################################################################
        #########################################################################################

        # 数据压缩，为了存储数据
        w=0
        # 重建的图像
        # print(fake_im_for_save.shape)
        fake_im_for_save = fake_im_[w,...]
        # print(fake_im_for_save.shape)
        # fake_im_for_savet1=fake_im_[0,...]
        # fake_im_for_savet2=fake_im_[1,...]
        # 真实的ground truth图像
        real_im_for_save = real_im_[w,...]
        # ssim_avg[i] = ssim(real_im_for_save, fake_im_for_save, data_range=1, multichannel=True)
        
        
        fake_im_for_save1 = (np.clip(fake_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8 0 2600
        real_im_for_save1 = (np.clip(real_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        fake_im_for_save2 = np.clip(fake_im_for_save, 0, 1) * 255.0
        real_im_for_save2 = np.clip(real_im_for_save, 0, 1) * 255.0
        fake_im_for_save3 = np.clip(fake_im_for_save, 0, 1) 
        real_im_for_save3 = np.clip(real_im_for_save, 0, 1) 
        # fake_im_for_save1 = (np.clip(fake_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8 0 2600
        # real_im_for_save1 = (np.clip(real_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        # fake_im_for_save2 = np.clip(fake_im_for_save, 0, 1) * 255.0
        # real_im_for_save2 = np.clip(real_im_for_save, 0, 1) * 255.0
        # fake_im_for_save3 = np.clip(fake_im_for_save, 0, 1) 
        # real_im_for_save3 = np.clip(real_im_for_save, 0, 1) 
        for t in range(n):
            for tt in range(n):
                if real_im_for_save1[t, tt] == 0:
                    fake_im_for_save1[t, tt] = 0
        for t in range(n):
            for tt in range(n):
                if real_im_for_save2[t, tt] == 0:
                    fake_im_for_save2[t, tt] = 0
        for t in range(n):
            for tt in range(n):
                if real_im_for_save3[t, tt] == 0:
                    fake_im_for_save3[t, tt] = 0     
        print('nmse=',NMSE(real_im_for_save2,fake_im_for_save2))
        nmse_avg[i] = NMSE(real_im_for_save2,fake_im_for_save2)
        # p1=real_im_for_save.unsqueeze(0)
        # print(real_im_for_save.shape)
        # p2=fake_im_for_save.unsqueeze(0)
        ssim_avg[i] = ssim(np.transpose(real_im_for_save,(1,0)), np.transpose(fake_im_for_save,(1,0)), data_range=1, multichannel=True)
        # ssim_avg[i] = ssim(real_im_for_save, fake_im_for_save)
        fake_im_for_saveimg = cv2.applyColorMap(fake_im_for_save1, cv2.COLORMAP_HOT)
        real_im_for_saveimg = cv2.applyColorMap(real_im_for_save1, cv2.COLORMAP_HOT)
        # fake_im_for_saveimg = fake_im_for_save1
        # real_im_for_saveimg = real_im_for_save1

        # fake_im_for_savetxt = fake_im_for_save3 * 1980.0
        # real_im_for_savetxt = real_im_for_save3 * 1980.0

        # print(fake_im_for_save.max())
        X=model.fake_B
        Y=model.real_B
        for t in range(n):
            for tt in range(n):
                if Y[:,0,t, tt] == 0:
                    X[:,0,t, tt] = 0
                    X[:,1,t, tt] = 0
        diff_gen_x10 = torch.mul(torch.abs(torch.sub(X[:,w,:,:],Y[:,w,:,:])), 2)
        # print(diff_gen_x10.shape)
        # diff_gen_x10 = diff_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # diff_gen_x10 = abs(real_im_for_save1-fake_im_for_save1)
        # diff_gen_x10 = (diff_gen_x10).round().astype(np.uint8)  # float32 to uint8
        # diff_gen_x10 = diff_gen_x10.numpy()
        diff_gen_x10 = diff_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_gen_x10 = (diff_gen_x10 * 255.0).astype(np.uint8)  # float32 to uint8
        diff_gen_x10_color = cv2.applyColorMap(diff_gen_x10, cv2.COLORMAP_HOT)
        
        #########################################################################################
        #########################################################################################
        # 图像名字
        # input_image_dir = os.path.join(opt.results_dir, 'ZF' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
        fake_image_dir = os.path.join(opt.results_dir, 'Recon' + "/"+ str(volumn_name) + str(slice_idx.numpy()) + ".png")
        real_image_dir = os.path.join(opt.results_dir, 'GT' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
        # assistant_image_dir = os.path.join(opt.results_dir, 'Assist' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
        diff_gen_image_dir = os.path.join(opt.results_dir, 'Different_rec' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
        # diff_lq_image_dir = os.path.join(opt.results_dir, 'Different_zero' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")

        fake_image_dir1 = os.path.join(opt.results_dir, 'Recon' + "/"+ str(volumn_name) + str(slice_idx.numpy()) + ".txt")
        real_image_dir1 = os.path.join(opt.results_dir, 'GT' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".txt")
        diff_gen_image_dir1 = os.path.join(opt.results_dir, 'Different_rec' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".txt")
        # 保存图像
        # print(fake_im_for_saveimg.shape)
        # print(real_im_for_saveimg.shape)
        # print(diff_gen_x10_color.shape)
        cv2.imwrite(fake_image_dir, fake_im_for_saveimg[2:222,2:222,:])
        cv2.imwrite(real_image_dir, real_im_for_saveimg[2:222,2:222,:])
        cv2.imwrite(diff_gen_image_dir, diff_gen_x10_color[2:222,2:222,:])

        #保存txt
        # img = Image.open('/media/mri/Orange/EXP_MRF/9_ResViT_BraTS_1_2new/EXP_results_T1_288_m2/Recon/50_time_series[50].png')  
        # np.savetxt(real_image_dir1,real_im_for_savetxt,fmt="%f")
        # np.savetxt(fake_image_dir1,fake_im_for_savetxt,fmt="%f")
        # np.savetxt(diff_gen_image_dir1,diff_gen_x10_color,fmt="%f")
        #########################################################################################
        #########################################################################################

    # mae = np.mean(mae_avg)
    # mae_std = np.std(mae_avg)
    # mean_psnr = np.mean(psnr_avg)
    # std_psnr = np.std(psnr_avg)
    mean_ssim = np.mean(ssim_avg)
    std_ssim = np.std(ssim_avg)
    mean_nmse = np.mean(nmse_avg)
    std_nmse = np.std(nmse_avg)
    # mean_another_ssim_avg = np.mean(another_ssim_avg)
    # another_std_ssim = np.std(another_ssim_avg)
    
    print('testing -\n mean_nmse:%.5f, std_nmse:%.5f ,mean_ssim:%.5f, std_ssim:%.5f,' % (mean_nmse,std_nmse,mean_ssim,std_ssim))
