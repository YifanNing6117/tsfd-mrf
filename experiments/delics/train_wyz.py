# train_wyz.py with a toggle for Automatic Mixed Precision (AMP)

import os
import time
import numpy as np
import torch
# 导入混合精度训练所需的库 (即使关闭也先导入，避免ImportError)
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from skimage.metrics import peak_signal_noise_ratio as psnr
from options.train_options_wyz import TrainOptions
from data import CreateDataLoader
from models import create_model
from util1.visualizer import Visualizer

def NMSE(x1_batch, x2_batch):
    x1_batch = np.asarray(x1_batch)
    x2_batch = np.asarray(x2_batch)
    if x1_batch.ndim == 4 and x1_batch.shape[1] == 1:
        x1_batch = np.squeeze(x1_batch, axis=1)
    if x2_batch.ndim == 4 and x2_batch.shape[1] == 1:
        x2_batch = np.squeeze(x2_batch, axis=1)
    diff_sq = (x1_batch - x2_batch)**2
    x1_sq = x1_batch**2
    sum_diff_sq_per_sample = np.sum(diff_sq, axis=(-1, -2))
    sum_x1_sq_per_sample = np.sum(x1_sq, axis=(-1, -2))
    nmse_per_sample = sum_diff_sq_per_sample / (sum_x1_sq_per_sample + 1e-9)
    return np.mean(nmse_per_sample)

def print_log(logger, message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # ==============================================================================
    # 提醒: 请确保在您的 TrainOptions 类 (通常在 options/train_options_wyz.py 文件中)
    # 的 initialize 方法里添加以下这行代码:
    #
    # self.parser.add_argument('--use_amp', action='store_true', help='Enable Automatic Mixed Precision (AMP) training')
    # ==============================================================================

    # --- 1. 数据加载 ---
    opt.phase = 'train'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f'================================ Training images = {dataset_size} ================================')

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger_path = os.path.join(save_dir, 'log.txt')
    with open(logger_path, 'w+') as logger_file:
        print_log(logger_file, f"Experiment Name: {opt.name}")
        # 记录AMP状态
        print_log(logger_file, f"Automatic Mixed Precision (AMP): {'Enabled' if opt.use_amp else 'Disabled'}")


    opt.phase = 'valid'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print(f'================================ Validation images = {dataset_size_val} ================================')

    # --- 2. 指标记录数组 ---
    num_epochs_total = opt.niter + opt.niter_decay
    psnr_avg1 = np.zeros([num_epochs_total, dataset_size_val * opt.batchSize])
    nmse_avg1 = np.zeros([num_epochs_total, dataset_size_val * opt.batchSize])
    psnr_avg2 = np.zeros([num_epochs_total, dataset_size_val * opt.batchSize])
    nmse_avg2 = np.zeros([num_epochs_total, dataset_size_val * opt.batchSize])

    # --- 3. 实例化模型 ---
    model = create_model(opt)
    optimizer_G = model.optimizer_G 
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs_total, eta_min=1e-6)
    print("CosineAnnealingLR scheduler was created.")

    # --- 4. 可视化器 ---
    visualizer = Visualizer(opt)

    # --- 5. 训练循环 ---
    total_steps = 0
    
    # 根据开关初始化 GradScaler
    scaler = GradScaler() if opt.use_amp else None
    if scaler:
        print("Automatic Mixed Precision (AMP) training enabled.")
    else:
        print("Standard FP32 training (AMP disabled).")
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        model.train()
        opt.phase = 'train'
        epoch_train_losses = []

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data)
            optimizer_G.zero_grad()
            
            # 根据开关选择训练流程
            if opt.use_amp:
                # --- AMP 开启时的流程 ---
                with autocast():
                    model.forward()
                    loss_G = model.criterionPixel(model.fake_B, model.real_B)
                    model.loss_G = loss_G

                scaler.scale(model.loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
            else:
                # --- AMP 关闭时的标准流程 ---
                model.forward()
                loss_G = model.criterionPixel(model.fake_B, model.real_B)
                model.loss_G = loss_G
                
                model.loss_G.backward()
                optimizer_G.step()
            
            # --- 记录和打印训练损失 ---
            if total_steps % opt.print_freq == 0:
                current_loss = model.loss_G.item()
                epoch_train_losses.append(current_loss) 
    
                t_comp = (time.time() - iter_start_time) / opt.batchSize 
                loss_message = (f"Epoch: {epoch}/{num_epochs_total}, "
                    f"Iter: {epoch_iter}/{dataset_size*opt.batchSize}, "
                    f"Time/Sample: {t_comp:.3f}s, "
                    f"Pixel_Loss: {current_loss:.6f}")
                print(loss_message)

            # --- 训练时指标计算 ---
            if total_steps % (opt.print_freq * 10) == 0:
                with torch.no_grad():
                    model.eval()
                    with autocast(enabled=opt.use_amp):
                        pass
                    
                    fake_im_train = model.fake_B.cpu().numpy()
                    real_im_train = model.real_B.cpu().numpy()
                    
                    faket1_train = fake_im_train[:,0,...]
                    faket2_train = fake_im_train[:,1,...]
                    realt1_train = real_im_train[:,0,...]
                    realt2_train = real_im_train[:,1,...]
                    
                    psnr_batch1_train = []
                    psnr_batch2_train = []
                    for j_train in range(real_im_train.shape[0]):
                        psnr_batch1_train.append(psnr(realt1_train[j_train, ...], faket1_train[j_train, ...], data_range=1))
                        psnr_batch2_train.append(psnr(realt2_train[j_train, ...], faket2_train[j_train, ...], data_range=1))
                    
                    nmse_batch1_train = NMSE(realt1_train, faket1_train)
                    nmse_batch2_train = NMSE(realt2_train, faket2_train)
                    
                    print(f'[Epoch{epoch}] Training Batch Metrics - PSNR_T1: {np.mean(psnr_batch1_train):.3f}, NMSE_T1: {nmse_batch1_train:.3f}, PSNR_T2: {np.mean(psnr_batch2_train):.3f}, NMSE_T2: {nmse_batch2_train:.3f}')
                    model.train()

            # --- 保存最新模型 ---
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
            
        # 打印当前epoch的平均训练损失
        if epoch_train_losses:
            avg_epoch_train_loss = np.mean(epoch_train_losses)
            print(f"Epoch {epoch} average training pixel loss: {avg_epoch_train_loss:.4f}")
            with open(logger_path, 'a') as logger_file:
                print_log(logger_file, f"Epoch {epoch} average training pixel loss: {avg_epoch_train_loss:.4f}")


        # === Validation Step ===
        if epoch % opt.save_epoch_freq == 0:
            model.eval()
            opt.phase = 'valid'

            current_epoch_psnr1_list = []
            current_epoch_nmse1_list = []
            current_epoch_psnr2_list = []
            current_epoch_nmse2_list = []

            with torch.no_grad():
                with autocast(enabled=opt.use_amp):
                    for val_i, data_val in enumerate(dataset_val):
                        model.set_input(data_val)
                        model.test()
                        
                        fake_im_val = model.fake_B.cpu().numpy()
                        real_im_val = model.real_B.cpu().numpy()
                        
                        faket1_val = fake_im_val[:,0,...]
                        realt1_val = real_im_val[:,0,...]
                        faket2_val = fake_im_val[:,1,...]
                        realt2_val = real_im_val[:,1,...]

                        for j_val in range(real_im_val.shape[0]):
                            real_im_1_val = realt1_val[j_val, ...]
                            fake_im_1_val = faket1_val[j_val, ...]
                            real_im_2_val = realt2_val[j_val, ...]
                            fake_im_2_val = faket2_val[j_val, ...]
                            
                            current_epoch_psnr1_list.append(psnr(real_im_1_val, fake_im_1_val, data_range=1))
                            current_epoch_psnr2_list.append(psnr(real_im_2_val, fake_im_2_val, data_range=1))
                        
                        current_epoch_nmse1_list.append(NMSE(realt1_val, faket1_val))
                        current_epoch_nmse2_list.append(NMSE(realt2_val, faket2_val))
            
            mean_psnr1_val = np.mean(current_epoch_psnr1_list)
            mean_nmse1_val = np.mean(current_epoch_nmse1_list)
            mean_psnr2_val = np.mean(current_epoch_psnr2_list)
            mean_nmse2_val = np.mean(current_epoch_nmse2_list)

            with open(logger_path, 'a') as logger_file:
                val_log_msg = f'[Epoch {epoch}] Validation - PSNR_T1: {mean_psnr1_val:.3f}, NMSE_T1: {mean_nmse1_val:.3f}, PSNR_T2: {mean_psnr2_val:.3f}, NMSE_T2: {mean_nmse2_val:.3f}'
                print_log(logger_file, val_log_msg)
                print_log(logger_file, '')

            print('########## saving the model at the end of epoch %d, iters %d #############' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, num_epochs_total, time.time() - epoch_start_time))
        
        scheduler_G.step()
        
        current_lr = optimizer_G.param_groups[0]['lr']
        lr_log_msg = f"Learning rate for next epoch ({epoch+1}) set to: {current_lr:.7f}"
        print(lr_log_msg)
        with open(logger_path, 'a') as logger_file:
             print_log(logger_file, lr_log_msg)
