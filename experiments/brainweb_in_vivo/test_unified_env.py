# 文件名: test_unified_env.py
import os
import time
import numpy as np
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model

def NMSE(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.sum((x1 - x2) ** 2) / (np.sum(x1 ** 2) + 1e-8)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.phase = 'test'

    # 🌟 核心修改：从环境变量读取，防止 TestOptions 报错
    modality = os.environ.get('TEST_MODALITY', 't1')
    results_dir = os.environ.get('TEST_RESULTS_DIR', './results')
    
    dataset_type = 'brainweb' if 'brainweb' in opt.dataroot.lower() else 'subjects'
    w_channel = 0 if modality == 't1' else 1
    diff_multiplier = 10 if dataset_type == 'brainweb' else 5

    print("="*50)
    print(f"SOTA 网络 200 序列 统一测试脚本已启动 (PID: {os.getpid()})")
    print(f"数据集: {dataset_type}")
    print(f"模态: {modality} (使用数据通道: {w_channel})")
    print("="*50)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    
    num_images = min(len(dataset), opt.how_many)
    psnr_scores = np.zeros(num_images)
    ssim_scores = np.zeros(num_images)
    nmse_scores = np.zeros(num_images)
    mae_scores = np.zeros(num_images)
    inference_times = np.zeros(num_images)

    for i, data in enumerate(dataset):
        if i >= num_images:
            break

        model.set_input(data)
        
        start_time = time.time()
        model.test()
        end_time = time.time()
        
        inference_times[i] = end_time - start_time

        real_im_tensor = model.real_B.squeeze(0).cpu().data.numpy()
        fake_im_tensor = model.fake_B.squeeze(0).cpu().data.numpy()

        real_im = real_im_tensor[w_channel, ...]
        fake_im = fake_im_tensor[w_channel, ...]

        real_im_norm = np.clip(real_im, 0, 1)
        fake_im_norm = np.clip(fake_im, 0, 1)

        if hasattr(model, 'mask') and model.mask is not None:
            mask_tensor = model.mask.squeeze().cpu().data.numpy()
            mask = mask_tensor > 0
        else:
            mask = real_im_norm > 0

        real_im_masked = real_im_norm * mask
        fake_im_masked = fake_im_norm * mask

        real_valid = real_im_masked[mask]
        fake_valid = fake_im_masked[mask]

        if len(real_valid) > 0:
            mae_scores[i] = np.mean(np.abs(real_valid - fake_valid))
            nmse_scores[i] = NMSE(real_valid, fake_valid)
        else:
            mae_scores[i] = 0.0
            nmse_scores[i] = 0.0

        psnr_scores[i] = psnr(real_im_masked, fake_im_masked, data_range=1.0)
        ssim_scores[i] = ssim(real_im_masked, fake_im_masked, data_range=1.0)

        img_path = model.get_image_paths()
        slice_idx = model.get_slice_idx()
        volumn_name = img_path[0].split('/')[-1].split('.')[0]
        
        slice_str = str(slice_idx.item()) if torch.is_tensor(slice_idx) else str(slice_idx)
        print(f'{i+1:04d}/{num_images}: 处理中 {volumn_name}-{slice_str} | 推理时间: {inference_times[i]:.4f}s | PSNR: {psnr_scores[i]:.2f}')

        os.makedirs(os.path.join(results_dir, 'GT'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'Recon'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'Different_rec'), exist_ok=True)
        
        if dataset_type == 'subjects' and modality == 't2':
            fake_vis = fake_im_norm * 1980.0
            real_vis = real_im_norm * 1980.0
            np.clip(fake_vis, None, 400, out=fake_vis)
            np.clip(real_vis, None, 400, out=real_vis)
            fake_vis = (fake_vis / 400.0 * 255.0).round().astype(np.uint8)
            real_vis = (real_vis / 400.0 * 255.0).round().astype(np.uint8)
        else:
            norm_factors = {
                ('brainweb', 't1'): 252.0, ('brainweb', 't2'): 209.7375,
                ('subjects', 't1'): 255.0
            }
            norm_factor = norm_factors.get((dataset_type, modality), 255.0)
            fake_vis = (fake_im_norm * norm_factor).round().astype(np.uint8)
            real_vis = (real_im_norm * norm_factor).round().astype(np.uint8)

        fake_vis = fake_vis * mask
        real_vis = real_vis * mask

        recon_img_color = cv2.applyColorMap(fake_vis, cv2.COLORMAP_HOT)
        gt_img_color = cv2.applyColorMap(real_vis, cv2.COLORMAP_HOT)
        
        diff_tensor = torch.mul(torch.abs(model.fake_B[:, w_channel, :, :] - model.real_B[:, w_channel, :, :]), diff_multiplier)
        diff_img = diff_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        diff_img = diff_img * mask 
        diff_img_uint8 = (diff_img * 255.0).astype(np.uint8)
        diff_img_color = cv2.applyColorMap(diff_img_uint8, cv2.COLORMAP_HOT)
        
        base_filename = f"{volumn_name}_slice{slice_str}.png"
        cv2.imwrite(os.path.join(results_dir, 'Recon', base_filename), recon_img_color)
        cv2.imwrite(os.path.join(results_dir, 'GT', base_filename), gt_img_color)
        cv2.imwrite(os.path.join(results_dir, 'Different_rec', base_filename), diff_img_color)

    total_time = np.sum(inference_times)
    mean_psnr, std_psnr = np.mean(psnr_scores), np.std(psnr_scores)
    mean_ssim, std_ssim = np.mean(ssim_scores), np.std(ssim_scores)
    mean_nmse, std_nmse = np.mean(nmse_scores), np.std(nmse_scores)
    mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
    
    print("\n" + "="*50)
    print(f"TESTING COMPLETE for {dataset_type.upper()} - {modality.upper()}")
    print(f"  PSNR: {mean_psnr:.4f} +/- {std_psnr:.4f}")
    print(f"  SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}")
    print(f"  NMSE: {mean_nmse:.6f} +/- {std_nmse:.6f}")
    print(f"  MAE:  {mean_mae:.6f} +/- {std_mae:.6f}")
    print("="*50)

    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Summary for: {opt.name}\n")
        f.write(f"Dataset: {dataset_type}, Modality: {modality}\n")
        f.write("="*30 + "\n\n")
        f.write(f"PSNR: {mean_psnr:.4f} +/- {std_psnr:.4f}\n")
        f.write(f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}\n")
        f.write(f"NMSE: {mean_nmse:.6f} +/- {std_nmse:.6f}\n")
        f.write(f"MAE:  {mean_mae:.6f} +/- {std_mae:.6f}\n")

