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
    return np.sum((x1 - x2) ** 2) / (np.sum(x1 ** 2) + 1e-8)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.phase = 'test'

    modality = opt.modality
    w_channel = 0 if modality == 't1' else 1
    diff_multiplier = 10 

    print("="*60)
    print(f"�� Deli-CS 全局物理级精准测试已启动 (PID: {os.getpid()})")
    print(f"评估模态: {modality.upper()} (通道: {w_channel})")
    print("="*60)

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

        fake_im_tensor = model.fake_B.squeeze(0).cpu().data.numpy()
        fake_im = fake_im_tensor[w_channel, ...]
        fake_im_norm = np.clip(fake_im, 0, 1)
        
        real_im_tensor = model.real_B.squeeze(0).cpu().data.numpy()
        real_im = real_im_tensor[w_channel, ...]
        real_im_norm = np.clip(real_im, 0, 1)

        # �� 极简物理反归一化
        physical_max = 5000.0 if modality == 't1' else 4000.0
        fake_physical = fake_im_norm * physical_max
        real_physical = real_im_norm * physical_max

        # �� 提取掩码，杜绝背景误差稀释
        mask_tensor = model.mask.squeeze().cpu().data.numpy()
        mask = mask_tensor > 0

        real_valid_ms = real_physical[mask]
        fake_valid_ms = fake_physical[mask]

        if len(real_valid_ms) > 0:
            mae_scores[i] = np.mean(np.abs(real_valid_ms - fake_valid_ms))
            nmse_scores[i] = NMSE(real_valid_ms, fake_valid_ms)
        else:
            mae_scores[i], nmse_scores[i] = 0.0, 0.0

        # PSNR/SSIM 仍用 0-1 空间计算
        real_im_masked = real_im_norm * mask
        fake_im_masked = fake_im_norm * mask
        psnr_scores[i] = psnr(real_im_masked, fake_im_masked, data_range=1.0)
        ssim_scores[i] = ssim(real_im_masked, fake_im_masked, data_range=1.0)

        img_path = model.get_image_paths()[0]
        slice_idx = model.get_slice_idx()
        case_id = img_path.split('/')[-1].split('_')[0] 
        slice_str = str(slice_idx.item()) if torch.is_tensor(slice_idx) else str(slice_idx)
        
        print(f'{i+1:04d}/{num_images}: {case_id}-Z{slice_str} | NMSE: {nmse_scores[i]:.4f} | MAE: {mae_scores[i]:.2f}ms')

        os.makedirs(os.path.join(opt.results_dir, 'GT'), exist_ok=True)
        os.makedirs(os.path.join(opt.results_dir, 'Recon'), exist_ok=True)
        os.makedirs(os.path.join(opt.results_dir, 'Different_rec'), exist_ok=True)
        
        fake_vis = (fake_im_masked * 255.0).round().astype(np.uint8)
        real_vis = (real_im_masked * 255.0).round().astype(np.uint8)

        cv2.imwrite(os.path.join(opt.results_dir, 'Recon', f"{case_id}_slice{slice_str}.png"), cv2.applyColorMap(fake_vis, cv2.COLORMAP_HOT))
        cv2.imwrite(os.path.join(opt.results_dir, 'GT', f"{case_id}_slice{slice_str}.png"), cv2.applyColorMap(real_vis, cv2.COLORMAP_HOT))
        
        diff_img = np.clip(np.abs(fake_im_masked - real_im_masked) * diff_multiplier, 0, 1)
        cv2.imwrite(os.path.join(opt.results_dir, 'Different_rec', f"{case_id}_slice{slice_str}.png"), cv2.applyColorMap((diff_img * 255.0).astype(np.uint8), cv2.COLORMAP_HOT))

    total_time = np.sum(inference_times)
    mean_psnr, std_psnr = np.mean(psnr_scores), np.std(psnr_scores)
    mean_ssim, std_ssim = np.mean(ssim_scores), np.std(ssim_scores)
    mean_nmse, std_nmse = np.mean(nmse_scores), np.std(nmse_scores)
    mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
    
    print("\n" + "="*60)
    print(f"✅ DELI-CS 物理级测试完成: {modality.upper()}")
    print(f"  PSNR: {mean_psnr:.4f} +/- {std_psnr:.4f}")
    print(f"  SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}")
    print(f"  NMSE: {mean_nmse:.6f} +/- {std_nmse:.6f}")
    print(f"  MAE:  {mean_mae:.2f} +/- {std_mae:.2f} (单位: ms)")
    print("="*60)

    metrics_path = os.path.join(opt.results_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Summary for: {opt.name}\nDataset: Deli-CS, Modality: {modality.upper()}\n")
        f.write("="*30 + "\n")
        f.write(f"PSNR: {mean_psnr:.4f} +/- {std_psnr:.4f}\n")
        f.write(f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}\n")
        f.write(f"NMSE: {mean_nmse:.6f} +/- {std_nmse:.6f}\n")
        f.write(f"MAE (ms):  {mean_mae:.2f} +/- {std_mae:.2f}\n")

