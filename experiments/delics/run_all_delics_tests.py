import os
import sys
import time
import datetime
import numpy as np
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model

# =============================================================================
#                        顶部配置区：直接改这里
# =============================================================================
GPU_ID = '2'
MODEL_NAME = 'delics_2min_50DWTrealtfmambampu_v5_nokaiming_lr1e-5_lr1e-6'
CHECKPOINTS_DIR = 'EXP_checkpoints'
DATA_ROOT = 'delics_2min_dataset.csv'

TEST_MODALITIES = ['t1', 't2']

# 统一总输出目录
OUTPUT_ROOT = 'delics_results'

DIFF_MULTIPLIER = 1.7
PHYSICAL_MAX_T1 = 5000.0
PHYSICAL_MAX_T2 = 4000.0

HOW_MANY = 100
MODEL_TYPE = 'resvit_one'
WHICH_MODEL_NETG = 'DWTtfmambampu_v5'
DATASET_MODE = 'delics'
NORM = 'none'
INPUT_NC = 400
OUTPUT_NC = 2
FINE_SIZE = 256
LOAD_SIZE = 256
WHICH_EPOCH = 'latest'
PRE_TRAINED_RESNET = '0'
PRE_TRAINED_TRANSFORMER = '0'
# =============================================================================


def NMSE(x1, x2):
    return np.sum((x1 - x2) ** 2) / (np.sum(x1 ** 2) + 1e-8)


def build_testoptions_argv(modality, results_dir):
    """
    构造给 TestOptions 使用的伪命令行参数
    """
    argv = [
        'run_all_delics_tests.py',
        '--dataroot', DATA_ROOT,
        '--name', MODEL_NAME,
        '--gpu_ids', GPU_ID,
        '--checkpoints_dir', CHECKPOINTS_DIR,
        '--dataset_mode', DATASET_MODE,
        '--serial_batches',
        '--nThreads', '1',
        '--batchSize', '1',
        '--loadSize', str(LOAD_SIZE),
        '--fineSize', str(FINE_SIZE),
        '--max_dataset_size', '100000',
        '--resize_or_crop', 'none',
        '--no_flip',
        '--model', MODEL_TYPE,
        '--input_nc', str(INPUT_NC),
        '--output_nc', str(OUTPUT_NC),
        '--which_model_netG', WHICH_MODEL_NETG,
        '--norm', NORM,
        '--no_dropout',
        '--phase', 'test',
        '--which_epoch', WHICH_EPOCH,
        '--how_many', str(HOW_MANY),
        '--results_dir', results_dir,
        '--modality', modality,
        '--mask', '1',
        '--pre_trained_resnet', PRE_TRAINED_RESNET,
        '--pre_trained_transformer', PRE_TRAINED_TRANSFORMER,
    ]
    return argv


def build_opt(modality, results_dir):
    old_argv = sys.argv
    try:
        sys.argv = build_testoptions_argv(modality, results_dir)
        opt = TestOptions().parse()
    finally:
        sys.argv = old_argv
    return opt


def run_single_test(modality):
    # 统一总目录 / 模型名 / t1(or t2)
    results_dir = os.path.join(OUTPUT_ROOT, MODEL_NAME, modality)
    os.makedirs(results_dir, exist_ok=True)

    opt = build_opt(modality, results_dir)

    w_channel = 0 if modality == 't1' else 1
    physical_max = PHYSICAL_MAX_T1 if modality == 't1' else PHYSICAL_MAX_T2

    print("=" * 80)
    print(f"▶️  开始测试: [{modality.upper()}]")
    print(f"模型名: {MODEL_NAME}")
    print(f"输出目录: {results_dir}")
    print(f"差异图倍率: {DIFF_MULTIPLIER}")
    print("=" * 80)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    num_images = min(len(dataset), opt.how_many)
    psnr_scores = np.zeros(num_images)
    ssim_scores = np.zeros(num_images)
    nmse_scores = np.zeros(num_images)
    mae_scores = np.zeros(num_images)
    inference_times = np.zeros(num_images)

    gt_dir = os.path.join(results_dir, 'GT')
    recon_dir = os.path.join(results_dir, 'Recon')
    diff_dir = os.path.join(results_dir, 'Different_rec')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)

    for i, data in enumerate(dataset):
        if i >= num_images:
            break

        model.set_input(data)

        start_time = time.time()
        model.test()
        inference_times[i] = time.time() - start_time

        fake_im_tensor = model.fake_B.squeeze(0).cpu().data.numpy()
        real_im_tensor = model.real_B.squeeze(0).cpu().data.numpy()

        fake_im = fake_im_tensor[w_channel, ...]
        real_im = real_im_tensor[w_channel, ...]

        fake_im_norm = np.clip(fake_im, 0, 1)
        real_im_norm = np.clip(real_im, 0, 1)

        fake_physical = fake_im_norm * physical_max
        real_physical = real_im_norm * physical_max

        mask_tensor = model.mask.squeeze().cpu().data.numpy()
        mask = mask_tensor > 0

        real_valid_ms = real_physical[mask]
        fake_valid_ms = fake_physical[mask]

        if len(real_valid_ms) > 0:
            mae_scores[i] = np.mean(np.abs(real_valid_ms - fake_valid_ms))
            nmse_scores[i] = NMSE(real_valid_ms, fake_valid_ms)
        else:
            mae_scores[i] = 0.0
            nmse_scores[i] = 0.0

        real_im_masked = real_im_norm * mask
        fake_im_masked = fake_im_norm * mask

        psnr_scores[i] = psnr(real_im_masked, fake_im_masked, data_range=1.0)
        ssim_scores[i] = ssim(real_im_masked, fake_im_masked, data_range=1.0)

        img_path = model.get_image_paths()[0]
        slice_idx = model.get_slice_idx()
        case_id = img_path.split('/')[-1].split('_')[0]
        slice_str = str(slice_idx.item()) if torch.is_tensor(slice_idx) else str(slice_idx)

        print(f'{i+1:04d}/{num_images}: {case_id}-Z{slice_str} | NMSE: {nmse_scores[i]:.4f} | MAE: {mae_scores[i]:.2f}ms')

        fake_vis = (fake_im_masked * 255.0).round().astype(np.uint8)
        real_vis = (real_im_masked * 255.0).round().astype(np.uint8)

        cv2.imwrite(
            os.path.join(recon_dir, f"{case_id}_slice{slice_str}.png"),
            cv2.applyColorMap(fake_vis, cv2.COLORMAP_HOT)
        )
        cv2.imwrite(
            os.path.join(gt_dir, f"{case_id}_slice{slice_str}.png"),
            cv2.applyColorMap(real_vis, cv2.COLORMAP_HOT)
        )

        diff_img = np.clip(np.abs(fake_im_masked - real_im_masked) * DIFF_MULTIPLIER, 0, 1)
        cv2.imwrite(
            os.path.join(diff_dir, f"{case_id}_slice{slice_str}.png"),
            cv2.applyColorMap((diff_img * 255.0).astype(np.uint8), cv2.COLORMAP_HOT)
        )

    total_time = np.sum(inference_times)
    mean_psnr, std_psnr = np.mean(psnr_scores), np.std(psnr_scores)
    mean_ssim, std_ssim = np.mean(ssim_scores), np.std(ssim_scores)
    mean_nmse, std_nmse = np.mean(nmse_scores), np.std(nmse_scores)
    mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)

    print("\n" + "=" * 80)
    print(f"✅ 测试完成: {modality.upper()}")
    print(f"PSNR: {mean_psnr:.4f} +/- {std_psnr:.4f}")
    print(f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}")
    print(f"NMSE: {mean_nmse:.6f} +/- {std_nmse:.6f}")
    print(f"MAE : {mean_mae:.2f} +/- {std_mae:.2f} (ms)")
    print("=" * 80)

    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Summary for: {MODEL_NAME}\n")
        f.write(f"Dataset: Deli-CS, Modality: {modality.upper()}\n")
        f.write(f"Results Dir: {results_dir}\n")
        f.write("=" * 40 + "\n")
        f.write(f"PSNR: {mean_psnr:.4f} +/- {std_psnr:.4f}\n")
        f.write(f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}\n")
        f.write(f"NMSE: {mean_nmse:.6f} +/- {std_nmse:.6f}\n")
        f.write(f"MAE (ms): {mean_mae:.2f} +/- {std_mae:.2f}\n")
        f.write(f"Total Inference Time: {total_time:.4f} s\n")
        f.write(f"Diff Multiplier: {DIFF_MULTIPLIER}\n")

    summary_text = (
        f"--- {modality.upper()} ---\n"
        f"Results Dir: {results_dir}\n"
        f"PSNR: {mean_psnr:.4f} +/- {std_psnr:.4f}\n"
        f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}\n"
        f"NMSE: {mean_nmse:.6f} +/- {std_nmse:.6f}\n"
        f"MAE (ms): {mean_mae:.2f} +/- {std_mae:.2f}\n"
        f"Total Inference Time: {total_time:.4f} s\n"
    )

    return summary_text


def main():
    start_time = datetime.datetime.now()
    print("=" * 80)
    print(f"开始 Deli-CS 自动测试，启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    all_summaries = []
    for modality in TEST_MODALITIES:
        summary = run_single_test(modality)
        all_summaries.append(summary)

    end_time = datetime.datetime.now()
    report_name = f"delics_summary_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_name, 'w', encoding='utf-8') as f:
        header = (
            f"Deli-CS 批量测试报告\n"
            f"模型: {MODEL_NAME}\n"
            f"GPU: {GPU_ID}\n"
            f"总耗时: {end_time - start_time}\n"
            f"Diff Multiplier: {DIFF_MULTIPLIER}\n"
            f"{'=' * 60}\n\n"
        )
        print("\n" + header)
        f.write(header)

        for summary in all_summaries:
            print(summary)
            f.write(summary + "\n")

    print(f"✅ 最终报告已保存在: {report_name}")


if __name__ == '__main__':
    main()
