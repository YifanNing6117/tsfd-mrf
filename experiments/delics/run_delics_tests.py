import os
import subprocess
import datetime
import argparse

# ==============================================================================
GPU_ID = '2'  
MODEL_NAME = 'delics_2min_50DWTrealtfmambampu_v5_nokaiming_lr1e-4_lr1e-6' # 替换为你最新的模型名
CHECKPOINTS_DIR = 'EXP_checkpoints' 
# ==============================================================================

# 只测 Deli-CS 的 T1 和 T2
TEST_JOBS = [
    ('delics', 't1'),
    ('delics', 't2'),
]

UNIFIED_TESTER_SCRIPT = 'test_delics.py'

def run_single_test(dataset, modality, config):
    print("-" * 80)
    print(f"▶️  开始测试: [{dataset.upper()}] 模态=[{modality.upper()}]")
    
    dataroot = 'delics_dataset.csv' # 指向你生成的 CSV
    how_many = 100                  # 设大一点，DataLoader会自动在测试集跑完时停止
    results_dir = f'delics_results_{modality}/'
    os.makedirs(results_dir, exist_ok=True)
    
    command = [
        'python', config['tester_script'],
        '--name', config['model_name'],
        '--checkpoints_dir', config['checkpoints_dir'],
        '--gpu_ids', config['gpu_id'],
        '--dataroot', dataroot,
        '--how_many', str(how_many),
        '--results_dir', results_dir,
        '--modality', modality,
        
        # --- 核心修复：完全适配 MRF 的通道数 ---
        '--model', 'resvit_one',
        '--which_model_netG', 'DWTrealtfmambampu_v5',
        '--dataset_mode', 'delics',   # 必须是 delics
        '--norm', 'none',             # 保持和你训练时一致
        '--input_nc', '400',          # ✅ 修复：输入必须是 400！
        '--output_nc', '2',           # ✅ 修复：模型输出必须是 2！
        
        '--serial_batches', 
        '--fineSize', '256',
        '--loadSize', '256', 
        '--which_epoch', 'latest',
        '--pre_trained_resnet', '0', 
        '--pre_trained_transformer', '0'
    ]
    
    print(f"  执行命令: {' '.join(command)}")
    
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"✅ 测试成功: 模态=[{modality}]")
        metrics_file = os.path.join(results_dir, 'metrics.txt')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                summary = f.read()
            return results_dir, summary
        return results_dir, "测试完成，但未找到 metrics.txt"
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: 模态=[{modality}]")
        print(e.stderr)
        return None, e.stderr

def main(args):
    start_time = datetime.datetime.now()
    print("=" * 80)
    print(f"�� 开始 Deli-CS 自动测试，启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    config = {
        'gpu_id': args.gpu,
        'model_name': MODEL_NAME,
        'checkpoints_dir': CHECKPOINTS_DIR,
        'tester_script': UNIFIED_TESTER_SCRIPT,
    }

    all_results = []
    for job_dataset, job_modality in TEST_JOBS:
        results_dir, summary_text = run_single_test(job_dataset, job_modality, config)
        all_results.append({
            'modality': job_modality,
            'results_dir': results_dir,
            'summary': summary_text
        })
    
    end_time = datetime.datetime.now()
    summary_filename = f"delics_summary_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(summary_filename, 'w', encoding='utf-8') as f:
        report_header = f"Deli-CS 批量测试报告\n模型: {MODEL_NAME}\n总耗时: {end_time - start_time}\n{'='*40}\n\n"
        print("\n" + report_header)
        f.write(report_header)
        
        for result in all_results:
            section_header = f"--- 测试结果: {result['modality'].upper()} ---\n"
            print(section_header, end='')
            f.write(section_header)
            print(result['summary'])
            f.write(result['summary'] + "\n\n")

    print(f"✅ 最终报告已保存在: {summary_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=GPU_ID)
    args = parser.parse_args()
    main(args)

