import os
import subprocess
import datetime
import argparse

# ==============================================================================
# --- 模型和路径配置 ---
GPU_ID = '0'  
# ⚠️ 替换为您正在训练的 200 序列的模型文件夹名称
MODEL_NAME = '200_50DWTrealtfmambampu_v5_nokaiming_lr8e-4_lr1e-5' 
CHECKPOINTS_DIR = 'EXP_checkpoints' 

TEST_JOBS = [
    ('subjects', 't1'),
    ('subjects', 't2'),
    ('brainweb', 't1'),
    ('brainweb', 't2'),
]

UNIFIED_TESTER_SCRIPT = 'test_unified_env.py'
# ==============================================================================

def run_single_test(dataset, modality, config):
    print("-" * 80)
    print(f"▶️  开始测试: 数据集=[{dataset}], 模态=[{modality}]")
    
    if dataset == 'brainweb':
        dataroot = 'testBrainweb.csv'
        how_many = 20
        results_dir = f'brainweb_{modality}/'
    elif dataset == 'subjects':
        dataroot = 'testsubjects.csv'
        how_many = 34
        results_dir = f'subjects_{modality}/'
    else:
        return None, f"未知数据集: {dataset}"

    os.makedirs(results_dir, exist_ok=True)
    
    # 使用环境变量传递自定义参数，防止老框架的 parser 报错
    run_env = os.environ.copy()
    run_env['TEST_MODALITY'] = modality
    run_env['TEST_RESULTS_DIR'] = results_dir

    command = [
        'python', config['tester_script'],
        '--name', config['model_name'],
        '--checkpoints_dir', config['checkpoints_dir'],
        '--gpu_ids', config['gpu_id'],
        '--dataroot', dataroot,
        '--how_many', str(how_many),
        
        # --- 对齐您的新 SOTA 训练命令 ---
        '--model', 'resvit_one',
        '--which_model_netG', 'DWTtfmambampu_v5',
        '--dataset_mode', 'aligned2', 
        '--norm', 'none',        # ⚠️ 对齐新命令: norm none
        '--output_nc', '1',
        
        # 🌟 核心魔法：传入 200，让 Dataset 截取 200 帧并补零
        '--input_nc', '200',     
        
        '--serial_batches', 
        '--fineSize', '256',     # 新网络直接用 256
        '--loadSize', '256', 
        '--which_epoch', 'latest',
        '--pre_trained_resnet', '0', 
        '--pre_trained_transformer', '0',
        '--display_id', '0'
    ]
    
    print(f"  执行命令: {' '.join(command)}")
    
    try:
        process = subprocess.run(command, env=run_env, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"✅ 测试成功: 数据集=[{dataset}], 模态=[{modality}]")
        metrics_file = os.path.join(results_dir, 'metrics.txt')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                summary = f.read()
            return results_dir, summary
        else:
            return results_dir, "测试完成，但未找到 metrics.txt 文件。"
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: 数据集=[{dataset}], 模态=[{modality}]")
        print("错误日志:\n", e.stderr)
        return None, e.stderr

def main(args):
    start_time = datetime.datetime.now()
    print("=" * 80)
    print(f"🚀 开始全自动批量测试，启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU ID: {args.gpu}, 模型: {MODEL_NAME}")
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
        if results_dir:
            all_results.append({
                'dataset': job_dataset,
                'modality': job_modality,
                'results_dir': results_dir,
                'summary': summary_text
            })
    
    end_time = datetime.datetime.now()
    summary_filename = f"final_summary_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
    print("\n" + "=" * 80)
    print("🎉 所有测试任务已完成，正在生成最终报告...")
    print("=" * 80)
    
    with open(summary_filename, 'w', encoding='utf-8') as f:
        report_header = (
            f"批量测试最终报告\n模型名称: {MODEL_NAME}\n"
            f"报告生成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"总耗时: {end_time - start_time}\n{'='*40}\n\n"
        )
        print(report_header)
        f.write(report_header)
        
        for result in all_results:
            section_header = f"--- 测试结果: {result['dataset'].upper()} - {result['modality'].upper()} ---\n"
            result_location = f"结果目录: {result['results_dir']}\n"
            print(section_header, end=''); f.write(section_header)
            print(result_location, end=''); f.write(result_location)
            print(result['summary']); f.write(result['summary'] + "\n\n")

    print(f"\n✅ 最终报告已成功保存至: {summary_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=GPU_ID)
    args = parser.parse_args()
    main(args)

