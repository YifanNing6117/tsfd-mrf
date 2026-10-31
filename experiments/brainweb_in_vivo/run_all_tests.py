import os
import subprocess
import datetime
import argparse

# ==============================================================================
# ---------------------------- 1. 中央配置区 ---------------------------------
# 您未来所有不常变的参数，都在这里修改一次即可，一劳永逸！
# ==============================================================================

# --- 模型和路径配置 ---
GPU_ID = '0'  # 您想使用的GPU编号
MODEL_NAME = '50DWTrealtfmambampu_v5_nokaiming_lr8e-4_lr1e-5' # 要测试的模型名称
CHECKPOINTS_DIR = 'EXP_checkpoints' # checkpoints的根目录

# --- 待执行的测试任务列表 ---
# 格式: (数据集名称, 模态)
# 脚本会按顺序自动执行列表中的所有任务
TEST_JOBS = [
    ('subjects', 't1'),
    ('subjects', 't2'),
    ('brainweb', 't1'),
    ('brainweb', 't2'),
]

# --- 统一的测试脚本文件名 ---
# 这个是后台干活的脚本，一般不需要改
UNIFIED_TESTER_SCRIPT = 'test_unified.py'

# ==============================================================================
# --------------------------- 脚本主逻辑 (无需修改) ----------------------------
# ==============================================================================

def run_single_test(dataset, modality, config):
    """构建并执行单个测试任务"""
    print("-" * 80)
    print(f"▶️  开始测试: 数据集=[{dataset}], 模态=[{modality}]")
    
    # 根据任务确定可变参数
    if dataset == 'brainweb':
        dataroot = 'testBrainweb.csv'
        how_many = 20
        results_dir = f'brainweb{1 if modality == "t1" else 2}/'
    elif dataset == 'subjects':
        dataroot = 'testsubjects.csv'
        how_many = 34
        results_dir = f'subjects{1 if modality == "t1" else 2}/'
    else:
        print(f"❌ 错误: 未知的测试任务数据集 '{dataset}'")
        return None, f"未知数据集: {dataset}"

    os.makedirs(results_dir, exist_ok=True)
    
    # 构建完整命令
    command = [
        'python', config['tester_script'],
        '--name', config['model_name'],
        '--checkpoints_dir', config['checkpoints_dir'],
        '--gpu_ids', config['gpu_id'],
        '--dataroot', dataroot,
        '--how_many', str(how_many),
        '--results_dir', results_dir,
        '--modality', modality,
        # --- 以下是基本不变的参数 ---
        '--model', 'resvit_one',
        '--which_model_netG', 'DWTtfmambampu_v5',
        '--dataset_mode', 'aligned2', '--norm', 'batch', '--output_nc', '1',
        '--input_nc', '1', '--serial_batches', '--fineSize', '256',
        '--loadSize', '256', '--which_epoch', 'latest',
        '--pre_trained_resnet', '0', '--pre_trained_transformer', '0'
    ]
    
    print(f"  执行命令: {' '.join(command)}")
    
    # 执行子进程并等待其完成
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"✅ 测试成功: 数据集=[{dataset}], 模态=[{modality}]")
        # 查找 metrics.txt 文件中的关键指标
        metrics_file = os.path.join(results_dir, 'metrics.txt')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                summary = f.read()
            return results_dir, summary
        else:
            return results_dir, "测试完成，但未找到 metrics.txt 文件。"
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: 数据集=[{dataset}], 模态=[{modality}]")
        print("错误日志:")
        print(e.stderr)
        return None, e.stderr
    except FileNotFoundError:
        error_msg = f"❌ 严重错误: 找不到执行脚本 '{config['tester_script']}'"
        print(error_msg)
        return None, error_msg


def main(args):
    """主函数，执行所有测试任务并生成总结报告"""
    start_time = datetime.datetime.now()
    print("=" * 80)
    print(f"�� 开始全自动批量测试，启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        all_results.append({
            'dataset': job_dataset,
            'modality': job_modality,
            'results_dir': results_dir,
            'summary': summary_text
        })
    
    end_time = datetime.datetime.now()
    
    # --- 生成最终的统一报告 ---
    summary_filename = f"final_summary_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
    print("\n" + "=" * 80)
    print("�� 所有测试任务已完成，正在生成最终报告...")
    print("=" * 80)
    
    with open(summary_filename, 'w', encoding='utf-8') as f:
        report_header = (
            f"批量测试最终报告\n"
            f"模型名称: {MODEL_NAME}\n"
            f"报告生成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"总耗时: {end_time - start_time}\n"
            f"{'='*40}\n\n"
        )
        print(report_header)
        f.write(report_header)
        
        for result in all_results:
            section_header = f"--- 测试结果: {result['dataset'].upper()} - {result['modality'].upper()} ---\n"
            result_location = f"结果目录: {result['results_dir']}\n" if result['results_dir'] else "结果目录: N/A\n"
            
            print(section_header, end='')
            f.write(section_header)
            
            print(result_location, end='')
            f.write(result_location)
            
            print(result['summary'])
            f.write(result['summary'] + "\n\n")

    print(f"\n✅ 最终报告已成功保存至: {summary_filename}")
    print("�� 全部流程结束。")


if __name__ == "__main__":
    # 允许从命令行覆盖GPU_ID，增加灵活性
    parser = argparse.ArgumentParser(description='一键全自动测试脚本')
    parser.add_argument('--gpu', type=str, default=GPU_ID, 
                        help=f'指定使用的GPU ID。默认为在脚本中配置的: {GPU_ID}')
    args = parser.parse_args()
    main(args)
