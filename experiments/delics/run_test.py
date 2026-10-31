import os
import subprocess
import argparse
import time

def run_test(dataset, modality, gpu_id='0'):
    """
    根据指定的数据集和模态，构建并执行统一的测试命令。

    Args:
        dataset (str): 要测试的数据集，可选值为 'brainweb' 或 'subjects'。
        modality (str): 要测试的模态，可选值为 't1' 或 't2'。
        gpu_id (str): 使用的GPU ID。
    """
    # ------------------- 1. 根据输入参数确定可变配置 -------------------
    script_name = 'test_unified.py'  # 始终使用我们新的统一脚本

    if dataset == 'brainweb':
        dataroot = 'testBrainweb.csv'
        how_many = 20
        # T1结果在brainweb1/, T2在brainweb2/
        results_dir = f'brainweb{1 if modality == "t1" else 2}/'
    elif dataset == 'subjects':
        dataroot = 'testsubjects.csv'
        how_many = 34
        results_dir = f'subjects{1 if modality == "t1" else 2}/'
    else:
        print(f"错误: 未知的数据集 '{dataset}'。请选择 'brainweb' 或 'subjects'。")
        return

    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # ------------------- 2. 定义基本不变的命令参数 -------------------
    base_command = [
        'python',
        script_name,
        '--name', 'chongjian_triple_dim180_c_CNNEMA_Swin_unguided',
        '--model', 'resvit_one',
        '--which_model_netG', 'MambaSwinCNNEMAUnguided',
        '--dataset_mode', 'aligned2',
        '--norm', 'batch',
        '--output_nc', '1',
        '--input_nc', '1',
        '--serial_batches',
        '--fineSize', '256',
        '--loadSize', '256',
        '--checkpoints_dir', 'EXP_checkpoints',
        '--which_epoch', 'latest',
        '--pre_trained_resnet', '0',
        '--pre_trained_transformer', '0'
    ]

    # ------------------- 3. 组合成最终的命令 -------------------
    # 在这里添加了所有可变参数，包括新的 --modality 参数
    full_command = base_command + [
        '--dataroot', str(dataroot),
        '--how_many', str(how_many),
        '--results_dir', str(results_dir),
        '--gpu_ids', str(gpu_id),
        '--modality', str(modality)
    ]

    # ------------------- 4. 执行命令并展示结果 -------------------
    print("="*80)
    print(f"准备执行统一测试: 数据集={dataset}, 模态={modality}")
    command_str = ' '.join(full_command)
    print(f"将要执行的完整命令:\n{command_str}")
    print("="*80)

    try:
        # 使用 subprocess.Popen 实时打印子进程的输出，确保您能看到测试过程
        process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   text=True, encoding='utf-8', bufsize=1)
        for line in process.stdout:
            print(line, end='')
        
        process.wait() # 等待子进程完全结束
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)

        print(f"\n测试成功完成: 数据集={dataset}, 模态={modality}")
        print(f"结果图片已保存在目录: {results_dir}")

        # --- 读取并打印最终的指标文件 ---
        metrics_file = os.path.join(results_dir, 'metrics.txt')
        if os.path.exists(metrics_file):
            print("\n---------- 最终测试指标总结 ----------")
            with open(metrics_file, 'r', encoding='utf-8') as f:
                print(f.read())
            print("--------------------------------------\n")
        else:
            print(f"\n警告: 未找到指标文件 {metrics_file}。")

    except FileNotFoundError:
        print(f"\n错误: 找不到执行脚本 '{script_name}'。请确保此脚本与 'run_test.py' 在同一目录下。")
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 执行命令时发生错误。返回码: {e.returncode}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统一的MRF神经网络测试脚本')
    parser.add_argument('--dataset', type=str, required=True, choices=['brainweb', 'subjects'],
                        help="选择要测试的数据集: 'brainweb' 或 'subjects'")
    parser.add_argument('--modality', type=str, required=True, choices=['t1', 't2'],
                        help="选择要测试的模态: 't1' 或 't2'")
    parser.add_argument('--gpu', type=str, default='0', help="指定使用的GPU ID，默认为 '0'")
    args = parser.parse_args()
    run_test(args.dataset, args.modality, args.gpu)
