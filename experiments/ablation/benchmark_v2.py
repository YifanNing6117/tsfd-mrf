import torch
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch.utils.data import Dataset, DataLoader

# 模拟一个真实的数据集加载过程
class DummyDataset(Dataset):
    def __init__(self, shape, length=100):
        self.shape = shape # 这里的 shape 不需要 batch 维度，如 (400, 256, 256)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 模拟 CPU 读取图像/处理数据的耗时
        return torch.randn(*self.shape)

def benchmark_model_with_dataloader(model, input_shape=(1, 400, 256, 256), device='cuda', num_workers=2):
    print(f"========== 端到端评测开始: {model.__class__.__name__} ==========")

    # 1. 准备模型
    model = model.to(device)
    model.eval()
    
    # 拆解 input_shape
    batch_size = input_shape[0]
    data_shape = input_shape[1:]  # 去掉 batch 维度，供 Dataset 使用

    # ---------------------------------------------------------
    # 测评 1: 计算参数量 (Params) 和 计算复杂度 (FLOPs)
    # ---------------------------------------------------------
    print("\n[1] 计算复杂度 (FLOPs) & 参数量 (Params)...")
    dummy_input = torch.randn(*input_shape).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False) # 屏蔽不必要的警告

    total_flops = flops.total()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"总参数量 (Params): {total_params / 1e6:.2f} M (Million)")
    print(f"计算复杂度 (FLOPs): {total_flops / 1e9:.2f} G (Giga-FLOPs)")

    # ---------------------------------------------------------
    # 测评 2: 端到端推理时间 (Data Loading + Transfer + Inference)
    # ---------------------------------------------------------
    print(f"\n[2] 测试端到端推理时间 (DataLoader 模拟)...")
    
    # 构建 DataLoader
    dataset = DummyDataset(data_shape, length=200) # 生成足够多的样本
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    
    print("正在预热 GPU...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # 准备记录各个阶段的时间
    iterations = 50
    data_load_times = []  # CPU 准备数据的时间
    transfer_times = []   # CPU 搬运到 GPU 的时间
    compute_times = []    # GPU 实际推理计算的时间
    total_pipeline_times = [] # 单个 Batch 的端到端总时间

    torch.cuda.synchronize()
    data_iter = iter(dataloader)

    with torch.no_grad():
        for i in range(iterations):
            # 记录整个 Pipeline 起点
            t_pipeline_start = time.time()

            # A. 测量数据加载时间 (CPU 耗时)
            t0 = time.time()
            batch_data = next(data_iter)
            t1 = time.time()

            # B. 测量数据搬运时间 (CPU -> GPU)
            batch_data = batch_data.to(device)
            torch.cuda.synchronize()  # 等待搬运完成
            t2 = time.time()

            # C. 测量模型前向推理时间 (GPU 耗时)
            _ = model(batch_data)
            torch.cuda.synchronize()  # 等待计算完成
            t3 = time.time()

            # 统计各部分耗时 (单位: ms)
            data_load_times.append((t1 - t0) * 1000)
            transfer_times.append((t2 - t1) * 1000)
            compute_times.append((t3 - t2) * 1000)
            total_pipeline_times.append((t3 - t_pipeline_start) * 1000)

    # 计算平均时间
    avg_data_load = sum(data_load_times) / iterations
    avg_transfer = sum(transfer_times) / iterations
    avg_compute = sum(compute_times) / iterations
    avg_total = sum(total_pipeline_times) / iterations

    # 根据端到端总时间计算真实的 FPS
    fps = (1000 / avg_total) * batch_size

    print(f"数据加载线程数 (num_workers): {num_workers}")
    print(f"--- 耗时拆解 (每个 Batch, Batch_Size={batch_size}) ---")
    print(f" 1. 数据读取 (Data Loading) : {avg_data_load:.2f} ms (占 {avg_data_load/avg_total*100:.1f}%)")
    print(f" 2. 显存拷贝 (CPU -> GPU)   : {avg_transfer:.2f} ms (占 {avg_transfer/avg_total*100:.1f}%)")
    print(f" 3. 模型推理 (GPU Compute)  : {avg_compute:.2f} ms (占 {avg_compute/avg_total*100:.1f}%)")
    print(f"--------------------------------------------------")
    print(f"端到端总耗时 (Total latency): {avg_total:.2f} ms")
    print(f"真实系统处理帧数 (True FPS) : {fps:.2f} frames/s")
    print("========================================================\n")


if __name__ == "__main__":
    # 模拟导入你的模型
    from models.STQ_net import STQ_Net  # 替换成你实际的 import 路径

    model = STQ_Net(in_chans=400, num_out_chans=2, img_size=256)

    # 传入模型和输入尺寸，这里 num_workers 可以根据你 CPU 核心数调整
    benchmark_model_with_dataloader(
        model=model, 
        input_shape=(1, 400, 256, 256), 
        device='cuda',
        num_workers=4  # 修改数据加载的线程数
    )

