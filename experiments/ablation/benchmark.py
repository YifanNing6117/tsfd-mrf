import torch
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table


# 导入你想要测试的模型
# from models.LG_ViT import LG_ViT
# from models.sandwich_v6 import SandwichVideoMamba

def benchmark_model(model, input_shape=(1, 400, 256, 256), device='cuda'):
    print(f"========== 评测开始: {model.__class__.__name__} ==========")

    # 1. 准备模型和输入
    model = model.to(device)
    model.eval()  # 必须开启 eval 模式，关闭 Dropout 和 BatchNorm 的动态更新
    dummy_input = torch.randn(*input_shape).to(device)

    # ---------------------------------------------------------
    # 测评 1: 计算参数量 (Params) 和 计算复杂度 (FLOPs)
    # ---------------------------------------------------------
    print("\n[1] 计算复杂度 (FLOPs) & 参数量 (Params)...")
    flops = FlopCountAnalysis(model, dummy_input)

    # 获取总 FLOPs 和 Params
    total_flops = flops.total()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"总参数量 (Params): {total_params / 1e6:.2f} M (Million)")
    print(f"计算复杂度 (FLOPs): {total_flops / 1e9:.2f} G (Giga-FLOPs)")

    # (可选) 打印详细的网络层级参数表，如果不需要可以注释掉
    # print("\n详细层级占用:")
    # print(parameter_count_table(model))

    # ---------------------------------------------------------
    # 测评 2: 计算实际推理时间 (Inference Time)
    # ---------------------------------------------------------
    print("\n[2] 测试推理时间 (Inference Time)...")

    # A. 预热 (Warm-up)
    # GPU 刚开始运行时会有初始化延迟，必须先跑几轮预热
    print("正在预热 GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # B. 正式计时
    # CUDA 是异步执行的，必须使用 torch.cuda.synchronize() 等待 GPU 完成所有任务
    iterations = 50  # 测 50 次取平均
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize()  # 再次同步，确保 50 次全部跑完
    end_time = time.time()

    # C. 计算平均时间
    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_time_ms

    print(f"平均推理时间 (Time per image): {avg_time_ms:.2f} ms")
    print(f"每秒处理帧数 (FPS): {fps:.2f} frames/s")
    print("========================================================\n")


if __name__ == "__main__":
    # 这里实例化你要测试的模型
    # 假设我们测 LG_ViT
    from models.SOTIP import SOTIP_Net  # 替换成你实际的 import 路径

    model = STQ_Net(in_chans=400, num_out_chans=2, img_size=256)

    # 传入模型和符合你任务的输入尺寸[Batch=1, C=400, H=256, W=256]
    benchmark_model(model, input_shape=(1, 400, 256, 256))
