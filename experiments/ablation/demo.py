import torch
import time
from fvcore.nn import FlopCountAnalysis

def benchmark_model_cpu_gpu(model, input_shape=(1, 400, 256, 256), gpu_device='cuda'):
    print(f"========== 评测开始: {model.__class__.__name__} ==========")

    # =========================================================
    # 第一步: 在 CPU 上计算参数量和 FLOPs (利用大内存防爆)
    # =========================================================
    print("\n[1] 在 CPU 上计算复杂度 (FLOPs) & 参数量 (Params)...")
    
    # 强制将模型放到 CPU 
    model = model.to('cpu')
    model.eval()
    
    # 参数量不区分设备，直接计算
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量 (Params): {total_params / 1e6:.2f} M (Million)")

    # 生成 CPU 上的输入
    dummy_input_cpu = torch.randn(*input_shape).to('cpu')
    
    try:
        flops = FlopCountAnalysis(model, dummy_input_cpu)
        flops.unsupported_ops_warnings(False) # 屏蔽不支持算子的满屏警告
        total_flops = flops.total()
        print(f"计算复杂度 (FLOPs): {total_flops / 1e9:.2f} G (Giga-FLOPs)")
    except Exception as e:
        print(f"\n[注意] CPU 上测 FLOPs 失败: {e}")
        print("原因分析: 你的模型(特别是Mamba)中包含了官方纯CUDA算子，不支持在CPU上执行！")
        print("如果是这样，你只能把 channel 调小 (如100) 放到 GPU 上测出单份 FLOPs 再乘 4。")
    
    # 测试完后，彻底释放 CPU 上的大张量，防止占用系统内存
    del dummy_input_cpu
    if 'flops' in locals(): 
        del flops

    # =========================================================
    # 第二步: 将模型转移到 GPU，仅测试纯净的推理时间
    # =========================================================
    print(f"\n[2] 将模型转移到 {gpu_device.upper()} 测试推理时间 (Inference Time)...")
    
    try:
        model = model.to(gpu_device)
        dummy_input_gpu = torch.randn(*input_shape).to(gpu_device)
    except RuntimeError as e:
        if "Out of memory" in str(e):
            print(f"\n[致命错误] 仅仅是把模型和输入放到 GPU 上就已经爆显存了！")
            return
        raise e

    # A. 预热 (Warm-up)
    print("正在预热 GPU (开启 FP16 自动混合精度，节省一半显存)...")
    try:
        with torch.no_grad():
            # 使用 autocast 开启半精度推理，进一步防止爆显存，且符合主流推理部署情况
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for _ in range(10): 
                    _ = model(dummy_input_gpu)

        # B. 正式计时
        iterations = 30  # 为了避免跑太久，取30次平均
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for _ in range(iterations):
                    _ = model(dummy_input_gpu)

        torch.cuda.synchronize() 
        end_time = time.time()

        # C. 计算平均时间
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        fps = 1000 / avg_time_ms

        print(f"平均推理时间 (Time per pass): {avg_time_ms:.2f} ms")
        print(f"每秒处理帧数/次 (FPS): {fps:.2f} passes/s")
        
    except RuntimeError as e:
        if "Out of memory" in str(e):
            print("\n[错误] 推理时间测试失败：前向传播时 GPU 爆显存 (Out of Memory)！")
            print("结论: 哪怕不测FLOPs、开半精度，你的显卡依然装不下 400 个通道的前向特征图。建议换A100或缩小通道数测试。")
        else:
            print(f"推理测试发生未知错误: {e}")

    print("========================================================\n")


if __name__ == "__main__":
    from models.DWT_tfmamba_MPU_v5 import CSBNet  # 替换成你的实际导入
    
    model = CSBNet(in_chans=400, num_out_chans=2, img_size=256)
    
    # 启动评测
    benchmark_model_cpu_gpu(model, input_shape=(1, 400, 256, 256))

