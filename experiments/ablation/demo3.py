import torch
import numpy as np

def standard_speed_test(model_class, input_shape=(1, 400, 256, 256)):
    print(f"========== 标准常规推理测速: {model_class.__name__} ==========")
    device = 'cuda'

    # 1. 常规加载模型 (保持默认的全精度 FP32 权重)
    model = model_class(in_chans=input_shape[1], num_out_chans=2, img_size=input_shape[2])
    model = model.to(device)
    model.eval()

    # 2. 常规生成输入数据
    dummy_input = torch.randn(*input_shape).to(device)

    # 3. 标准预热阶段 (Warm-up)
    print("正在进行标准预热 (10次前向传播)...")
    with torch.no_grad():
        # 默认开启自动混合精度(AMP)，这是学术界测算大模型时的标准规范
        with torch.autocast(device_type=device, dtype=torch.float16):
            for _ in range(10):
                _ = model(dummy_input)

    # 4. 正式测速阶段
    iterations = 50
    print(f"开始正式测速 (循环 {iterations} 次取平均)...")
    
    # 记录每次的耗时
    timings = np.zeros((iterations, 1))
    
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16):
            for i in range(iterations):
                # 使用标准的 CUDA 时钟记录器 (排除 Python 自身代码运行的误差)
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                
                starter.record()
                _ = model(dummy_input)
                ender.record()
                
                # 同步，确保当前这次循环彻底算完
                torch.cuda.synchronize()
                
                # 计算这单次的时间 (毫秒)
                curr_time = starter.elapsed_time(ender)
                timings[i] = curr_time

    # 5. 统计计算
    avg_time_ms = np.mean(timings)
    std_time_ms = np.std(timings) # 计算方差，看看波动大不大
    fps = 1000.0 / avg_time_ms

    print("\n" + "="*50)
    print("�� 【常规推理速度报告 (Academic Standard)】")
    print(f"输入尺寸: {input_shape}")
    print(f"平均单次耗时 (Latency): {avg_time_ms:.2f} ms  (±{std_time_ms:.2f} ms)")
    print(f"每秒处理帧数 (FPS): {fps:.2f} 帧/秒")
    print("="*50 + "\n")


if __name__ == "__main__":
    from models.DWT_tfmamba_MPU_v5 import CSBNet  # 替换成你的实际导入
    
    standard_speed_test(CSBNet, input_shape=(1, 400, 256, 256))

