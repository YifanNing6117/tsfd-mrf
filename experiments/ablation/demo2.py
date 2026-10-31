import torch
import traceback
from fvcore.nn import FlopCountAnalysis

def get_model_flops(model_class, in_chans=400):
    print("========== 开始专门计算 FLOPs (降维安全版) ==========")
    device = 'cuda'

    # ==========================================
    # 策略 A: 测 128x128 (计算量是 256x256 的 1/4)
    # ==========================================
    print("\n[尝试 1] 测算 128x128 尺寸 (由于面积是原本的1/4，结果将乘以 4)...")
    try:
        model_128 = model_class(in_chans=in_chans, num_out_chans=2, img_size=128).to(device)
        model_128.eval()
        
        dummy_128 = torch.randn(1, in_chans, 128, 128).to(device)
        
        flops = FlopCountAnalysis(model_128, dummy_128)
        flops.unsupported_ops_warnings(False)
        flops_128 = flops.total()
        
        print(f"✅ 128x128 测试成功！消耗 FLOPs: {flops_128 / 1e9:.2f} G")
        print(f"🎉 换算得到 256x256 真实 FLOPs: {flops_128 * 4 / 1e9:.2f} G (Giga-FLOPs)")
        
        # 测算成功直接返回，清理显存
        del model_128, dummy_128, flops
        torch.cuda.empty_cache()
        return

    except Exception as e:
        print(f"❌ 128x128 依然失败，清理显存准备尝试更小尺寸...")
        torch.cuda.empty_cache()

    # ==========================================
    # 策略 B: 测 64x64 (计算量是 256x256 的 1/16) 保底方案
    # ==========================================
    print("\n[尝试 2] 测算 64x64 尺寸 (由于面积是原本的1/16，结果将乘以 16)...")
    try:
        model_64 = model_class(in_chans=in_chans, num_out_chans=2, img_size=64).to(device)
        model_64.eval()
        
        dummy_64 = torch.randn(1, in_chans, 64, 64).to(device)
        
        flops = FlopCountAnalysis(model_64, dummy_64)
        flops.unsupported_ops_warnings(False)
        flops_64 = flops.total()
        
        print(f"✅ 64x64 测试成功！消耗 FLOPs: {flops_64 / 1e9:.2f} G")
        print(f"🎉 换算得到 256x256 真实 FLOPs: {flops_64 * 16 / 1e9:.2f} G (Giga-FLOPs)")

    except Exception as e:
        print("\n[致命错误] 连 64x64 都爆显存了，请检查模型代码是否存在异常：")
        traceback.print_exc()

if __name__ == "__main__":
    from models.DWT_tfmamba_MPU_v5 import CSBNet  # 替换成你的实际导入
    
    # 传入类本身，方便脚本内部实例化不同大小的模型
    get_model_flops(CSBNet, in_chans=400)
