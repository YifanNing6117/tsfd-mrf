# TSFD-MRF

最终主模型 DWTMAMBA V5 的网络实现与论文结果表。

## 文件

| 文件 | 内容 |
| --- | --- |
| `model/dwt_mamba_v5.py` | 完整 V5 网络，入口类 `TSFDMRF` |
| `images/brainweb_comparison.png` | BrainWeb：200、400 帧的完整方法对比 |
| `images/invivo_comparison.png` | in vivo 3D MRF：200、400 帧的完整方法对比 |
| `images/delics_comparison.png` | Deli-CS：论文两组设置的完整方法对比 |
| `images/model_complexity.png` | 参数量与 FLOPs 对比 |
| `images/component_ablation.png` | 三个数据集、五种模块消融与完整模型的对比 |
| `requirements.txt` | 主模型所需依赖 |

## 主模型

安装与 CUDA 环境匹配的 PyTorch，以及 `requirements.txt` 中的依赖。Mamba selective scan 需要兼容的运行环境。

```python
from model.dwt_mamba_v5 import TSFDMRF

model = TSFDMRF(num_out_chans=2, img_size=256)
```

输入为 `[B, 400, 256, 256]`。200 帧实验先按原预处理补齐到 400 通道。首层卷积固定接收 400 通道，`in_chans` 参数不会修改首层卷积。

文件名、主类和主要模块类名已按论文术语整理：TSM、PT-Mamba、GSF、CIFM。网络计算和参数属性保持与回收的最终 V5 一致；重命名类不会改变 `state_dict` 的参数键。该文件包含主网络定义，训练和数据处理流程保留在本地原项目中。

TSM 先对 PT-Mamba 与 GSF 输出做 sigmoid 门控融合，再加回输入残差。CIFM 将共享交互特征分别加回两路输入，外部不再重复执行同一次残差相加。

## 论文结果

以下图片直接根据提供的 Overleaf 项目 `mrf.tex` 中的结果表生成，保留论文中的方法、数值精度、标准差及加粗和下划线标记。in vivo 与 Deli-CS 在原稿中属于同一张表，在这里分开显示；每张方法对比图将两组序列长度上下排布，方便在线查看。

这些是论文报告值，与个别原始 TXT 记录存在差异。此前由 TXT 提取的 CSV 留在本地存档，本页统一展示论文表格中的数据。此次没有重新计算指标或执行训练、GPU 推理。

### BrainWeb

![BrainWeb comparison](images/brainweb_comparison.png)

### In Vivo 3D MRF

![In vivo comparison](images/invivo_comparison.png)

### Deli-CS

![Deli-CS comparison](images/delics_comparison.png)

### Model Complexity

![Model complexity](images/model_complexity.png)

### Component Ablation

![Component ablation](images/component_ablation.png)
