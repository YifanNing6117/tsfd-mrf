# TSFD-MRF

最终主模型 DWTMAMBA V5 的网络实现与选定实验结果。

## 文件

| 文件 | 内容 |
| --- | --- |
| `model/dwt_mamba_v5.py` | 完整 V5 网络，入口类 `TSFDMRF` |
| `results/brainweb_invivo_400frames.csv` | 400 帧 BrainWeb、in vivo 的 T1/T2 指标 |
| `results/brainweb_invivo_200frames.csv` | 200 帧 BrainWeb、in vivo 的 T1/T2 指标 |
| `results/delics_2min.csv` | Deli-CS 两分钟采集的 T1/T2 指标 |
| `results/ablation_400frames.csv` | 完整模型与去除 GSF、LWE、CIFM 的对照指标 |
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

## 结果来源

400 帧结果来自 2026-03-30 的最终主模型记录；200 帧结果来自 2026-03-19 的短序列记录。Deli-CS 采用 2026-03-25、与论文主结果相符的记录，不混入后来不同测试设置下的结果。消融结果来自 2026-03-29 的三个对应模块实验。

CSV 保留原记录的均值和标准差，并统一模型、数据集与模态名称。`without_gsf`、`without_lwe`、`without_cifm` 分别表示去除对应模块。原记录含单位时保留 `mae_unit=ms`；其余标记为 `reported_scale`，不将其换算为毫秒。Deli-CS 的两分钟指采集时长，CSV 不据此推断指纹帧数。

本次整理验证了主模型语法、重命名前后的计算结构，以及结果数值提取；没有重新执行训练或 GPU 推理。
