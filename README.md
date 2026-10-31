# TSFD-MRF

TSFD-MRF 的 DWTMAMBA V5 网络、实验入口和已保存结果。主模型实现为 `tsfd_mrf/model.py` 中的 `CSBNet`，与回收的 `CSB-NET/models/DWT_tfmamba_MPU_v5.py` 内容一致。原实验框架通过 `DWTtfmambampu_v5` 选择该网络。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `tsfd_mrf/model.py` | 独立的 V5 网络实现 |
| `experiments/brainweb_in_vivo/` | BrainWeb 与 in vivo 数据加载、训练、评估及原框架依赖 |
| `experiments/delics/` | Deli-CS 实验入口、数据加载与模型变体 |
| `experiments/ablation/` | 消融实验代码和比较模型 |
| `results/raw/` | 原始 TXT 汇总、指标、训练日志和选项记录 |
| `results/index.csv` | 结果文件索引，区分指标、训练日志与选项 |
| `results/paper_images/` | Overleaf 项目中的实验对比图和消融图 |
| `checkpoints/archive_catalog.json` | 权重包成员、大小和 SHA-256 |
| `provenance/source_manifest.json` | 回收文件到仓库文件的映射与校验值 |

## 查看网络

安装与本机 CUDA 匹配的 PyTorch，随后安装 `requirements.txt` 中的依赖。`mamba-ssm` 的 selective scan 需要兼容的运行环境；本次整理没有重新训练或执行完整 GPU 推理。

```python
from tsfd_mrf import CSBNet

model = CSBNet(num_out_chans=2, img_size=256)
```

当前 V5 的首层卷积固定接收 400 个通道。短序列实验需按原数据加载器补齐到 400；仅修改构造函数的 `in_chans` 参数不会改变首层卷积。原实验入口通常分别评估 T1、T2，请保留对应输出通道配置和权重。

## 运行原实验入口

在目标实验目录运行脚本，避免 `data`、`models` 和 `options` 之间的相对导入冲突。例如：

```bash
pip install -r requirements-experiments.txt
cd experiments/brainweb_in_vivo
python run_all_tests.py --gpu 0
```

运行前需准备原始数据和相应权重，并检查入口顶部的模型名、数据 CSV、序列长度和 GPU 编号。CSV 保留原样本顺序与划分，数据路径统一改为 `./DATA_ROOT/<原父目录>/<文件名>`，请根据实际位置替换。权重仍放在各实验目录下的 `EXP_checkpoints/<实验名>/latest_net_G.pth`。

Deli-CS 的批量入口为 `experiments/delics/run_all_delics_tests.py`；消融入口为 `experiments/ablation/run_all_tests.py`。这些目录保留了历史框架为模型选择器导入的其他实现；最终主模型以 V5 文件及入口中的 `DWTtfmambampu_v5` 为准。

## 结果来源

仓库收录历史实验的原始记录，包含不同学习率、序列长度和消融配置。它们不代表同一个最终配置；请结合每份汇总中的模型名和对应 `opt.txt` 阅读。`results/index.csv` 可定位全部原始记录。论文图片直接取自提供的 Overleaf ZIP，没有重新计算指标。

例如，`results/raw/brainweb_in_vivo/final_summary_20260330_150903.txt` 记录以下均值：

| 数据集 | 模态 | PSNR | SSIM | NMSE |
| --- | --- | ---: | ---: | ---: |
| Subjects | T1 | 33.8665 | 0.9883 | 0.005524 |
| Subjects | T2 | 35.1340 | 0.9874 | 0.037107 |
| BrainWeb | T1 | 50.0097 | 0.9994 | 0.000100 |
| BrainWeb | T2 | 46.9943 | 0.9990 | 0.000376 |

该表是指定记录的摘录；论文表格与其他实验应分别核对来源。较早的失败测试和效果较差的配置也保留在原始结果中。

## 权重与数据

大型权重和原始数据尚未上传。五个已打包权重文件共约 17.38 GB，覆盖 29 个实验的 latest 和最高保存轮次，目录与校验信息见 `checkpoints/archive_catalog.json`。目前没有对外权重下载链接。

Windows 会合并仅大小写不同的目录，因此消融实验的原 `50SOTIP` 目录映射为 `50SOTIP__uppercase`，原 `50sotip` 保留。消融权重包使用相同映射。`50woswin` 只找到第 19 轮权重，其余所选最高轮次为 50。

## 验证与架构说明

本次整理检查了 Python 语法、文件来源哈希和凭据模式，详细结果见 `provenance/validation.json`。本地尚未验证完整依赖环境、训练或权重推理。

TSM 内部先对 PT-Mamba 和 GSF 输出做 sigmoid 门控融合，再加回 TSM 输入；CIFM 内部将共享交互特征分别加到两路输入上。CIFM 返回后，外部没有再次做同一残差相加。详见 `docs/architecture.md`。

仓库保留原文件内的署名和依赖引用，未为历史第三方实现新增统一授权协议。
