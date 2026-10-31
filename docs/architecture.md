# V5 架构与原图核对

最终主模型位于 `tsfd_mrf/model.py`。其 `SpatialMambaBlock` 对应论文 TSM，`ManifoldProjectionUnit` 对应论文 CIFM。

每个 TSM 子层计算：

```python
residual = x
x_temporal_res = temporal(LayerNorm(x))
x_freq_res = frequency(x)
alpha = sigmoid(learnable_scalar)
x_fused = alpha * x_temporal_res + (1 - alpha) * x_freq_res
x = residual + x_fused
```

`alpha` 是每个子层独立的可训练标量，经 sigmoid 映射后用于两分支加权；推理时不会根据当前样本生成新的门控值。PT-Mamba 输出还包含其内部归一化和训练时的 DropPath。

CIFM 计算双向交叉注意力，将两路交互结果拼接后通过 `shared_conv` 生成共享特征，再执行：

```python
x_m_refined = x_mamba + shared_feat
x_s_refined = x_swin + shared_feat
out_m = proj_m(x_m_refined)
out_s = proj_s(x_s_refined)
```

`CascadeStage.forward` 直接返回 CIFM 的两个输出。画图时，若 CIFM 展开图中已包含这两个残差加法，总架构的 CIFM 方框应直接返回融合后的两路特征，避免再画成一次外部残差运算。TSM 自身的输入残差是另一处运算，也应在 TSM 展开图中标出。

本仓库没有将此前缺少残差线的架构图作为修正后的图发布。
