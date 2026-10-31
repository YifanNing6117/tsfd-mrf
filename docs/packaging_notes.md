# 整理说明

主 V5 网络直接复制自回收代码，未修改网络计算、参数名或权重格式。

三个实验目录中的历史 `models/test_model.py` 原先使用已成为 Python 关键字的 `async=True`。整理时将其替换为 `non_blocking=True`，将旧 `util` 导入改为现有 `util1`，并按当前 `define_G` 函数签名补齐调用参数。最终 V5 实验使用 `resvit_one` 入口，该历史 TestModel 的兼容性调整不改动其网络实现。

数据 CSV 的绝对路径替换为本地 `DATA_ROOT` 占位路径；原样本划分和行顺序保留。对应原始哈希、修改说明及整理后哈希见 `provenance/source_manifest.json`。

语法检查覆盖所有 Python 文件。凭据检查采用私钥、GitHub token、AWS access-key ID 和常见 API-key 的固定模式，检查结果见 `provenance/validation.json`。没有重新执行训练或 GPU 推理。
