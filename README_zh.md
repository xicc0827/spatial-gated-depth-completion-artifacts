# 空间门控深度补全适配：复现工件

本仓库保存论文使用的冻结协议、逐条件评价记录、区域误差分解、采样清单、检查点哈希、重聚合脚本和核心插图统计。

运行 `python scripts/verify_artifacts.py` 可核验公开文件；运行 `python scripts/reaggregate.py` 可直接从逐图像记录复算 DIODE 与 ARKitScenes 的主要 RMSE 结果。复算不需要原始数据和模型权重。

仓库不分发 NYUv2、DIODE、ARKitScenes 原始数据，也不分发 CompletionFormer 或 OMNI-DC 权重。OR2 中含 DIODE RGB/深度像素的 10 个 NPZ 示例已从公开包剔除，其原始文件名和哈希保留在标记记录中。公开版已把本机绝对路径替换为环境变量占位符。
