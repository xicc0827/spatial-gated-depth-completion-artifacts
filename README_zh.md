# 空间门控深度补全适配：复现工件

本仓库保存论文使用的冻结协议、逐条件评价记录、区域误差分解、采样清单、检查点哈希、重聚合脚本和核心插图统计。

运行 `python scripts/verify_artifacts.py` 可核验公开文件；运行 `python scripts/reaggregate.py` 可直接从逐图像记录复算 DIODE 与 ARKitScenes 的主要 RMSE 结果。复算不需要原始数据和模型权重。

仓库不分发 NYUv2、DIODE、ARKitScenes 原始数据，也不分发 CompletionFormer 或 OMNI-DC 权重。OR2 中含 DIODE RGB/深度像素的 10 个 NPZ 示例已从公开包剔除，其原始文件名和哈希保留在标记记录中。公开版已把本机绝对路径替换为环境变量占位符。

本仓库原创代码、文档和派生研究工件按根目录 `LICENSE` 中的 MIT 许可证发布。该许可证不改变第三方数据集、上游项目、模型权重或 `resources/or3/source` 中 ARKitScenes 源快照的原始许可；具体边界和上游许可证副本见 `THIRD_PARTY_NOTICES.md`。

`figures` 目录包含三幅定量核心图的统计量与生成代码，以及定性成功／失败案例的机器可读筛选记录；第三方 RGB／深度像素仍不公开。

`training_snapshot/resource_comparison.json` 记录参数量、保留与额外执行更新、累计参数更新量及推理计时的解释边界；累计参数更新量只是更新规模代理，不代表 FLOPs、显存、墙钟时间或能耗。

## 版本固化

现有 `v1.0-submission` 标签保留为首个公开工件快照，不移动该标签。许可证和标签之后的文档更新将通过新的 `v1.0.1-submission` 发布版固化。`ZENODO_RELEASE_CHECKLIST.md` 记录尚需补齐的元数据及发布步骤。当前尚未生成 DOI；Zenodo 完成归档后，将在仓库中补入概念 DOI 和版本 DOI。
