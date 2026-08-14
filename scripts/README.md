# Script layout

CLI 按职责分组，所有命令从仓库根目录执行：

| 目录 | 边界 |
|---|---|
| `data/` | 数据下载、来源校验、位姿/配准、DA3 缓存、样本制备、split 与审计 |
| `model/` | 训练、无 GT 推理、2D 深度评测与 3D 重建评测 |
| `pipelines/` | 可续跑的端到端编排和环境 smoke check |
| `analysis/` | 不修改训练状态的统计、bootstrap 和可视化 |

数据实现的共享逻辑位于 `src/bim_priorda3/data/`，不应通过从一个 CLI 导入
另一个 CLI 来复用。新的共享功能应先放入 `src/`，CLI 只负责参数解析和结果
落盘。

- 训练前数据流程：[`docs/DATA_PREPARATION.md`](../docs/DATA_PREPARATION.md)
- 全部脚本索引和模型命令：[`docs/USER_GUIDE.md`](../docs/USER_GUIDE.md)
- 端到端实验顺序：[`docs/EXPERIMENT_PIPELINE.md`](../docs/EXPERIMENT_PIPELINE.md)
- 科研评测、消融和图件协议：[`docs/EVALUATION_PROTOCOL.md`](../docs/EVALUATION_PROTOCOL.md)

`scripts/` 不是 wheel CLI 包；公开复现模式是 clone 仓库、执行
`pip install -e ...`，然后从仓库根目录运行 `python scripts/<group>/<tool>.py`。
