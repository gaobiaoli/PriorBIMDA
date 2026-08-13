# Results

本目录只保存可公开审计的小型正式产物；训练中间 checkpoint、日志和缓存不在这里。

- `metrics.json`：论文/README 主表的紧凑机器可读版本。
- `manifest.json`：生产 checkpoint 的角色、字节数和 SHA-256；`publish=false` 表示未晋级。
- `checkpoints.sha256`：本机 `outputs/` 中三个保留 checkpoint 的校验表。
- `slabim/`：frozen/E2E test summary、逐帧 CSV、训练 history/run-state 和历史结构
  消融摘要。
- `stanford_area1/`：zero-shot val、target frozen val/test、未晋级 E2E val 结果。
  target frozen 与 E2E challenger 的 history/run-state 也保留在该目录。
- `region_cv/`：SLABIM 六区域交叉验证汇总；缺失的 seed 行在历史报告中如实标记，不应
  被解释为完整 3-seed 结果。
- `region_analysis/`：区域误差因素、相关性和示例选择数据。

正式 summary 是原始评测输出，内部可能含生成机器的绝对路径；这些字段只用于历史审计，
不参与运行时文件解析。公开复现应生成新的 local receipt/config，而不是编辑历史 summary。

校验本机 checkpoint：

```bash
sha256sum -c results/checkpoints.sha256
```

`outputs/` 被 `.gitignore` 排除。发布者应把 checkpoint 作为 GitHub Release/Hugging Face
asset 上传，并把 URL 写入 `manifest.json`；不要把 1.3 GB E2E 权重提交到 Git 历史。
