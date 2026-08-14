# Results

这里只保留统一尺度协议的正式小型产物：

- `metrics.json`：论文/README 主表的紧凑机器可读版本；
- `manifest.json`：两份生产 checkpoint、协议和结果文件的 SHA-256；
- `slabim/`：108 帧 pooled-clean test 的 summary、逐帧 CSV、三维重建 summary、history 与
  run-state；
- `stanford_area1/`：room-disjoint validation/test 的 summary、逐帧 CSV、history 与 run-state。

旧 q45-only SLABIM、BIM-direct 网络锚点、E2E challenger、region-CV、旧消融和事后区域分析
均已从活动项目删除，避免与当前公共方法混用。历史 summary 内的绝对路径只是审计字段；公开
复现应生成本机 child config/receipt，不能手工关闭 provenance 校验。

本机 checkpoint 校验：

```bash
sha256sum -c results/checkpoints.sha256
```

`outputs/` 默认不进入 Git。发布者应把两份约 30 MB 的 checkpoint 上传至 Release/Hugging
Face，并在 `manifest.json` 补充 URL；不得把第三方数据或大型旧 E2E 权重提交到 Git 历史。

SLABIM 三维融合评测在 BIM 坐标中使用冻结恢复位姿、4 像素采样步长和 5 cm voxel：
learned 的 Chamfer-L1 为 0.09109 m（direct 0.10515 m），F-score@10 cm 为 0.79622
（direct 0.74003）。详见 `slabim/reconstruction_test/summary.json`。
