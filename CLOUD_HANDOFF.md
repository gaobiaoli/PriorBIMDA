# BIM-PriorDA3 云端会话交接

> 新会话第一句话建议使用：
>
> `请先完整阅读 /workspace/BIM-PriorDA3/chatlog.md、CLOUD_HANDOFF.md、RESULTS_V3.md 和
> configs/slabim_single_frame_r50_v3.yaml，再检查云端环境。不要重新使用5FR2调参；
> 从“下一步工作”继续。`

## 1. 当前目标

面向土木室内重建，研究单帧 RGB + BIM 先验对 DA3 metric depth 的增强。最终深度用于三维
重建。融合 LiDAR PCD 仅用于监督和评测，不应进入深度模型前向。

当前实验使用 SLABIM 六个区域：

- 训练：3F_Region2、3F_Region3、4F_Region2、4F_Region3，共 565 帧；
- 验证：5F_Region3，共 82 帧；
- 测试：5F_Region2，共 164 帧。

GT 为中心帧前后各 50 个 PCD、最多 101 个扫描的遮挡感知融合。每个扫描先单独投影和
z-buffer，再保留最前方且深度一致的扫描簇。

## 2. 当前最佳方法

V1 从原始 DA3 自由预测残差，在 5FR3 验证集较好，但 5FR2 出现灾难帧。V2/V2.1 尝试
从解析强锚点预测有界残差，分别过于保守或跨区域过拟合。

当前最佳 V3 使用学习式候选融合：

1. 候选 A：全局 BIM/DA3 尺度 + 平滑局部 BIM 校正的解析强锚点；
2. 候选 B：冻结 V1 的学习结果；
3. V3 学习两者在 log-depth 空间的逐像素凸组合；
4. 候选门控监督直接比较两个候选相对融合 PCD GT 的误差；
5. 推理安全层依据帧级可信度决定整帧融合或回退强锚点。

关键代码：

- `src/bim_priorda3/baselines.py`
- `src/bim_priorda3/models/system.py`
- `src/bim_priorda3/losses.py`
- `src/bim_priorda3/data/slabim.py`
- `src/bim_priorda3/data/pose_recovery.py`
- `src/bim_priorda3/reconstruction.py`
- `scripts/prepare_strong_anchors.py`
- `scripts/cache_candidate_predictions.py`
- `scripts/run_slabim_experiments.py`
- `docs/EXPERIMENT_PIPELINE.md`

## 3. 已确认结果

5FR2、30,807,322 个相同 GT 有效像素：

| 方法 | AbsRel | RMSE m | MAE m | delta1 |
|---|---:|---:|---:|---:|
| 原始 DA3 | 0.24683 | 0.45560 | 0.34806 | 0.63132 |
| 全局尺度 | 0.06726 | 0.24093 | 0.10410 | 0.97402 |
| 强尺度 + 局部 BIM | 0.05260 | 0.22822 | 0.08253 | 0.97441 |
| V1 | 0.07739 | 0.23207 | 0.09996 | 0.94606 |
| V3 像素融合 | 0.05077 | 0.21090 | 0.07563 | 0.98315 |
| V3 + 帧安全层 | 0.04940 | 0.21338 | 0.07511 | 0.98226 |

研究口径必须保留：

- V3 像素融合的结构/checkpoint 在首次查看 5FR2 前固定，0.05077 是严格盲测结果；
- 帧安全阈值 0.5328558 只使用 5FR3 标签选择，但增加安全层的决定发生在检查 5FR2
  失效帧之后；0.04940 属于 post-hoc，必须在新区域复验后才能作为论文主结果；
- 旧尺度参数曾在 5FR2 前 82 帧搜索。后 82 帧中，强基线 AbsRel 0.05580，V3 安全模型
  0.05373；
- V3 在 1–5 m 提升，但 5FR2 的 0.2–1 m 从强基线 0.05534 退化到像素融合 0.06591，
  安全层为 0.06042；
- V1 在训练区域的候选是 in-sample prediction。论文版应使用 region-wise
  out-of-fold V1 候选，防止 stacking 训练分布偏乐观。

完整结果以 `RESULTS_V3.md` 为准。

## 4. 输入与 GT 边界

模型测试前向实际使用：

- 单帧 RGB；
- 由 RGB 得到的 DA3 深度/置信度；
- BIM 渲染深度、法向、边缘；
- 相机内参、camera-to-lidar 外参；
- 每帧 camera/lidar-to-BIM 位姿；
- 由上述输入得到的强锚点与冻结 V1 候选。

PCD GT 只用于训练监督、验证选择和测试指标。已做运行时检查：随机替换
`gt_depth/gt_valid/gt_weight/trust_target/trust_mask` 后，V1 和 V3 输出最大变化均为 0。

注意：当前 BIM 渲染位姿来自 `lidar_pose_local_to_bim_from_rosbag.txt`。如果该位姿的
BIM 对齐曾通过 PCD/ICP 标定，则 PCD 会通过“位姿标定”间接参与。论文应表述为
“given BIM-registered camera pose”，并增加独立视觉定位或位姿噪声实验。

`anchor_support` 是解析 BIM 校正场的空间支持，不是 PCD 的 `gt_support`。

## 5. 关键文件

- 当前配置：`configs/slabim_single_frame_r50_v3.yaml`
- 完整项目演化记录：`chatlog.md`
- V3 checkpoint：`outputs/slabim_single_frame_r50_v3/best.pt`
- V1 checkpoint：`outputs/slabim_single_frame_r50/best.pt`
- V3 严格盲测：`outputs/slabim_single_frame_r50_v3/evaluation/summary.json`
- V3 安全结果：`outputs/slabim_single_frame_r50_v3/evaluation_safe/summary.json`
- V3 验证：`outputs/slabim_single_frame_r50_v3/evaluation_val_safe/summary.json`
- 帧阈值选择：`outputs/slabim_single_frame_r50_v3/frame_gate_analysis.json`
- ±50 GT 密度：`outputs/gt_density_r10_vs_r50.json`
- 数据审计：`data/processed/slabim_504_r50/audit_r50.json`
- 迁移说明：`CLOUD_MIGRATION.md`

统一入口现可从 SLABIM 下载开始运行：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /workspace/SLABIM \
  --stages all
```

默认仅运行 `verify`，不会意外启动下载和训练。先用 `--dry-run --stages all` 检查完整命令。
下载、rosbag 位姿恢复、±50 GT、V1/V3、二维/三维评测和汇总均已在项目内，不再依赖
`/home/bgao491/HKUSTBIM` 脚本。

## 6. 环境和显存

本地验证环境：

- Python 3.9.25
- PyTorch 2.8.0+cu128
- NumPy 1.26.4
- OpenCV 4.11.0
- SciPy 1.13.1
- Open3D 0.19.0
- GPU NVIDIA RTX A400 4 GB

V3 为 1,605,410 个可训练参数，504×504、batch size 2 峰值 CUDA allocated 约
3.374 GiB。冻结候选训练 4 GB 可运行。接入 DA3 解码器特征建议至少 12 GB；24 GB 更适合
完整消融。

## 7. 下一步工作

优先级建议：

1. 新增至少一个完全未见区域，作为安全层和后续结构的最终测试集；
2. 按区域训练 out-of-fold V1，重新缓存训练区候选，再训练 V3；
3. 针对近距域偏移，学习 distance-conditioned calibration，但不得再用 5FR2 选参数；
4. 使用独立视觉 SLAM/BIM 定位位姿，进行 1/2/5/10 cm 与 0.5/1/2/5 度位姿噪声消融；
5. 云端显存允许时接入冻结 DA3 decoder feature，再逐步解冻最后 1–2 个 decoder block；
6. 三维重建评价补充点到面距离、Chamfer distance、completeness、accuracy 和构件级指标。

## 8. 新会话规则

- 先读本文件和 `RESULTS_V3.md`，不要从旧聊天记忆猜测；
- 不要使用 5FR2 继续选结构、阈值或超参数；
- 必须区分严格盲测 0.05077 与 post-hoc 安全层 0.04940；
- 任何新结果需记录配置、checkpoint、划分、随机种子和是否查看过测试集；
- 完成新实验后更新本文件的“已确认结果”和“下一步工作”。
