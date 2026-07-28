# BIM-PriorDA3 对话与项目演化日志

> 本文件是面向后续 LLM/研究人员的结构化对话记录，不是逐字聊天转录。
> 它记录用户目标如何演化、已经执行的实验、失败尝试、关键结果、数据边界以及下一步工作。
> 新会话应先完整阅读本文件、`CLOUD_HANDOFF.md` 和 `RESULTS_V3.md`，再开始任何实验。

## 0. 新会话必须遵守的规则

1. 不要再次使用 `5F_Region2` 选择网络结构、损失、阈值或超参数。该区域已经被多次查看。
2. 必须区分：
   - 严格盲测 V3 像素融合：5FR2 AbsRel `0.05077`；
   - post-hoc 帧安全层：5FR2 AbsRel `0.04940`。
3. 不要声称推理完全只需要“BIM + Image”。准确说法是：
   - RGB；
   - RGB 生成的 DA3 深度/置信度；
   - BIM 渲染先验；
   - 相机内参、外参及 BIM 配准位姿；
   - PCD 仅作为监督/评测 GT。
4. 当前 BIM 位姿可能通过 PCD/ICP 标定，因此论文中应使用
   “given BIM-registered camera pose”，并补充独立视觉定位或位姿噪声实验。
5. `anchor_support` 是 BIM 局部校正场支持度，不是 PCD 的 `gt_support`。
6. 任何新实验都应保存独立配置、checkpoint、history、evaluation，并更新本日志。

---

## 1. 用户总体目标

用户的最终目标是面向土木工程室内场景，利用单帧 RGB、Depth Anything 3 和 BIM 先验，
提高度量深度估计与三维重建精度，并形成具有期刊工作量和学术贡献的方法。

核心应用设定：

- 单帧深度推理；
- BIM 提供几何先验；
- LiDAR/PCD 提供训练与评测 GT；
- 最终用于全场景三维重建；
- 远距离区域可以在后续靠近采集，因此主要关注约 5 m 内；
- 目标不是用 PCD 直接修正测试深度，而是训练可部署的 RGB+BIM 模型。

---

## 2. 对话历史概览

### 2.1 GLB 导出与相机内参

最初用户要求修复 `segment-anything/demo.py` 无法导出 GLB 的问题，并解释导出的 NPZ。

遇到错误：

```text
AssertionError: Export to GLB: prediction.intrinsics is required but not available
```

用户指定默认相机内参：

```text
800 0   800
0   800 800
0   0   1
```

讨论结论：

- GLB 导出需要相机内参，才能把像素深度反投影为三维点；
- NPZ 通常保存深度、置信度、相机参数等 NumPy 数组；
- 可使用 NumPy、Open3D、Matplotlib 将深度或点云可视化。

这一阶段促成了后续所有深度到三维重建流程中的内参显式管理。

### 2.2 5F_Region2 单帧深度评测

用户要求在 HKUST 项目中使用 Depth Anything 对：

```text
C:\Users\bgao491\pythonProject\5F_Region2
```

进行单帧深度评测。

讨论和实现包括：

- 使用 PCD 投影生成稀疏 GT depth；
- 仅在有效 PCD 投影点上计算指标；
- 解释 AbsRel、RMSE、MAE、delta1/delta2/delta3；
- 按距离统计 `0–1 m`、`1–2 m`、`2–3 m`、`3–5 m`；
- 将相邻 PCD 融合后投影为更密集 GT；
- 使用 z-buffer 和前方一致深度簇处理遮挡。

用户发现初始 GT depth 只有上半部分。讨论认为原因包括：

- LiDAR 垂直视场与相机视场不同；
- 传感器安装姿态和遮挡；
- 相邻帧融合不足；
- 位姿或外参误差。

### 2.3 位姿恢复与核查

用户指出数据集中存在逐帧位姿，并要求核查：

```text
5F_Region2/points
```

后续发现官方 `pose_frame_to_bim.txt` 更像重复的固定 SLAM-to-BIM 变换，而不是运动轨迹。

因此从 ROS bag 中恢复逐帧 LiDAR 位姿：

核心脚本：

```text
/home/bgao491/HKUSTBIM/extract_lidar_poses_from_rosbag.py
```

流程：

```text
/livox/lidar 原始扫描
→ 与同时间戳官方SLAM-global PCD做ICP
→ LiDAR-local到SLAM-global轨迹
→ Savitzky–Golay平滑
→ 组合官方SLAM-to-BIM固定变换
→ LiDAR-local到BIM逐帧位姿
```

输出：

```text
lidar_pose_local_to_slam.txt
lidar_pose_local_to_slam_smoothed.txt
lidar_pose_local_to_bim_from_rosbag.txt
lidar_pose_local_to_slam.diagnostics.npz
```

位姿恢复说明：

```text
/home/bgao491/SLABIM/POSE_RECOVERY_README.md
```

重要边界：

- 当前脚本不是直接读取 `/tf` 或 `/odom`；
- 它读取 `/livox/lidar` 并通过官方 PCD ICP 恢复轨迹；
- 因此当前 BIM 配准位姿间接使用了 PCD 信息。

### 2.4 BIM 先验增强

用户指出数据中存在 BIM PLY：

```text
5F_Region2/mesh
```

随后实现/讨论了多种增强：

- 全局尺度矫正；
- `adaptive_bim`；
- `temporal_safe_bim`；
- 平滑局部校正场；
- BIM 深度渲染；
- BIM/DA3 一致性筛选；
- 帧级与像素级 BIM 可信度；
- 基于 MAD 的帧级异常判定；
- 近距离优先、远距离拒绝。

“平滑局部校正场”定义：

- 先计算 BIM 与尺度恢复 DA3 的局部 log-depth 残差；
- 仅在一致、非 BIM 边缘区域采样；
- 用大尺度 Gaussian 平滑形成低频校正场；
- 避免直接复制 BIM 的锐利错误或动态遮挡。

旧强基线的固定参数：

```yaml
scale_quantile: 0.45
consistency_log_threshold: 0.10
smoothing_sigma: 64.0
local_correction_alpha: 1.25
```

实现：

```text
src/bim_priorda3/baselines.py
```

### 2.5 多区域下载与交叉区域评测

用户要求下载 SLABIM 其他区域，解压后删除 ZIP 和 ROS bag，以节省空间。

最终使用区域：

- `3F_Region2`
- `3F_Region3`
- `4F_Region2`
- `4F_Region3`
- `5F_Region2`
- `5F_Region3`

相关下载脚本位于：

```text
/home/bgao491/HKUSTBIM/download_slabim_regions.py
/home/bgao491/HKUSTBIM/download_slabim_rosbag.py
```

多区域实验暴露了：

- 5FR2 的 BIM 对齐与强尺度基线异常好；
- 各区域单独搜索超参数只能有限改善；
- 性能对固定参数并不高度敏感；
- 位姿、BIM完整性、局部遮挡和区域域差异更关键。

### 2.6 学术论文与方法构思

用户要求查阅土木/CV领域研究、设计创新点、补充实验并撰写论文初稿。

核心论文方向逐步形成：

- 单帧 BIM-conditioned metric depth refinement；
- BIM 不是绝对真值，而是带不确定性的结构先验；
- 学习帧级和像素级 BIM 可信度；
- 鲁棒局部度量对齐；
- 对失效 BIM 或位姿异常进行安全回退；
- 将深度评价扩展到三维重建评价；
- 对位姿噪声、BIM缺失、动态遮挡和距离范围做消融。

历史论文草稿位于：

```text
/home/bgao491/HKUSTBIM/paper/BIMGuard_paper_draft_zh.md
```

### 2.7 DA3 多视图与三维重建

用户询问 DA3-BASE、多视图版本以及与单帧+BIM的比较。

进行过：

- DA3-BASE 多视图；
- DA3 多视图+BIM先验；
- 与传统 SfM-MVS 的概念比较；
- 原始 PCD 重建与预测深度重建 PLY 导出；
- 全场景重建精度讨论。

相关历史代码主要在：

```text
/home/bgao491/HKUSTBIM/
```

包括：

```text
evaluate_da3base_multiview_bim.py
evaluate_da3_multiview_reconstruction.py
evaluate_adaptive_bim_reconstruction.py
evaluate_colmap_sfm_mvs.py
evaluate_colmap_known_pose_mvs.py
visualize_rgb_fused_pcd_depth.py
```

### 2.8 参考 Prior Depth Anything，创建独立项目

用户要求参考论文 “Prior Depth Anything”，创建一个真实深度学习工程，并先实现单帧模型。

新项目：

```text
/home/bgao491/BIM-PriorDA3
```

工程包含：

```text
configs/
scripts/
src/bim_priorda3/
tests/
data/processed/
outputs/
```

初始设计 V1：

- 冻结 DA3，离线缓存初始深度；
- BIM 渲染深度、法向和边缘；
- 学习像素级和帧级 BIM 可信度；
- 可微鲁棒局部仿射对齐；
- 轻量 U-Net 输出 log-depth residual 和不确定性；
- LiDAR PCD 仅作为训练/评测监督。

---

## 3. ±50 帧 GT 升级

用户认为前后各 10 帧融合仍太稀疏，要求扩展到前后各 50 帧。

当前 GT：

- 中心帧前后各 50 个 PCD；
- 最多 101 个扫描；
- 每个扫描单独投影、z-buffer；
- 仅融合最前方深度一致簇；
- `gt_support` 与簇内 MAD 形成 GT 权重；
- 深度范围 `0.2–5 m`。

数据配置：

```text
configs/slabim_single_frame_r50.yaml
configs/slabim_single_frame_r50_v3.yaml
```

数据根：

```text
data/processed/slabim_504_r50
```

划分：

| split | regions | samples |
|---|---|---:|
| train | 3FR2, 3FR3, 4FR2, 4FR3 | 565 |
| val | 5FR3 | 82 |
| test | 5FR2 | 164 |

无区域泄漏。

GT 密度相对 ±10 帧：

| 区域 | 相对增加 |
|---|---:|
| 3FR2 | 35.43% |
| 3FR3 | 32.09% |
| 4FR2 | 25.50% |
| 4FR3 | 23.35% |
| 5FR2 | 25.91% |
| 5FR3 | 29.49% |

六区域有效像素总体增加约 29.13%。

相关文件：

```text
outputs/gt_density_r10_vs_r50.json
data/processed/slabim_504_r50/audit_r50.json
RESULTS_R50.md
```

---

## 4. V1 完整训练结果

V1 训练设置：

- 训练 565 帧；
- 验证 82 帧；
- 504×504 验证/测试；
- 384×384 随机训练裁剪；
- batch size 2；
- gradient accumulation 2；
- 约 145 万可训练参数；
- 最优 checkpoint 为第 23 个 epoch；
- 第 31 个 epoch 早停；
- 峰值 CUDA allocated 约 1.73 GiB。

V1 最佳验证：

```text
5FR3 AbsRel = 0.10854
```

5FR2 测试：

| 方法 | AbsRel | RMSE m | MAE m | delta1 |
|---|---:|---:|---:|---:|
| 原始 DA3 | 0.24683 | 0.45560 | 0.34806 | 0.63132 |
| 全局尺度 | 0.06726 | 0.24093 | 0.10410 | 0.97402 |
| 强尺度+局部BIM | 0.05260 | 0.22822 | 0.08253 | 0.97441 |
| V1粗对齐 | 0.10115 | 0.34028 | 0.16752 | 0.84864 |
| V1最终细化 | 0.07739 | 0.23207 | 0.09996 | 0.94606 |

结论：

- V1 相对原始 DA3 显著提升；
- 但不及旧解析强基线；
- 主要问题集中在 1 m 内和少数灾难帧；
- 例如 `000143`、`000101`、`000030`、`000028`。

根因审查：

1. V1 最终输出从原始 DA3 开始，而强尺度结果仅作为间接特征；
2. 网络需要重新学习已经被解析方法解决的尺度；
3. 可信度标签比较未尺度恢复 DA3 与 BIM，标签主要反映全局尺度差；
4. 不确定性项在部分阶段对总损失影响过强；
5. 跨区域域差异导致少数自由残差失控。

---

## 5. V2/V2.1 负结果

### V2：强锚点有界残差

改进：

- 全图预计算强锚点，避免随机裁剪破坏全局尺度；
- 最终输出从强锚点开始；
- 最大 log-residual 受限；
- 增加 preservation、degradation 和 update-gate 监督；
- 可信度标签改为比较“尺度恢复 DA3”与 BIM。

结果：

- 初始化与强基线逐像素一致；
- 但平均有效 log-residual 约 `9.6e-6`；
- 网络几乎不修改锚点；
- 验证约停留在 `0.12666`。

结论：安全约束与更新门共同造成梯度过弱。

### V2.1：残差教师与帧级残差头

改进：

- 对未门控残差提案直接做 GT 教师监督；
- 加入帧级全局残差头；
- 放松 preservation/degradation；
- 提高残差学习速度。

结果：

- 验证最好约 `0.12455`；
- 之后训练损失下降但验证回升；
- 仍不及 V1 的 `0.10854`；
- 说明自由残差即使从强锚点开始，仍会跨区域过拟合。

这些负结果应作为论文消融，而不是删除。

---

## 6. 当前最佳 V3：学习式候选融合

V3 不再从单一深度生成自由残差，而是在两个候选之间学习凸组合：

```text
候选A = 全局尺度 + 平滑局部BIM强锚点
候选B = 冻结V1学习深度
```

融合：

```text
log(D_final) = (1-g) log(D_anchor) + g log(D_candidate)
```

其中 `g` 为学习到的逐像素候选门控。

门控输入：

- RGB；
- 强锚点深度；
- V1候选深度；
- 两候选差异；
- V1不确定性；
- V1像素/帧可信度；
- BIM深度、有效性和边缘；
- 强锚点支持度。

监督：

```text
g* = sigmoid((error_anchor - error_candidate - margin) / temperature)
```

损失：

- 深度损失；
- 梯度损失；
- 像素候选门控 BCE；
- 帧级候选门控 BCE；
- 强锚点保持；
- 相对强锚点防退化。

优点：

- 输出始终位于两个候选之间；
- 不会创造超出两个候选的新灾难深度；
- 利用 V1 与解析强基线的互补性。

候选理论上限：

```text
5FR2 strong anchor AbsRel = 0.05260
5FR2 V1 candidate AbsRel = 0.07739
逐像素oracle选择 AbsRel = 0.03536
V1在约42.2%的5FR2有效像素上更准确
```

V3 训练：

- 约 1,605,410 个参数；
- 504×504；
- batch size 2；
- 峰值 CUDA allocated 约 3.374 GiB；
- 高学习率阶段第 3 个 epoch 获得最佳 5FR3 checkpoint；
- 后续高学习率与低学习率微调均未超过第 3 个 epoch。

验证结果：

| 方法 | 5FR3 AbsRel |
|---|---:|
| 原始 DA3 | 0.27727 |
| 强基线 | 0.12672 |
| V1 | 0.10854 |
| V3像素融合 | 0.08935 |

---

## 7. V3 最终测试与实验诚信边界

5FR2 相同 30,807,322 个有效 GT 像素：

| 方法 | AbsRel | RMSE m | MAE m | delta1 |
|---|---:|---:|---:|---:|
| 原始 DA3 | 0.24683 | 0.45560 | 0.34806 | 0.63132 |
| 全局尺度 | 0.06726 | 0.24093 | 0.10410 | 0.97402 |
| 强尺度+局部BIM | 0.05260 | 0.22822 | 0.08253 | 0.97441 |
| V1 | 0.07739 | 0.23207 | 0.09996 | 0.94606 |
| V3像素融合 | 0.05077 | 0.21090 | 0.07563 | 0.98315 |
| V3+帧安全层 | 0.04940 | 0.21338 | 0.07511 | 0.98226 |

严格盲测 V3 相对强基线：

- AbsRel 降低约 3.49%；
- RMSE 降低约 7.59%；
- MAE 降低约 8.36%。

帧安全层：

- 验证集选择阈值 `0.5328558087348938`；
- 帧可信度低于阈值时整帧回退强锚点；
- 测试 AbsRel `0.04940`；
- 但“增加安全层”这一决定是在查看 5FR2 逐帧失效之后作出的；
- 因此该结果属于 post-hoc，不能直接作为论文主结果。

旧强基线参数曾在 5FR2 前 82 帧上搜索。后 82 帧：

```text
strong baseline AbsRel = 0.05580
V3 safe AbsRel         = 0.05373
```

分距离：

| 距离 | 强基线 | V3像素融合 | V3安全 |
|---|---:|---:|---:|
| 0.2–1 m | 0.05534 | 0.06591 | 0.06042 |
| 1–2 m | 0.05096 | 0.04323 | 0.04352 |
| 2–3 m | 0.04774 | 0.04115 | 0.04165 |
| 3–5 m | 0.06135 | 0.05747 | 0.05792 |

剩余问题：

- 5FR2 的近距离强锚点异常准确；
- 训练/验证区域没有同样分布；
- V3 在 0.2–1 m 过度选择 V1；
- 不得再使用 5FR2 调近距离门控；
- 需要新区域和 out-of-fold 候选验证。

完整结果：

```text
RESULTS_R50.md
RESULTS_V3.md
```

---

## 8. 推理输入与 GT 泄漏核查

用户专门询问测试时是否只使用 BIM+Image、GT 是否仅来自 PCD。

实际前向输入：

```text
RGB
DA3 depth/confidence（由RGB得到）
BIM rendered depth/normals/edge
camera intrinsics
camera-to-lidar extrinsic
camera/lidar-to-BIM pose
strong anchor（由DA3+BIM得到）
V1 candidate（由RGB+DA3+BIM得到）
```

PCD 的直接用途：

- 训练：监督；
- 验证：checkpoint与阈值；
- 测试：GT depth、有效掩码和指标。

已做运行时测试：

```text
随机替换 gt_depth
随机替换 gt_valid
随机替换 gt_weight
随机替换 trust_target
随机替换 trust_mask
```

结果：

```text
V3最大输出变化 = 0.0
V1最大输出变化 = 0.0
```

因此模型前向没有直接读取 PCD GT。

但位姿边界必须诚实说明：

- 当前 BIM 渲染使用从 rosbag+PCD ICP 恢复的逐帧位姿；
- PCD 可能通过位姿标定间接参与；
- 后续需要用视觉 SLAM/BIM localization 替换，并做位姿噪声消融。

---

## 9. 云端迁移准备

用户准备迁移到云服务器，并希望新 LLM 不忘记上下文。

已经生成：

```text
CLOUD_HANDOFF.md
CLOUD_MIGRATION.md
MIGRATION_CHECKSUMS.sha256
scripts/verify_cloud_setup.py
```

代码已支持：

```bash
export BIM_PRIORDA3_SLABIM_ROOT=/workspace/SLABIM
```

当 manifest 中旧绝对路径失效时，会自动重定位：

```text
sample → 当前项目 data/processed/.../samples/
image  → $BIM_PRIORDA3_SLABIM_ROOT/sensor_data/.../images/data/
```

最小迁移：

- 整个 `BIM-PriorDA3`：约 3.8 GB；
- 六区域 RGB：约 4.3 GB；
- BIM+标定：约 477 MB；
- DA3代码：约 48 MB。

若重新生成 GT，再复制约 12.8 GB PCD 与位姿文件。ROS bag 不必迁移。

云端新会话推荐首句：

```text
请先完整阅读 /workspace/BIM-PriorDA3/chatlog.md、
CLOUD_HANDOFF.md、RESULTS_V3.md 和
configs/slabim_single_frame_r50_v3.yaml。
不要重新使用5FR2调参；从日志中的“下一步工作”继续。
```

---

## 10. 当前关键文件索引

### 项目与说明

```text
README.md
chatlog.md
CLOUD_HANDOFF.md
CLOUD_MIGRATION.md
RESULTS_R50.md
RESULTS_V3.md
```

### 配置

```text
configs/slabim_single_frame.yaml
configs/slabim_single_frame_r50.yaml
configs/slabim_single_frame_r50_v2.yaml
configs/slabim_single_frame_r50_v21.yaml
configs/slabim_single_frame_r50_v3.yaml
```

### 数据与模型代码

```text
src/bim_priorda3/data/preparation.py
src/bim_priorda3/data/geometry.py
src/bim_priorda3/data/dataset.py
src/bim_priorda3/baselines.py
src/bim_priorda3/models/system.py
src/bim_priorda3/models/trust.py
src/bim_priorda3/models/refiner.py
src/bim_priorda3/models/alignment.py
src/bim_priorda3/losses.py
```

### 训练、评测与迁移

```text
scripts/prepare_dataset.py
scripts/prepare_strong_anchors.py
scripts/cache_candidate_predictions.py
scripts/train.py
scripts/evaluate.py
scripts/infer.py
scripts/analyze_frame_gate.py
scripts/verify_cloud_setup.py
```

### 重要模型与结果

```text
outputs/slabim_single_frame_r50/best.pt
outputs/slabim_single_frame_r50/history.json
outputs/slabim_single_frame_r50/evaluation/summary.json

outputs/slabim_single_frame_r50_v3/best.pt
outputs/slabim_single_frame_r50_v3/history.json
outputs/slabim_single_frame_r50_v3/evaluation/summary.json
outputs/slabim_single_frame_r50_v3/evaluation_safe/summary.json
outputs/slabim_single_frame_r50_v3/evaluation_val_safe/summary.json
outputs/slabim_single_frame_r50_v3/frame_gate_analysis.json
```

### ROS bag 与历史评测

```text
/home/bgao491/HKUSTBIM/extract_lidar_poses_from_rosbag.py
/home/bgao491/HKUSTBIM/download_slabim_rosbag.py
/home/bgao491/HKUSTBIM/download_slabim_regions.py
/home/bgao491/SLABIM/POSE_RECOVERY_README.md
```

---

## 11. 环境与资源

本地环境：

```text
Python 3.9.25
PyTorch 2.8.0+cu128
CUDA runtime 12.8
NumPy 1.26.4
OpenCV 4.11.0
SciPy 1.13.1
Open3D 0.19.0
GPU NVIDIA RTX A400 4 GB
```

显存：

- V1 峰值 allocated 约 1.73 GiB；
- V3 峰值 allocated 约 3.374 GiB；
- 4 GB 可训练冻结候选 V3；
- 接入 DA3 decoder feature 建议至少 12 GB；
- 完整消融建议 24 GB。

---

## 12. 下一步工作建议

### 最高优先级

1. 获取一个或多个新的、完全未见区域；
2. 将新区域分成 validation 与 final test；
3. 禁止继续使用 5FR2 调参；
4. 在新 final test 上重新验证帧安全层。

### 提高方法严谨性

1. 对训练区域做 region-wise out-of-fold V1：
   - 留出一个训练区域；
   - 用其余区域训练 V1；
   - 为被留出区域生成候选；
   - 四区域轮换；
   - 使用 OOF 候选训练 V3。
2. 当前 V1 训练区候选是 in-sample，正式论文必须讨论或修复。
3. 对门控做 calibration：
   - Expected Calibration Error；
   - reliability diagram；
   - Brier score；
   - selective risk/coverage。

### 近距离优化

1. 在新验证区学习 distance-conditioned gate；
2. 增加近距离样本重采样；
3. 使用 anchor/candidate disagreement 和几何边缘共同约束；
4. 不允许用 5FR2 的 0–1 m 结果直接搜索阈值。

### 位姿实验

1. 使用视觉 SLAM 或 BIM localization 的相机位姿；
2. 增加平移噪声：
   - 1 cm
   - 2 cm
   - 5 cm
   - 10 cm
3. 增加旋转噪声：
   - 0.5°
   - 1°
   - 2°
   - 5°
4. 评价深度与三维重建性能随位姿误差的退化曲线。

### 云端网络升级

显存允许后：

1. 将 DA3 decoder feature 作为候选门控输入；
2. 首先冻结 DA3；
3. 然后只解冻最后 1–2 个 decoder block；
4. 使用低学习率、分层 learning rate；
5. 与当前冻结候选 V3 做严格相同划分消融。

### 三维重建评价

除单帧深度指标外，补充：

- point-to-point distance；
- point-to-plane distance；
- Chamfer distance；
- accuracy；
- completeness；
- F-score；
- 平面法向误差；
- 构件级完整率；
- BIM构件边界偏差；
- 不同距离范围的重建误差；
- 与 SfM-MVS、DA3多视图、原始PCD重建比较。

---

## 13. 当前状态一句话总结

项目已经从“DA3+BIM解析尺度增强”发展为“解析强锚点与学习候选的可信凸融合”。
严格盲测 V3 已小幅超过旧强基线，但近距离跨区域校准、训练候选 OOF、独立视觉位姿和新的
未见测试区域仍是形成可信期刊贡献前必须完成的工作。

---

## 14. 2026-07-27：SLABIM 全流程代码归入项目

用户要求继续优化项目，并把关键代码放入 BIM-PriorDA3，使项目可以直接基于下载好的
SLABIM 完成全部主实验和评测。

本次完成：

1. 新增 `src/bim_priorda3/data/slabim.py` 与 `scripts/download_slabim.py`：
   - Hugging Face SLABIM 可续传下载；
   - ZIP 路径安全检查；
   - core、rosbag-only、all 三种解压模式；
   - 解压成功后删除 archive；
   - 合并到现有 region，不用空目录覆盖已有数据。
2. 新增 `src/bim_priorda3/data/pose_recovery.py` 与
   `scripts/recover_slabim_poses.py`：
   - 解析 `/livox/lidar`；
   - 兼容 SLABIM 非标准 rosbag magic header；
   - raw local scan 到官方 SLAM-global PCD 的逐帧 ICP；
   - Savitzky–Golay 位姿平滑；
   - 与常量 map-to-BIM 变换复合；
   - 输出 fitness/RMSE diagnostics；
   - 成功后可删除 rosbag。
3. 新增 `scripts/verify_slabim_dataset.py`，检查 RGB/PCD/时间戳/位姿逐行对应。
4. 新增 `src/bim_priorda3/reconstruction.py` 与
   `scripts/evaluate_reconstruction.py`：
   - 在 BIM 坐标系融合不同深度方法；
   - 不进行评测时 ICP；
   - 计算 accuracy、completeness、Chamfer-L1、阈值 precision/recall/F-score；
   - 导出 PLY。
5. 新增 `scripts/run_slabim_experiments.py`：
   - download → poses → verify → prepare → audit → anchors →
     train/eval V1 → candidate cache → train/eval V3 → reconstruction → report；
   - 支持阶段选择、dry-run、已有 checkpoint 复用和状态记录。
6. 新增 `scripts/summarize_experiments.py` 和
   `docs/EXPERIMENT_PIPELINE.md`。
7. 配置不再依赖 `/home/bgao491/HKUSTBIM` DA3 cache 绝对路径；
   SLABIM 默认使用项目同级 `../SLABIM`，仍支持
   `BIM_PRIORDA3_SLABIM_ROOT`。
8. 修复 `compare_gt_density.py` 在迁移后无法对 stale absolute manifest path
   调用 `relative_to()` 的问题。

### 14.1 环境安装和路径约定

项目默认把与 `BIM-PriorDA3` 同级的 `../SLABIM` 作为数据根目录。迁移到其他目录时，
推荐显式传入 `--slabim-root`；也可以设置：

```bash
export BIM_PRIORDA3_SLABIM_ROOT=/workspace/SLABIM
```

完整安装：

```bash
cd /workspace/BIM-PriorDA3
python -m pip install -e '.[slabim,da3,dev]'
```

- `slabim` extra 安装 `rosbags`，只在重新解析 bag 时必需；
- `da3` extra 提供 Depth Anything 3；
- 已有 processed NPZ 和 checkpoint 时，可以暂不安装 DA3；
- 云端 PyTorch 应根据驱动/CUDA 单独安装，不能机械复制本机 wheel。

### 14.2 三种使用起点

#### A. 已有完整 SLABIM、恢复位姿和 PCD

先做只读式完整性检查：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /workspace/SLABIM \
  --stages verify
```

然后从样本制备开始：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /workspace/SLABIM \
  --stages prepare audit anchors train-v1 eval-v1 \
           cache-candidates train-v3 eval-v3 reconstruct report
```

#### B. 已下载 RGB/PCD/BIM，但缺少恢复后的逐帧位姿

只补充 rosbag，避免重复长期保留 archive：

```bash
python scripts/download_slabim.py \
  --root /workspace/SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3 \
  --rosbag-only

python scripts/recover_slabim_poses.py \
  --root /workspace/SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3 \
  --delete-rosbags
```

位姿恢复成功后保留：

```text
points/lidar_pose_local_to_slam.txt
points/lidar_pose_local_to_slam_smoothed.txt
points/lidar_pose_local_to_bim_from_rosbag.txt
points/lidar_pose_local_to_slam.diagnostics.npz
SLABIM/pose_recovery_summary.json
```

bag 默认只有在恢复成功后才删除。已存在有效位姿时，完整流水线不会重新下载 bag。

#### C. 从零下载并完成所有主实验

先 dry-run 检查将要执行的命令：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /workspace/SLABIM \
  --stages all \
  --dry-run
```

确认后执行：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /workspace/SLABIM \
  --stages all
```

默认阶段：

```text
download
-> poses
-> verify
-> prepare
-> audit
-> anchors
-> train-v1
-> eval-v1
-> cache-candidates
-> train-v3
-> eval-v3
-> reconstruct
-> report
```

下载写入 `.part` 文件并支持续传。ZIP 成功解压后删除；已有数据按文件树合并，不会用 staging
目录整体覆盖用户现有 region。

### 14.3 固定数据制备协议

统一入口不是对六区域使用相同 stride，而是复现原实验：

```text
train:
  3F_Region2  174
  3F_Region3  174
  4F_Region2  156
  4F_Region3   61
  total       565

validation:
  5F_Region3   82

test:
  5F_Region2  164
```

- train/validation 使用 `stride=2`；
- test 使用 `stride=1`；
- GT 使用中心前后各 50 个 LiDAR 扫描，最多 101 帧；
- 每个扫描先独立 z-buffer，再只融合最前方一致深度簇；
- 流水线通过 `--replace-regions-in-manifest` 替换这些区域的旧 manifest 记录；
- 旧 NPZ 不自动删除，但只要不在 manifest 中就不会被 Dataset 读取。

只有定义新协议时才使用：

```text
--train-val-stride N
--test-stride N
```

修改采样协议后应更换 `processed_root`、experiment name 和 output directory，不能与固定结果
混用。

### 14.4 位姿方法和质量检查

rosbag 中使用的主题是 `/livox/lidar`。本项目没有把 bag 当作直接可读的 GT trajectory，
而是：

```text
raw Livox scan (LiDAR-local)
  -> 与同时间 official PCD 做 point-to-point ICP
  -> local-to-SLAM
  -> Savitzky–Golay 平滑
  -> constant map-to-BIM @ local-to-SLAM
  -> local-to-BIM
```

`pose_frame_to_bim.txt` 必须是逐行相同的常量 map-to-BIM 变换；若平移或旋转随帧变化，
脚本会停止，避免错误复合。

历史六区域的合理量级：

- median ICP fitness 约 0.95–0.97；
- median ICP RMSE 约 0.116–0.120 m；
- 5FR2 为 fitness 0.94852、RMSE 0.12033 m、P95 RMSE 0.13964 m。

新区域明显偏离这些数值时，不应直接继续训练，应检查时间戳、bag topic、PCD 坐标和轨迹
跳变。该位姿是 PCD 配准恢复结果，不应在论文中称为传感器提供的 GT pose。

### 14.5 单帧深度评测

独立执行：

```bash
python scripts/evaluate.py \
  --config configs/slabim_single_frame_r50_v3.yaml \
  --checkpoint outputs/slabim_single_frame_r50_v3/best.pt \
  --split test \
  --output outputs/slabim_single_frame_r50_v3/evaluation_test
```

统一报告：

```bash
python scripts/summarize_experiments.py
```

二维评测同时输出：

- `base`：原始 DA3；
- `global_scale`：仅全局尺度；
- `previous_scale_local`：固定 scale + 平滑局部 BIM；
- `coarse`：网络粗结果；
- `refined`：最终结果；
- 0.2–1、1–2、2–3、3–5 m 分距离指标；
- `per_frame.csv`，用于检查灾难帧。

本次统一目录中的测试结果：

```text
V1 refined AbsRel  0.07739
V3 refined AbsRel  0.04940
raw DA3 AbsRel      0.24683
strong BIM AbsRel   0.05260
```

### 14.6 三维重建评测与 PLY

独立执行：

```bash
python scripts/evaluate_reconstruction.py \
  --config configs/slabim_single_frame_r50_v3.yaml \
  --checkpoint outputs/slabim_single_frame_r50_v3/best.pt \
  --split test \
  --save-clouds
```

默认设置：

```text
depth range       0.2–5.0 m
pixel stride      4
voxel size        0.05 m
prediction mask   all
alignment         recovered calibrated pose only; no evaluation-time ICP
```

`prediction-mask=all` 用于主重建实验，会评价所有预测表面，但受稀疏 LiDAR 参考覆盖影响。
`prediction-mask=gt` 只在 GT 有效像素生成预测点，是坐标链/同像素诊断，不能冒充完整场景
重建性能。

输出位于：

```text
outputs/slabim_single_frame_r50_v3/reconstruction_test/
├── summary.json
└── 5F_Region2/
    ├── base.ply
    ├── previous_scale_local.ply
    ├── refined.ply
    └── gt_fused.ply
```

指标含义：

- accuracy：每个预测点到 GT 最近点的距离；
- completeness：每个 GT 点到预测点云最近点的距离；
- Chamfer-L1：accuracy mean 与 completeness mean 的平均；
- threshold precision：预测点落在 GT 阈值内的比例；
- threshold recall：GT 点被预测点在阈值内覆盖的比例；
- F-score：precision 和 recall 的调和平均。

单区域输出不再重复保存内容相同的 `all_*.ply`，避免额外约 55 MB 占用。需要重新生成时
重新执行上述重建命令即可。

### 14.7 断点、强制重算和常见失败

- `outputs/pipeline_state.json` 记录每个阶段的命令、开始/完成时间和状态；
- 已有 `best.pt` 默认复用；
- `--force` 才会重算已有位姿、样本、候选或训练，使用前检查输出目录；
- `--keep-rosbags` 可阻止位姿恢复后删除 bag；
- CUDA OOM 时训练脚本保存 `oom_state.pt` 和 `OOM_README.txt`；
- manifest 内旧绝对路径失效时，Dataset 根据 region 和文件名重定位；
- `compare_gt_density.py` 同样按稳定的 region/sample filename 比较，不再依赖旧绝对路径；
- 若 `verify` 失败，先修复缺失 calibration、BIM、RGB/PCD 数量、时间戳或位姿，不要跳过；
- 当前 FastSAM 环境没有 pytest；安装 `.[dev]` 后运行 `python -m pytest -q`。

详细说明仍以 `docs/EXPERIMENT_PIPELINE.md` 为准；本节用于新会话或其他 LLM 在只阅读
`chatlog.md` 时也能恢复完整执行方法。

验证：

- 六区域中的 5FR2 原始数据检查通过：164 RGB、6965 PCD、两套 6965×8 位姿均与
  timestamp 对齐；
- 完整流水线 dry-run 命令链通过；
- 三维重建一帧 smoke test 成功，随后完成 5FR2 全 164 帧默认稠密重建：
  原始 DA3、强尺度+局部 BIM、V3 的 Chamfer-L1 分别为
  0.20232、0.06059、0.05634 m；V3 相对强基线降低 7.01%；
- 5 cm F-score 分别为 0.04378、0.60975、0.62839；
- 正式重建使用 0.2–5 m、pixel stride 4、5 cm voxel、全部预测像素，
  不进行 evaluation-time ICP；
- `compileall` 通过；
- 当前 FastSAM 环境未安装 pytest，未在本次会话执行 pytest suite。

论文口径不变：5FR2 已被历史工作查看，不能用于后续超参数选择；统一流水线虽然可以复现
固定测试结果，但新增结构必须在新验证/测试区域上完成选择与最终检验。
