# SLABIM 端到端实验流水线

## 1. 目标和边界

本项目可以从官方 SLABIM 下载产物开始，完成数据校验、逐 LiDAR 帧位姿恢复、遮挡感知
GT 制备、DA3/BIM 缓存、V1/V3 训练、二维深度评测、三维重建评测和统一报告。

固定划分为：

- train：3F_Region2、3F_Region3、4F_Region2、4F_Region3；
- validation：5F_Region3；
- test：5F_Region2。

验证集可以选择 checkpoint 或门控阈值；测试集只能在方案固定后报告。PCD 和融合 GT
只能进入数据制备、监督和指标计算，不能作为模型前向输入。

## 2. 原始数据目录

```text
SLABIM/
├── BIM/<floor>/mesh/*.ply
├── calibration_files/
│   ├── cam_intrinsics.txt
│   └── cam_to_lidar.txt
└── sensor_data/<region>/
    ├── images/data/*.png
    ├── images/timestamps.txt
    ├── points/data/*.pcd
    ├── points/timestamps.txt
    ├── points/pose_frame_to_bim.txt
    └── rosbag/*.bag                  # 位姿恢复后可删除
```

下载器使用 Hugging Face 官方数据集
`BobH62/SLABIM`，支持 `.part` 文件续传、安全路径检查、只解压 rosbag 或排除 rosbag。
它不会删除已有 region 目录，也不会用空 staging 目录替换现有数据。

## 3. 位姿恢复

SLABIM bag 没有被本项目当作可直接读取的逐帧 GT trajectory。恢复链为：

```text
/livox/lidar raw scan (LiDAR-local)
  -- point-to-point ICP -->
official synchronized PCD (SLAM-global)
  = local_to_slam

constant map_to_BIM from pose_frame_to_bim
  @ smoothed local_to_slam
  = local_to_BIM
```

输出：

- `lidar_pose_local_to_slam.txt`；
- `lidar_pose_local_to_slam_smoothed.txt`；
- `lidar_pose_local_to_bim_from_rosbag.txt`；
- `lidar_pose_local_to_slam.diagnostics.npz`；
- 根目录 `pose_recovery_summary.json`。

正常历史六区域的 median fitness 约为 0.95–0.97，median ICP RMSE 约为
0.116–0.120 m。新下载区域如果明显偏离，应先检查 bag/PCD 时间同步和轨迹跳变，
不能直接继续训练。

## 4. GT 和模型输入

每张 RGB 同步到中心 LiDAR 索引。以中心前后各 50 个 PCD 为候选：

1. SLAM-global PCD 用对应 `local_to_slam` 逆变换回扫描时 LiDAR-local；
2. 经每帧 `local_to_BIM` 转到公共 BIM 坐标；
3. 转到中心 LiDAR 和相机坐标；
4. 每个扫描独立投影并 z-buffer；
5. 每像素只融合最前方且互相一致的深度簇，拒绝后方遮挡点；
6. 支持数和簇内 MAD 形成 `gt_weight`。

缓存 NPZ 包含 RGB 路径、DA3 深度/置信度、BIM 深度/法向/边缘、融合 GT，以及强锚点和
冻结 V1 候选。测试推理不会读取 `gt_*`、PCD 或 rosbag。

## 5. 运行方式

只验证原始数据：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /data/SLABIM \
  --stages verify
```

查看完整流水线而不执行：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /data/SLABIM \
  --stages all \
  --dry-run
```

执行完整流水线：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /data/SLABIM \
  --stages all
```

统一入口默认严格复现既有采样协议：train/validation 使用 stride 2，test 使用 stride 1，
对应 565/82/164 帧。只有开展新协议实验时才修改 `--train-val-stride` 或
`--test-stride`，并应写入新的配置、输出目录和报告。流水线会替换这些区域在 manifest
中的旧记录，避免改变 stride 后旧样本仍被训练集读取；磁盘上的旧 NPZ 不自动删除。

在已经下载并恢复位姿的数据上运行核心实验：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /data/SLABIM \
  --stages verify prepare audit anchors train-v1 eval-v1 \
           cache-candidates train-v3 eval-v3 reconstruct report
```

小规模烟雾测试可增加 `--max-frames 2`。该参数只截断制备和重建；训练仍会读取 manifest
中已有的全部样本，因此应使用空的临时 processed root 配置测试全新制备。

## 6. 输出

```text
data/processed/slabim_504_r50/
├── manifest.jsonl
├── metadata.json
├── audit.json
├── da3_cache/
└── samples/<region>/*.npz

outputs/
├── pipeline_state.json
├── dataset_verification.json
├── slabim_single_frame_r50/
│   ├── best.pt
│   ├── history.json
│   ├── evaluation_val/
│   └── evaluation_test/
├── slabim_single_frame_r50_v3/
│   ├── best.pt
│   ├── history.json
│   ├── evaluation_val/
│   ├── evaluation_test/
│   └── reconstruction_test/
│       ├── summary.json
│       └── *.ply
└── experiment_summary/
    ├── summary.json
    └── REPORT.md
```

二维 `summary.json` 包含原始 DA3、global scale、固定 scale+local BIM、coarse、refined，
并按 0.2–1、1–2、2–3、3–5 m 分段。

三维报告包含：

- accuracy：预测点到 GT 最近表面的距离；
- completeness：GT 点到预测表面的距离；
- Chamfer-L1：上述两个平均距离的均值；
- 5/10/20 cm precision、recall、F-score；
- 每区域和合并场景结果。

## 7. 断点恢复和注意事项

- ZIP 下载中断会保留 `.part`，再次运行续传；
- archive 成功解压后默认删除；
- 已有位姿和 checkpoint 默认复用；
- 每阶段状态写入 `outputs/pipeline_state.json`；
- `--force` 会重算位姿、缓存或训练，使用前确认确实需要；
- `--keep-rosbags` 可保留 bag，否则位姿恢复成功后删除；
- CUDA OOM 时训练脚本保存 `oom_state.pt` 和说明文件；
- 不要使用 test 指标选择结构、checkpoint 或阈值；
- 3D `prediction-mask=all` 才反映预测表面覆盖，但会受稀疏 LiDAR GT 覆盖影响；
- 3D `prediction-mask=gt` 是同像素几何诊断，不能作为完整重建主指标。

## 8. 尚未自动化的论文扩展

当前统一入口覆盖项目主方法及其解析基线，但下列工作仍应作为独立论文实验实现并注册新
配置，不能暗中混入固定主结果：

- region-wise out-of-fold V1 候选；
- 独立视觉 SLAM/BIM localization 位姿；
- 位姿平移/旋转噪声曲线；
- COLMAP/SfM-MVS 与 DA3 多视图的统一可见域对比；
- point-to-plane、法向和 BIM 构件级重建指标；
- 新的、从未用于调参的最终测试区域。
