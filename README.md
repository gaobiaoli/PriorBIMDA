# BIM-PriorDA3

面向室内土木场景单帧深度估计的 BIM 先验学习式细化项目。方法参考
[Prior Depth Anything](https://arxiv.org/abs/2505.10565)，但不假定 BIM 一定正确：
网络同时学习帧级和逐像素 BIM 可信度，并只让可信先验参与鲁棒局部度量对齐。

## 当前范围

- 单张 RGB 独立推理，不使用相邻 RGB 或未来帧；
- DA3-Metric-Large 作为冻结的初始预测器，训练前缓存深度；
- BIM 渲染提供深度、有效掩码、相机空间法向和边缘；
- 相邻 LiDAR 扫描融合仅生成训练/评测 GT，推理不读取 PCD；
- 输出细化深度、BIM 可信度、局部尺度/偏移和预测不确定性。

DA3 在第一阶段被完全冻结并离线缓存，因此训练显存只用于可信度和细化网络。后续若数据量
足够，可以再接入 DA3 解码器特征并只解冻最后两层。

## 方法

### 学习式 BIM 可信度

可信度网络输入：

```text
RGB, DA3深度, DA3置信度, BIM深度, BIM有效掩码,
BIM法向, BIM边缘, |log(BIM)-log(DA3)|
```

融合 LiDAR GT 自动生成软标签：

```text
C* = sigmoid((error_DA3 - error_BIM - margin) / temperature)
```

像素分支预测局部可信度，瓶颈特征经全局池化后预测帧级可信度，两者的 logit 相加得到
最终门控。因此整帧位姿/BIM 明显异常时，帧级分支能整体抑制 BIM；局部动态物体或构件
差异则由像素分支拒绝。

`C*=1` 表示 BIM 比 DA3 更接近 GT，`C*=0` 表示应该拒绝 BIM。标签只在
LiDAR GT、BIM 和 DA3 同时有效的位置监督；PCD 不进入推理网络。

### 鲁棒局部度量对齐

在局部窗口中拟合：

```text
D_BIM = a(x) * D_DA3 + b(x)
```

权重由学习到的 BIM 可信度、BIM 有效性和非边缘掩码共同决定。模块执行一轮
Huber/IRLS 降权，降低错误构件、动态遮挡和轻微位姿误差的影响。支持不足时严格退回原始
DA3 深度。

### 零初始化条件细化

轻量 U-Net 预测有界的对数深度残差和 log variance。最后一层零初始化，所以训练开始时：

```text
D_refined = D_DA3
```

不会因随机初始化破坏已有 DA3 结果。

## 项目结构

```text
BIM-PriorDA3/
├── configs/                       # 实验、数据、模型、损失与训练配置
├── scripts/
│   ├── download_slabim.py         # 可续传下载、可选rosbag、解压后删压缩包
│   ├── recover_slabim_poses.py    # rosbag Livox扫描到官方PCD的逐帧ICP位姿
│   ├── verify_slabim_dataset.py   # 原始数据、时间戳和位姿完整性检查
│   ├── run_slabim_experiments.py # 可断点续跑的完整实验入口
│   ├── prepare_dataset.py         # DA3/BIM/融合PCD样本生成
│   ├── audit_dataset.py           # 数据质量与泄漏检查
│   ├── visualize_sample.py        # RGB/DA3/BIM/GT/可信度标签检查
│   ├── train.py                   # AMP、梯度累积、早停、断点恢复
│   ├── evaluate.py                # 总体及分距离评测
│   ├── evaluate_reconstruction.py # BIM坐标系点云融合、Chamfer/F-score
│   ├── summarize_experiments.py   # 汇总2D/3D结果
│   └── infer.py                   # 单帧结果与可信度可视化
├── src/bim_priorda3/
│   ├── data/
│   │   ├── geometry.py            # 位姿、投影、z-buffer、遮挡融合
│   │   ├── slabim.py              # 下载、安全解压和数据完整性
│   │   ├── pose_recovery.py       # rosbag解析、ICP、平滑和诊断
│   │   ├── preparation.py         # SLABIM/BIM/DA3 数据制备
│   │   └── dataset.py             # Dataset、增强、可信度标签
│   ├── models/
│   │   ├── trust.py               # 可学习 BIM 可信度
│   │   ├── alignment.py           # 可微鲁棒局部仿射场
│   │   ├── refiner.py             # 零初始化条件细化器
│   │   └── system.py              # 完整模型
│   ├── losses.py
│   ├── metrics.py
│   └── engine.py
└── tests/
```

## 从下载到完整实验

配置默认查找与项目同级的 `../SLABIM`，也可用环境变量或统一入口的
`--slabim-root` 指定任意位置。

安装完整数据工具：

```bash
cd /path/to/BIM-PriorDA3
pip install -e '.[slabim,da3,dev]'
```

仅检查已经下载好的六区域数据，不做修改：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /path/to/SLABIM \
  --stages verify
```

从互联网下载到训练、单帧深度评测、三维重建和报告的完整流水线：

```bash
python scripts/run_slabim_experiments.py \
  --slabim-root /path/to/SLABIM \
  --stages all
```

完整流程依次执行：

```text
download -> poses -> verify -> prepare -> audit -> anchors
-> train-v1 -> eval-v1 -> cache-candidates
-> train-v3 -> eval-v3 -> reconstruct -> report
```

下载采用 `.part` 断点文件，ZIP 安全解压后默认删除；位姿恢复成功后默认删除 rosbag。
流水线将每个阶段记录到 `outputs/pipeline_state.json`，已有 checkpoint 会复用。用
`--dry-run --stages all` 可先查看命令，用 `--force` 才会重算已有训练或位姿结果。
数据制备默认固定为 train/validation stride 2、test stride 1，复现 565/82/164 帧协议，
不会误把测试集也降采样或把训练集翻倍。

如果数据已经下载但还没有恢复位姿：

```bash
python scripts/download_slabim.py \
  --root /path/to/SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3 \
  --rosbag-only

python scripts/recover_slabim_poses.py \
  --root /path/to/SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3 \
  --delete-rosbags
```

位姿不是 rosbag 中直接提供的 GT trajectory：脚本把每个 `/livox/lidar` 局部扫描配准到
同时间的官方 SLAM-global PCD，再平滑轨迹并与官方常量 map→BIM 变换复合。
`pose_recovery_summary.json` 和每区域 `*.diagnostics.npz` 必须保留并检查。

更详细的阶段、输入输出和失败恢复说明见
[`docs/EXPERIMENT_PIPELINE.md`](docs/EXPERIMENT_PIPELINE.md)。

## 数据坐标和 GT

官方 PCD 是 SLAM 全局坐标。本项目使用以下坐标链，不能把 PCD 直接当作局部扫描：

```text
SLAM-global PCD
  -> inverse(lidar_pose_local_to_slam_smoothed)
LiDAR-local
  -> lidar_pose_local_to_bim_from_rosbag
BIM/world
  -> inverse(center_lidar_to_BIM)
center LiDAR
  -> inverse(camera_to_lidar)
camera
```

每个邻近扫描先单独投影并 z-buffer；随后只保留最前方且彼此深度一致的扫描簇，后方遮挡点
不会进入 GT。`gt_support` 和簇内 MAD 共同形成监督权重。

## 安装

当前本机环境：

```bash
cd /home/bgao491/BIM-PriorDA3
conda run -n FastSAM python -m pip install -e . --no-deps
```

新服务器建议：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e /path/to/depth-anything-3
```

若需要从 rosbag 重新恢复位姿，使用 `pip install -e '.[slabim]'`。

## 制备数据

完整配置默认以四个区域训练、`5F_Region3` 验证选 checkpoint，
`5F_Region2` 只做最终跨区域测试：

```bash
conda run -n FastSAM python scripts/prepare_dataset.py \
  --config configs/slabim_single_frame_r50.yaml \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region3 \
  --stride 2

conda run -n FastSAM python scripts/prepare_dataset.py \
  --config configs/slabim_single_frame_r50.yaml \
  --regions 5F_Region2
```

已经存在的样本和 DA3 缓存会自动复用。`--overwrite` 才会重算。

## 训练和评测

```bash
conda run -n FastSAM python scripts/audit_dataset.py

conda run -n FastSAM python scripts/visualize_sample.py \
  --sample-id 5F_Region2/000082

conda run -n FastSAM python scripts/train.py \
  --config configs/slabim_single_frame_r50.yaml

conda run -n FastSAM python scripts/evaluate.py \
  --config configs/slabim_single_frame_r50.yaml \
  --checkpoint outputs/slabim_single_frame_r50/best.pt

conda run -n FastSAM python scripts/infer.py \
  --config configs/slabim_single_frame_r50.yaml \
  --checkpoint outputs/slabim_single_frame_r50/best.pt \
  --index 0
```

评测报告原始 DA3、粗对齐和学习式细化三种结果，并分别统计
`0.2–1 m`、`1–2 m`、`2–3 m`、`3–5 m`。

三维重建评测：

```bash
python scripts/evaluate_reconstruction.py \
  --config configs/slabim_single_frame_r50_v3.yaml \
  --checkpoint outputs/slabim_single_frame_r50_v3/best.pt \
  --split test \
  --save-clouds
```

它在 BIM 坐标系直接融合原始 DA3、解析 BIM 强基线和学习结果，报告双向最近邻
accuracy/completeness、Chamfer-L1，以及 5/10/20 cm precision、recall、F-score。
评测不做 post-hoc ICP。默认 `prediction-mask=all` 用于真实重建覆盖率；
`prediction-mask=gt` 只用于同一 GT 像素的诊断，不能替代完整重建结果。

本次 ±50 帧 GT 的完整训练、验证和最终测试结果见
[`RESULTS_R50.md`](RESULTS_R50.md)。

在强尺度基线之上继续优化的学习式候选融合 V3、消融和最终提升见
[`RESULTS_V3.md`](RESULTS_V3.md)。

迁移至新机器或开启新会话前，请复制并优先阅读
[`chatlog.md`](chatlog.md)、
[`CLOUD_HANDOFF.md`](CLOUD_HANDOFF.md) 与
[`CLOUD_MIGRATION.md`](CLOUD_MIGRATION.md)。

## 显存

默认 504×504、batch size 1、AMP、梯度累积 4。当前模型约 145 万可训练参数。发生
CUDA OOM 时训练脚本会写出 `oom_state.pt` 和 `OOM_README.txt` 后退出，不会继续损坏
训练状态。迁移云端时复制以下内容即可：

```text
BIM-PriorDA3/
depth-anything-3/
SLABIM/calibration_files/
SLABIM/BIM/
SLABIM/sensor_data/*/images/
BIM-PriorDA3/data/processed/
```

若已复制 `data/processed`，训练服务器不需要 PCD；只有重新制备 GT 时才需要
`SLABIM/sensor_data/*/points/`。
