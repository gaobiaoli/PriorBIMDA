# 训练前数据操作手册

本页只说明“从网络公开来源到可以训练的 dataset”。数据代码集中在
`scripts/data/`，模型训练、推理和评测集中在 `scripts/model/`，两者不再混放。
所有命令均从仓库根目录执行。

## 1. 代码边界

```text
scripts/data/       下载 → 来源校验 → 位姿/配准 → DA3/BIM 制备 → split → 数据审计
scripts/model/      train → infer → 2D/3D evaluate
scripts/pipelines/  把上述步骤组成可续跑流程
scripts/analysis/   结果分析、bootstrap 和可视化
```

`src/bim_priorda3/data/` 是可复用的数据库实现；`scripts/data/` 只是 CLI 入口。
这样可以在不导入训练脚本的情况下完成数据准备和审计。

## 2. 通用规则

1. 先安装项目：`pip install -e '.[slabim,stanford,da3,dev]'`。
2. 原始数据放在仓库外；`data/processed/` 是可重建缓存，不进入 Git。
3. 下载器校验固定 revision、字节数和 SHA-256；不应用手工改名文件绕过校验。
4. `ignore.txt` 中的 SLABIM 坏帧在构建 split 时排除，不是在评测时临时删除。
5. 正式深度范围是 `0.2–5.0 m`。
6. smoke 制备必须使用过滤参数和独立输出，不能覆盖 canonical manifest。
7. annotation、alignment、scale receipt 是协议产物；新数据使用 `.local` 文件，
   不覆盖仓库中的冻结回执。

## 3. SLABIM：从下载到训练就绪

### 3.1 原始数据

```bash
.venv/bin/python scripts/data/download_slabim.py \
  --root ../SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3 \
  --include-rosbag
```

下载清单固定在 `src/bim_priorda3/data/slabim_download_manifest.json`。该步产生
SLABIM BIM、相机/雷达标定、RGB、SLAM PCD 和用于恢复位姿的 rosbag。

### 3.2 位姿恢复与原始来源校验

```bash
.venv/bin/python scripts/data/recover_slabim_poses.py \
  --root ../SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3 \
  --output data/provenance/slabim_pose_recovery.local.json

.venv/bin/python scripts/data/verify_slabim_dataset.py \
  --root ../SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3
```

位姿由 rosbag 轨迹与官方 SLAM PCD 做离线 ICP 得到；这是训练输入，不是模型
在推理时自动估计的。若已删除 rosbag，不要用 `--overwrite` 重算位姿。

### 3.3 制备 NPZ/manifest

```bash
.venv/bin/python scripts/data/prepare_dataset.py --config configs/slabim.yaml
```

输出默认位于 `data/processed/slabim_504_r50/`：

- `manifest.jsonl`：样本 ID、RGB 和 NPZ 路径、制备 fingerprint；
- `samples/<region>/<frame>.npz`：DA3 base/confidence、BIM depth/normal/edge、GT 及 mask；
- DA3 使用配置中固定的 Metric Large revision。

新区域无 GT 路径必须加 `--inference-only`，并使用独立 `processed_root`。

### 3.4 split、坏帧与泄漏审计

仓库已提供正式 pooled-clean annotation。若重建，使用新文件名：

```bash
.venv/bin/python scripts/data/build_global_split.py \
  --manifest data/processed/slabim_504_r50/manifest.jsonl \
  --ignore-file ignore.txt \
  --output data/annotations/slabim_clean_global.local.jsonl

.venv/bin/python scripts/data/audit_dataset.py \
  --config configs/slabim.yaml --ignore-file ignore.txt
```

正式历史数量是：811 条 manifest population，103 条 excluded（90 坏帧 + 13
fused-LiDAR embargo），活动 train/val/test = `496/104/108`。审计必须确认
fused-LiDAR 在活动 split 间无交集。

### 3.5 缓存非学习基线

```bash
.venv/bin/python scripts/data/cache_bim_baselines.py --config configs/slabim.yaml
```

这一步在样本中固定 global scale 与 direct BIM anchor。完成后才进入
`scripts/model/train.py`。

## 4. Area_1 + BIMSyn：从授权下载到训练就绪

### 4.1 获取与逐文件校验

2D-3D-S 需先接受 Stanford/Matterport 数据许可；BIMSyn 未提供清晰的
再分发许可，用户需自行确认有权使用。

```bash
.venv/bin/python scripts/data/download_stanford_area1.py \
  --stanford-root ../Stanford2D3DS \
  --bimsyn-root ../BIMSyn \
  --accept-stanford-license \
  --acknowledge-bimsyn-license

.venv/bin/python scripts/data/verify_stanford_bimsyn_sources.py \
  --area-root ../Stanford2D3DS/no_xyz \
  --area-tar ../Stanford2D3DS/no_xyz/area_1_no_xyz.tar \
  --ifc-root ../BIMSyn/BIM_model/ifc \
  --output data/provenance/stanford_area1_sources.local.json
```

默认下载 Area_1 noXYZ 规则视图与 44 个 IFC。RVT 仅用于来源审计，可选
`--include-rvt`；模型计算不依赖 RVT。BIMSyn 清单位于
`src/bim_priorda3/data/bimsyn_models_manifest.json`。

### 4.2 固定 BIM 配准

已发布结果使用 `data/provenance/stanford_area1_bimsyn_alignment.json`。若需从
公开数据重建，必须写新回执：

```bash
.venv/bin/python scripts/data/register_stanford_bimsyn.py \
  --semantic-obj ../Stanford2D3DS/no_xyz/area_1/3d/semantic.obj \
  --ifc-dir ../BIMSyn/BIM_model/ifc \
  --output data/provenance/stanford_area1_alignment.local.json

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1_transfer.yaml \
  --alignment-receipt data/provenance/stanford_area1_alignment.local.json \
  --preparation-only \
  --output configs/local/stanford_area1_prepare.yaml
```

配准使用目标域 semantic structural mesh，是 scan-calibrated/oracle-style 协议；
不能把它宣称为无标定泛化。每个 room 只保留一个固定变换，禁止逐帧
使用 GT 重配准。

### 4.3 DA3 缓存与样本制备

```bash
.venv/bin/python scripts/data/cache_stanford_da3.py \
  --config configs/local/stanford_area1_prepare.yaml --log-every 100

.venv/bin/python scripts/data/prepare_stanford_area1.py \
  --config configs/local/stanford_area1_prepare.yaml
```

输出 `data/processed/stanford_area1_504/manifest.jsonl` 和 10,327 个样本。只有不带
`--rooms`、`--max-frames-per-room` 且 `stride=1` 的完整运行才会发布 canonical
manifest。数据层将 44 个固定 room IFC 合并为一个全局 core BIM，只保留
wall/floor/ceiling/column/beam。
若不重建 alignment，上述两条 `--config` 直接使用
`configs/stanford_area1_transfer.yaml`，不需要 preparation-only child config。

### 4.4 room split 与本机配置

```bash
.venv/bin/python scripts/data/build_stanford_room_split.py \
  --manifest data/processed/stanford_area1_504/manifest.jsonl \
  --output data/annotations/stanford_area1_room.local.jsonl \
  --receipt data/annotations/stanford_area1_room.local.receipt.json

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --alignment-receipt data/provenance/stanford_area1_alignment.local.json \
  --output configs/local/stanford_area1_selector.yaml
```

split 以 room/camera UUID 为组，历史数量是 30/7/7 rooms 和
`7013/1673/1641` frames。不得按帧随机拆分。若使用仓库内已冻结的 alignment，
可省略 `--alignment-receipt`。

### 4.5 train-only robust scale 回执

```bash
.venv/bin/python scripts/data/select_stanford_scale_caps.py \
  --config configs/local/stanford_area1_selector.yaml \
  --output data/provenance/stanford_area1_scale.local.json \
  --workers 8

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --alignment-receipt data/provenance/stanford_area1_alignment.local.json \
  --scale-receipt data/provenance/stanford_area1_scale.local.json \
  --experiment-output-dir outputs/stanford_area1_local \
  --output configs/local/stanford_area1.yaml
```

selector 只能打开 train NPZ，并在回执中记录 validation/test opened=0。如果使用
仓库的冻结 alignment，上述两条 `--alignment-receipt` 均可省略。如需零样本迁移或
E2E，再以对应 transfer/E2E base config 生成各自的 local child config。

## 5. 训练前验收清单

执行 `scripts/model/train.py` 前，至少确认：

- 原始数据 verifier 退出码为 0；
- canonical manifest 存在，且每个活动 ID 只出现一次；
- annotation 穷举覆盖 manifest population，无未知/缺失 ID；
- SLABIM ignore 与 fused-LiDAR 隔离通过；Area_1 room/camera UUID 不跨 split；
- config 中 annotation raw SHA 和 split fingerprint 与当前文件一致；
- Area_1 alignment SHA、robust receipt SHA/协议和 train-only 开启计数验证通过；
- DA3 revision 为配置锁定值，缓存没有混入旧图像或其他 process resolution；
- 新训练使用空的 output directory，不把 smoke/history checkpoint 冒充正式运行。

可用以下命令做最后检查：

```bash
.venv/bin/python scripts/data/audit_dataset.py --config configs/slabim.yaml

.venv/bin/python scripts/pipelines/verify_cloud_setup.py \
  --config configs/slabim.yaml \
  --checkpoint outputs/slabim/accepted.pt
```

`scripts/pipelines/verify_cloud_setup.py` 是已有 checkpoint 的搬迁/smoke 验证；
它不替代 fresh 数据
verifier 或 annotation audit。

## 6. 开始模型阶段

数据验收通过后，才使用下列入口：

```bash
# SLABIM
.venv/bin/python scripts/model/train.py --config configs/slabim_pretrain.yaml --device cuda
.venv/bin/python scripts/model/train.py \
  --config configs/slabim.yaml \
  --init-checkpoint outputs/slabim_pretrain/accepted.pt --device cuda

# Area_1 target model
.venv/bin/python scripts/model/train.py \
  --config configs/local/stanford_area1.yaml \
  --init-checkpoint outputs/slabim/accepted.pt \
  --allow-cross-dataset-initialization --device cuda
```

训练、评测及 E2E 命令见 [使用与脚本手册](USER_GUIDE.md)。
