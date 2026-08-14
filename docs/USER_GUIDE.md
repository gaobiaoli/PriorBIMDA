# 使用与脚本手册

本文面向从公开网络资源开始复现的用户。所有命令均从仓库根目录执行；路径可通过配置、CLI
或 `BIM_PRIORDA3_SLABIM_ROOT` 调整。先运行 `--help` 和 `--dry-run`，再启动会持续数小时的
下载、DA3 缓存或训练。

## 1. 环境与约定

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[slabim,stanford,da3,dev]'
```

已发布结果的最终验证环境为 Python 3.11.15、PyTorch 2.13.0+cu130、
Depth Anything 3 0.1.1、Open3D 0.19.0、IfcOpenShell 0.8.5、NumPy 1.26.4 和
OpenCV 4.11.0.86。`pyproject.toml` 声明的是受支持下界；PyTorch/CUDA 应按实际 GPU
平台安装。模型仓库 revision 和数据文件哈希由配置/manifest 另外锁定。

推荐目录布局：

```text
workspace/
├── PriorBIMDA/
├── SLABIM/
├── Stanford2D3DS/
│   ├── no_xyz/area_1/
│   └── metadata/assets/semantic_labels.json
└── BIMSyn/BIM_model/
    ├── ifc/
    └── rvt/                 # 可选
```

正式深度指标均限定 `0.2–5.0 m`。`ignore.txt` 是源数据错误清单，必须在 split 建立前排除，
不能只在最终评测时临时过滤。`outputs/` 是本机运行目录并被 Git 忽略；可提交结果放在
`results/`。

## 2. SLABIM 完整流程

### 2.1 一键可续跑流程

```bash
.venv/bin/python scripts/pipelines/run_slabim_experiments.py \
  --slabim-root ../SLABIM --stages all --dry-run

.venv/bin/python scripts/pipelines/run_slabim_experiments.py \
  --slabim-root ../SLABIM --stages all
```

默认 stage：

1. `download`：从固定 revision 下载 BIM、标定、RGB、PCD，以及恢复位姿需要的 rosbag。
2. `poses`：将 rosbag 轨迹与官方 SLAM PCD 配准，生成 camera/LiDAR 到 BIM 的位姿。
3. `verify`：核对原始目录、文件哈希和各 region 所需输入。
4. `prepare`：运行 pinned DA3，渲染 BIM，生成 NPZ + manifest。
5. `audit`：验证 811 条 exhaustive annotation、90 个坏帧、13 个 LiDAR embargo，以及
   `496/104/108` train/val/test。
6. `pretrain`：使用 `configs/slabim_pretrain.yaml`。
7. `finetune`：从 pretrain checkpoint 初始化 `configs/slabim.yaml`。
8. `evaluate`：固定 test support 评测 raw DA3、scale、BIM-direct、refined。
9. `reconstruct`：将预测反投影到 BIM 坐标并进行 3D 评测。

运行状态写到 `outputs/pipeline_state_slabim.json`。默认跳过已验证阶段；`--force` 只应在确认
对应原始输入仍完整时使用。删除 rosbag 后不要强制重算 poses。

### 2.2 手动执行

```bash
.venv/bin/python scripts/data/download_slabim.py \
  --root ../SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3 \
  --include-rosbag

.venv/bin/python scripts/data/recover_slabim_poses.py \
  --root ../SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3 \
  --output data/provenance/slabim_pose_recovery.local.json

.venv/bin/python scripts/data/verify_slabim_dataset.py \
  --root ../SLABIM \
  --regions 3F_Region2 3F_Region3 4F_Region2 4F_Region3 5F_Region2 5F_Region3

.venv/bin/python scripts/data/prepare_dataset.py \
  --config configs/slabim.yaml

.venv/bin/python scripts/data/audit_dataset.py \
  --config configs/slabim.yaml \
  --ignore-file ignore.txt

.venv/bin/python scripts/model/train.py \
  --config configs/slabim_pretrain.yaml --device cuda

.venv/bin/python scripts/model/train.py \
  --config configs/slabim.yaml \
  --init-checkpoint outputs/slabim_pretrain/accepted.pt \
  --device cuda

.venv/bin/python scripts/model/evaluate.py \
  --config configs/slabim.yaml \
  --checkpoint outputs/slabim/accepted.pt \
  --split test \
  --output outputs/slabim/evaluation_test
```

E2E 是可选第三阶段；它加载当前 frozen checkpoint，并只微调 DA3 的最后阶段：

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/slabim_e2e.yaml \
  --init-checkpoint outputs/slabim/accepted.pt \
  --device cuda
```

### 2.3 新区域无 GT 推理

复制 `configs/slabim_inference_example.yaml`，只改 region、原始根目录和独立的
`processed_root`。制备时必须使用 `--inference-only`：

```bash
.venv/bin/python scripts/data/prepare_dataset.py \
  --config configs/slabim_inference_example.yaml \
  --regions 5F_Region1 \
  --inference-only \
  --replace-regions-in-manifest

.venv/bin/python scripts/model/infer.py \
  --config configs/slabim_inference_example.yaml \
  --checkpoint outputs/slabim/accepted.pt \
  --regions 5F_Region1 \
  --output-dir outputs/slabim_inference \
  --save-previews
```

该路径不加载 LiDAR GT，但仍需要可靠的相机到 BIM 位姿。

## 3. Area_1 + BIMSyn 完整流程

### 3.1 获取与校验

先完成 2D-3D-S 许可登记，并自行确认 BIMSyn 数据使用权限：

```bash
.venv/bin/python scripts/data/download_stanford_area1.py \
  --stanford-root ../Stanford2D3DS \
  --bimsyn-root ../BIMSyn \
  --accept-stanford-license \
  --acknowledge-bimsyn-license
```

默认下载 30.44 GiB `Area_1 noXYZ` TAR，并只解出规则视图的 RGB/depth/pose/semantic 及
`semantic.obj/.mtl`；还下载 44 个 IFC。`--include-rvt` 会额外下载 44 个 RVT，但计算不用
RVT。下载器支持续传并验证固定 size/hash manifest。

```bash
.venv/bin/python scripts/data/verify_stanford_bimsyn_sources.py \
  --area-root ../Stanford2D3DS/no_xyz \
  --area-tar ../Stanford2D3DS/no_xyz/area_1_no_xyz.tar \
  --ifc-root ../BIMSyn/BIM_model/ifc \
  --output data/provenance/stanford_area1_sources.local.json
```

删除 TAR 后可省略 `--area-tar`；校验仍检查 4×10,327 个模态、语义 mesh 和 IFC manifest。

### 3.2 固定 BIM 配准

正式配置使用仓库内冻结 receipt：

```text
data/provenance/stanford_area1_bimsyn_alignment.json
SHA256 079ff394fbfa9317953e0358d71e0548cd39171278dd16121d6c300c5a23e6d6
```

它把 44 个 room-local IFC 合入一个 Area_1 全局坐标系，仅保留 wall/floor/ceiling/
column/beam，排除 door/window/furniture/proxy/MEP。若需重建配准，不要直接覆盖冻结文件：

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

该方法用目标域 semantic structural mesh 求 4-DoF yaw+translation，属于 scan-calibrated/
oracle-style 协议。`--preparation-only` 在 manifest 尚未生成时只锁定 alignment，
不伪造 split provenance。实际部署应使用测量控制点或定位系统的外参。

### 3.3 DA3 缓存与样本制备

```bash
.venv/bin/python scripts/data/cache_stanford_da3.py \
  --config configs/local/stanford_area1_prepare.yaml \
  --log-every 100

.venv/bin/python scripts/data/prepare_stanford_area1.py \
  --config configs/local/stanford_area1_prepare.yaml
```

若使用仓库内冻结 alignment，可跳过 preparation-only child config，上述两条命令
直接使用 `configs/stanford_area1_transfer.yaml`。

不要用带 `--rooms`、`--max-frames-per-room` 或 `stride!=1` 的 smoke run 发布 canonical
manifest；过滤运行只生成样本，不会覆盖正式 manifest。

### 3.4 在新制备 fingerprint 上固定 split 与 scale

annotation 按 room 分组，不能逐帧随机拆分，否则同一 camera UUID 的不同视角会泄漏。
固定 `seed=42` 产生 30/7/7 rooms：

```bash
.venv/bin/python scripts/data/build_stanford_room_split.py \
  --manifest data/processed/stanford_area1_504/manifest.jsonl \
  --output data/annotations/stanford_area1_room.local.jsonl \
  --receipt data/annotations/stanford_area1_room.local.receipt.json
```

先生成只绑定新 split 的 selector config：

```bash
.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --output configs/local/stanford_area1_selector.yaml
```

scale cap 只能访问 train IDs；脚本会做固定 48 候选、leave-one-train-room-out 和第二遍
direct audit，并记录 val/test opened=0：

```bash
.venv/bin/python scripts/data/select_stanford_scale_caps.py \
  --config configs/local/stanford_area1_selector.yaml \
  --output data/provenance/stanford_area1_scale.local.json \
  --workers 8
```

最后生成与新 manifest/split/receipt 绑定的训练和迁移配置：

```bash
.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1_transfer.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --scale-receipt data/provenance/stanford_area1_scale.local.json \
  --output configs/local/stanford_area1_transfer.yaml

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --scale-receipt data/provenance/stanford_area1_scale.local.json \
  --experiment-output-dir outputs/stanford_area1_local \
  --output configs/local/stanford_area1.yaml
```

若使用重建的 alignment，再给上述命令增加
`--alignment-receipt data/provenance/stanford_area1_alignment.local.json`。

### 3.5 零样本迁移、训练与最终评测

先在 validation 上记录 SLABIM source 模型的零样本迁移；必须显式允许跨数据集 checkpoint：

```bash
.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/local/stanford_area1_transfer.yaml \
  --checkpoint outputs/slabim/accepted.pt \
  --split val \
  --cross-dataset \
  --output outputs/stanford_area1_transfer/frozen_val \
  --batch-size 8 --inference-seed 42
```

然后只用 train/val 训练 target frozen 模型：

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/local/stanford_area1.yaml \
  --init-checkpoint outputs/slabim/accepted.pt \
  --allow-cross-dataset-initialization \
  --device cuda
```

训练会对跨域 residual heads 精确归零，以统一尺度化 DA3 为乘法锚点；BIM-direct 只作
确定性基线和验收对照。先在 val 锁定
checkpoint；只在模型、配置和 claim 全部冻结后运行一次 test：

```bash
.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/local/stanford_area1.yaml \
  --checkpoint outputs/stanford_area1_local/accepted.pt \
  --split val \
  --output outputs/stanford_area1_local/formal_val \
  --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/local/stanford_area1.yaml \
  --checkpoint outputs/stanford_area1_local/accepted.pt \
  --split test \
  --output outputs/stanford_area1_local/formal_test \
  --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42
```

E2E 是 challenger，而非自动替代。它必须从 target frozen `accepted.pt` 初始化并保留 residual
heads；若 val 不优于 frozen，则不得访问 test 或发布为主模型。

## 4. 评测输出与判定

- `summary.json`：checkpoint/config/data hash、固定支持集聚合、距离子集和统计检验。
- `per_frame.csv`：逐帧方法指标；支持复核 bootstrap 和失败样本。
- `history.json`：逐 epoch 训练/验证指标与 acceptance gate。
- `run_state.json`：初始化来源、数据验证、随机种子、AMP/optimizer step 和完成状态。
- `accepted.pt`：通过 BIM-direct 安全门的 checkpoint；不等于一定优于另一个 learned model。

Area_1 主 claim 应同时检查 `all`、`furniture` 和 `bim_foreground_conflict` 的 pixel/frame/room
聚合，并报告 paired-room bootstrap。任何方法不得通过输出 NaN、非正值或空洞缩小支持集；
正式 evaluator 会 fail-fast。

## 5. 脚本索引

### `scripts/data/`：训练前数据操作

| 脚本 | 作用 |
|---|---|
| `data/download_slabim.py` | 按 region 续传/校验 SLABIM BIM、标定、图像、PCD、rosbag |
| `data/recover_slabim_poses.py` | 从 rosbag 与 SLAM PCD 恢复 SLABIM 位姿 |
| `data/verify_slabim_dataset.py` | 校验原始 SLABIM 输入是否完备 |
| `data/prepare_dataset.py` | 制备 SLABIM supervised 或 inference-only NPZ/manifest |
| `data/build_global_split.py` | 构建 SLABIM pooled annotation，不复制源文件 |
| `data/audit_dataset.py` | 审计 manifest、annotation、ignore、stride 和 LiDAR 隔离 |
| `data/download_stanford_area1.py` | 下载/解压 Area_1，并从 BIMSyn 发布目录下载 IFC/RVT |
| `data/verify_stanford_bimsyn_sources.py` | 按固定清单验证 Area_1 和 BIMSyn 并写 receipt |
| `data/register_stanford_bimsyn.py` | 估计固定 BIM room → Area 4-DoF 变换 |
| `data/cache_stanford_da3.py` | 按 pinned revision 缓存 Area_1 单帧 DA3 |
| `data/prepare_stanford_area1.py` | 生成 Area_1 RGB/DA3/global-envelope-BIM/GT/semantic-subset 样本 |
| `data/build_stanford_room_split.py` | 构建 room/camera-disjoint 30/7/7 annotation |
| `data/select_stanford_scale_caps.py` | 只用 train split 选择 robust scale cap 并写不可覆盖 receipt |
| `data/materialize_runtime_config.py` | 把 manifest/split/alignment/scale hash 固定到本机 child config |

详细的输入、输出、执行顺序和训练前验收条件见
[训练前数据操作手册](DATA_PREPARATION.md)。

### `scripts/model/`：训练、推理与评测

| 脚本 | 作用 |
|---|---|
| `model/train.py` | frozen 或 E2E 训练；严格初始化/resume/provenance/acceptance |
| `model/evaluate.py` | SLABIM 固定支持集 2D 深度评测 |
| `model/evaluate_stanford_area1.py` | Area_1 全体/家具/冲突子集及 room-bootstrap 评测 |
| `model/evaluate_reconstruction.py` | 反投影、融合并评测 3D 重建 |
| `model/infer.py` | 无 GT 新区域推理与输出保存 |

### `scripts/pipelines/`：实验编排

| 脚本 | 作用 |
|---|---|
| `pipelines/run_slabim_experiments.py` | SLABIM 可续跑主流程 |
| `pipelines/verify_cloud_setup.py` | 搬迁后做配置、checkpoint、前向与 GT 独立性 smoke check |

### `scripts/analysis/`：分析与可视化

| 脚本 | 作用 |
|---|---|
| `analysis/paired_bootstrap.py` | 从 SLABIM 逐帧 CSV 做配对 bootstrap |
| `analysis/visualize_sample.py` | 检查一个已制备 SLABIM 样本 |
| `analysis/visualize_prediction_sample.py` | 可视化指定 SLABIM 帧的多方法预测 |
| `analysis/visualize_stanford_sample.py` | 可视化 Area_1 帧、家具/冲突 mask 与预测 |
| `analysis/generate_paper_assets.py` | 从冻结结果生成量化图、敏感性图和三套过程图素材 |
| `analysis/export_stanford_qualitative_panels.py` | 按固定 validation 规则导出三套独立定性 panel 与统一色条 |

## 6. 可恢复性与发布

1. 不覆盖正式 annotation、alignment、scale receipt 或 formal test 目录。
2. fresh run 使用新的 `configs/local/*.yaml` 和新的 output 目录。
3. checkpoint 的 `model` 配置必须严格一致；跨数据集只允许 `--init-checkpoint` 或显式
   transfer evaluation，resume 永远不允许跨数据集。
4. 发布前运行：

   ```bash
   .venv/bin/pytest -q
   .venv/bin/ruff check src scripts tests
   sha256sum -c results/checkpoints.sha256
   ```

5. `outputs/` 不进入 Git。把两份统一主 checkpoint 上传到 Release/Hugging Face，并将真实 URL
   补入 `results/manifest.json`；大型 E2E checkpoint 不应直接提交 Git。
6. 所有者必须在公开发布前选择软件 `LICENSE`；第三方数据许可需单独遵守，不能随代码
   LICENSE 自动继承。
