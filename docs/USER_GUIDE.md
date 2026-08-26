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

发布结果包含两种明确分开的支持域：SLABIM/Area_1 universal 兼容表限定 `0.2–5.0 m`；
当前 Area_1 推荐学习模型使用官方全部正深度，仅排除 `0/65535`。二者不得混表。
`ignore.txt` 是源数据错误清单，必须在 split 建立前排除，不能只在最终评测时临时过滤。
`outputs/` 是本机运行目录并被 Git 忽略；可提交结果放在 `results/`。

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
  --acknowledge-bimsyn-license \
  --include-pano
```

默认下载 30.44 GiB `Area_1 noXYZ` TAR，并只解出规则视图的 RGB/depth/pose/semantic 及
`semantic.obj/.mtl`；`--include-pano` 额外解出 equirectangular RGB/depth/pose/semantic；
还下载 44 个 IFC。`--include-rvt` 会额外下载 44 个 RVT，但计算不用 RVT。下载器支持续传
并验证固定 size/hash manifest。只运行 regular-view 任务时可省略 `--include-pano`。

```bash
.venv/bin/python scripts/data/verify_stanford_bimsyn_sources.py \
  --area-root ../Stanford2D3DS/no_xyz \
  --area-tar ../Stanford2D3DS/no_xyz/area_1_no_xyz.tar \
  --ifc-root ../BIMSyn/BIM_model/ifc \
  --require-pano \
  --output data/provenance/stanford_area1_sources.local.json
```

删除 TAR 后可省略 `--area-tar`；校验仍检查 4×10,327 个 regular 模态、语义 mesh 和 IFC
manifest。`--require-pano` 还会严格检查全景四模态清单和一一配对；未下载 pano 时必须省略
它，不能把 regular-only receipt 当作全景来源证明。

### 3.2 固定 BIM 配准

正式配置使用仓库内冻结 receipt：

```text
data/provenance/stanford_area1_bimsyn_alignment.json
SHA256 079ff394fbfa9317953e0358d71e0548cd39171278dd16121d6c300c5a23e6d6
```

它把 44 个 room-local IFC 合入一个 Area_1 全局坐标系。旧正式 benchmark 使用
wall/floor/ceiling/column/beam 的 bounded-core prior；当前 hit-only 诊断还保留
door/window，只排除 furniture/proxy/MEP，并以“有限正值命中”而不是 0.2–5.0 m 生成 BIM
mask。两者的 GT support 都是 0.2–5.0 m，且 processed root 必须分开。若需重建配准，不要
直接覆盖冻结文件：

```bash
.venv/bin/python scripts/data/register_stanford_bimsyn.py \
  --semantic-obj ../Stanford2D3DS/no_xyz/area_1/3d/semantic.obj \
  --ifc-dir ../BIMSyn/BIM_model/ifc \
  --output data/provenance/stanford_area1_alignment.local.json

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1_attentive_scale_da3_features_hit_only.yaml \
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

若使用仓库内冻结 alignment，可跳过 preparation-only child config；重制当前规则时上述两条
命令直接使用 `configs/stanford_area1_attentive_scale_da3_features_hit_only.yaml`。不要把它的
输出指向旧 `stanford_area1_504`，否则会混淆两份 preparation fingerprint。

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

### 3.5 可选 pano tangent DA3 缓存

只有全景实验需要这一步。`cache_stanford_pano_da3.py` 从 ERP RGB 生成 perspective tangent
images，使用配置锁定的 DA3 revision 缓存 z-depth/confidence，并把每个文件和相机几何写入
不可变 manifest。缓存阶段不读取 pano depth、pano semantic 或 regular GT：

```bash
.venv/bin/python scripts/data/cache_stanford_pano_da3.py \
  --config configs/local/stanford_area1.yaml \
  --split val \
  --preset nested14 \
  --face-resolution 504 \
  --log-every 1
```

脚本结束时会打印实际 manifest 路径，形如
`<processed_root>/pano_da3/nested14_r504_<geometry-hash>/manifests/val_full.json`。正式 Route P
当前要求完整 `nested14` manifest；`cubemap6` 可用于缓存/几何敏感性，但不能替代该正式输入。
`--max-stations` 生成的 manifest 明确标为 exploratory。test 缓存必须在 protocol 冻结后增加
`--split test --confirm-test`。

### 3.6 零样本迁移、训练与最终 regular-view 评测

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

下面的 bounded-core、`0.2–5.0 m` 冻结 DA3 feature 模型属于历史研究候选，不是当前全深度
发布 checkpoint。该研究线已回退到无加法版本。制备完成后，先一次性缓存与同一 pinned DA3
预处理绑定的第 11/23 层 token，再从头
执行 scale/refiner/joint 三阶段训练：

```bash
.venv/bin/python scripts/data/cache_stanford_da3_features.py \
  --config configs/stanford_area1_attentive_scale_da3_features.yaml \
  --device cuda --batch-size 16

.venv/bin/python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_da3_features.yaml \
  --device cuda

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale_da3_features.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_da3_features/accepted.pt \
  --split val \
  --output results/stanford_area1/attentive_scale_da3_features_stage_ablation_val \
  --device cuda --batch-size 8 --inference-seed 42
```

对 frame-disabled attention 模型，评测器同时报告 `coarse`、`scale_plus_low` 与 `refined`，
对应 `scale`、`scale+r_low` 与 `scale+r_low+r_detail`。只有在 validation 冻结 checkpoint
和输出选择后才能执行一次 test；现有 Area_1 test 已揭盲，候选结果只可作诊断。hybrid
additive 配置和结果仍保留用于复现负实验，但不再作为活动候选。

若复现无 BIM 距离截断的 hit-only 诊断，将上述三个命令的配置替换为
`configs/stanford_area1_attentive_scale_da3_features_hit_only.yaml`，评测输出使用
`attentive_scale_da3_features_hit_only_{val,test}`，并显式增加
`--allow-unverified-robust-comparator`。该标志不是关闭数据 provenance；它只记录旧的
train-only robust-cap receipt 与新 manifest 不同。现有结果为 val/test
`0.06306/0.06585`，未超过 bounded-core prior 的 `0.06209/0.06244`，因此不能把 hit-only
配置替代上面的推荐候选。

若只做“不限制 0.2--5.0 m”的全图诊断，不需要重新制备或训练；在相同评测命令中增加
`--depth-support all-valid`。该模式从官方 regular 深度 PNG 重新读取 GT，保留全部正深度，
仅排除 `0` 与无效哨兵 `65535`。输出会显式记录为 out-of-training-support diagnostic，
不能替代默认 `configured` 模式的论文主表：

```bash
.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_da3_features_hit_only/accepted.pt \
  --split val --depth-support all-valid \
  --output results/stanford_area1/attentive_scale_da3_features_hit_only_all_valid_val \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --allow-unverified-robust-comparator
```

当前同一 checkpoint 的 all-valid pixel-micro final AbsRel 为 validation `0.07214`、
test `0.06744`；默认 0.2--5.0 m 结果分别为 `0.06306`、`0.06585`。原 bounded-core
checkpoint 在相同 all-valid support 上为 validation `0.07420`、test `0.06433`，结果保存在
`attentive_scale_da3_features_bounded_prior_all_valid_{val,test}`。

当前 Area_1 推荐发布模型让训练监督、validation 选点和最终评测全部使用官方全深度，而不
只是对旧 checkpoint 做全图诊断。使用以下冻结配置：

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --device cuda

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth/accepted.pt \
  --split val --depth-support all-valid \
  --output results/stanford_area1/attentive_scale_da3_features_hit_only_full_depth_val \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --allow-unverified-robust-comparator
```

把 `val` 与目录后缀替换为 `test` 即可复现 test。该配置从官方 PNG 动态重载 GT，原始值
`0`/`65535` 之外的全部深度都进入 loss 和指标，并把模型输出上限提高到 128 m；没有使用
`0.2--5.0 m` GT mask。现有 pixel-micro final AbsRel 为 validation `0.06861`、test
`0.06741`。checkpoint 由 validation 选择，SHA256 为
`f330a987d638482636e225ebdf326612209fa672ea3c5c77a11049f05b655349`。它是当前公开复现和
部署入口；Area_1 test 已在此前迭代中揭盲，所以论文仍须把 test 数字标成 post-hoc，而不能
称为新盲测。完整逐阶段指标和旧 checkpoint 的同 support 对照见
[ATTENTIVE_SCALE_EXPERIMENT.md](ATTENTIVE_SCALE_EXPERIMENT.md#full-depth-supervised-retraining)。

如需检查训练集拟合程度，可把同一评测命令中的 `--split val` 改为 `--split train`，输出目录
改为 `attentive_scale_da3_features_hit_only_full_depth_train`。这会以 inference mode、无数据
增强、无参数更新的方式评测全部 7,013 帧；GT 只用于预测后的指标计算。当前 final train
AbsRel 为 `0.06672`，但该诊断不得用于 checkpoint 选择或论文主结果。

固定 BIM/DA3 ratio 分位数的 train-only 搜索与冻结 test 审计使用独立脚本：

```bash
.venv/bin/python scripts/analysis/search_stanford_scale_quantile.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --split train \
  --output results/stanford_area1/fixed_scale_quantile_full_depth_train/selection.json

.venv/bin/python scripts/analysis/search_stanford_scale_quantile.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --split test \
  --selection-receipt results/stanford_area1/fixed_scale_quantile_full_depth_train/selection.json \
  --output results/stanford_area1/fixed_scale_quantile_full_depth_test
```

test 模式不提供 `--quantile` 参数，必须读取 train receipt，防止看过 test 后反复试数值。
当前 train 选择 q56，但 test AbsRel `0.12759` 差于 robust scale `0.10840`，因此这是负诊断，
不得替换 universal scale 协议。

不要把 `configs/stanford_area1_reliability_gated_full_depth.yaml` 用作发布配置。它是回退后的
负实验：final validation/test AbsRel 为 `0.06928/0.06884`，均差于上述发布模型；配置和结果
仅用于审计。

### 3.7 全景联合评测

不传 `--tangent-manifest` 时，评测器运行 Route R：把同一 station 的 regular predictions
变换成 pano radial range，在 regular-covered 固定 support 上比较单来源选择、多来源融合和
BIM 分支：

```bash
.venv/bin/python scripts/model/evaluate_stanford_pano.py \
  --config configs/stanford_area1.yaml \
  --split val \
  --output outputs/stanford_area1/pano_val_training_free \
  --device cuda \
  --batch-size 8 \
  --pano-height 512 \
  --bootstrap-repetitions 10000 \
  --seed 42
```

若使用 fresh-data 本机 config，请替换上面的 config 路径。缓存 `nested14` 后，使用缓存
命令打印的准确路径启用 Route P、pano-only 和 regular+pano raw-DA3 分析：

```bash
PANO_TANGENT_MANIFEST=/absolute/path/to/printed/val_full.json

.venv/bin/python scripts/model/evaluate_stanford_pano.py \
  --config configs/local/stanford_area1.yaml \
  --tangent-manifest "$PANO_TANGENT_MANIFEST" \
  --split val \
  --output outputs/stanford_area1/pano_val_route_p_training_free \
  --device cuda \
  --batch-size 8 \
  --pano-height 512 \
  --bootstrap-repetitions 10000 \
  --seed 42
```

当前主协议只验证 training-free pano 联合估计及确定性 `universal_scale`/`bim_direct`，因此
上述命令故意不加载 checkpoint。`--checkpoint` 只用于复核已归档的 learned 输出，不参与
fusion 选择、pano 主表或素材图。

`single_best_view` 是在每个 ERP 像素选一个来源的 no-fusion mosaic，并不表示整个 station 只
输入一张 regular image；论文中的“严格单图”使用独立的整站单帧 support。validation-only
诊断命令为：

```bash
.venv/bin/python scripts/analysis/evaluate_stanford_pano_single_plus_tangent.py \
  --config configs/stanford_area1.yaml \
  --tangent-manifest "$PANO_TANGENT_MANIFEST" \
  --output outputs/stanford_area1/pano_val_single_plus_tangent
```

该脚本没有 test、checkpoint、BIM 或可调 fusion 参数入口；它比较同一 GT-free selected
whole frame、+tangent6 与 +tangent14。当前 validation 结果显示覆盖率接近整球，但相同单图
support 上误差显著增加，不能用多 regular 的正结果替代“pano 优于单图”的证据。

上述严格单图实验只在所选一张图对应的窄 ERP support 上评测。若目标是验证全景联合能否改善
数据集原有的 regular-view 深度，应使用 round-trip 入口：全部同站 regular 预测先投到 ERP，
加入可选 pano tangent 后联合，再反投影回每一张原始 regular，并在每张图完整、相同的
`gt_valid` 上计算指标：

```bash
.venv/bin/python scripts/analysis/evaluate_stanford_pano_regular_roundtrip.py \
  --config configs/local/stanford_area1.yaml \
  --tangent-manifest "$PANO_TANGENT_MANIFEST" \
  --output outputs/stanford_area1/pano_val_regular_roundtrip_reproduction
```

该脚本固定为 validation-only，不读取 pano GT，不暴露 test/checkpoint/BIM/fusion 参数。正式
1,673 帧 validation 结果中，同一 weighted-log 融合器加入 tangent14 后 AbsRel
0.26682→0.18654（−30.09%，7/7 房间改善）；原始逐帧 DA3 为 0.27710。该结论用于冻结未来
盲测协议，不能作为已经揭盲 test 的事后替换。

若问题是“在每张 regular 已经做相同 BIM scale 后，多 regular ERP 联合还能提高多少”，运行
不含 tangent/pano 输入的同基线入口：

```bash
.venv/bin/python scripts/analysis/evaluate_stanford_bim_scale_roundtrip.py \
  --config configs/local/stanford_area1.yaml \
  --output outputs/stanford_area1/pano_val_bim_scale_regular_roundtrip
```

正式 validation 结果为逐帧 universal scale AbsRel 0.08644、同站 joint Huber 回投 0.07595，
相对下降 12.14%；7/7 房间改善，room-cluster 95% CI 为 [−0.01231,−0.00700]。该脚本固定
1,673 张原始 regular 的完整相同 GT support，不读取 pano RGB/GT，不加载 tangent、checkpoint
或 learned 模型。

正式主指标为精确
solid-angle 加权后的 equal-station macro，并用 room-cluster paired bootstrap 给区间。
`--max-stations` 只允许 smoke run。test 缓存和 evaluator 都要求 `--confirm-test`，只能在
validation 参数与方法选择全部冻结后各运行一次。完整的固定 support、pano-only
coverage 和结论边界见[全景评测协议](PANO_DEPTH_EVALUATION.md)。

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
| `data/download_stanford_area1.py` | 下载/选择性解压 Area_1 regular/pano，并从 BIMSyn 发布目录下载 IFC/RVT |
| `data/verify_stanford_bimsyn_sources.py` | 按固定清单验证 Area_1 regular、可选 pano 和 BIMSyn 并写 receipt |
| `data/register_stanford_bimsyn.py` | 估计固定 BIM room → Area 4-DoF 变换 |
| `data/cache_stanford_da3.py` | 按 pinned revision 缓存 Area_1 单帧 DA3 |
| `data/cache_stanford_da3_features.py` | 为冻结 DA3-feature 模型缓存第 11/23 层 token，并绑定完整 manifest/revision |
| `data/cache_stanford_pano_da3.py` | 生成 cubemap6/nested14 tangent RGB，缓存 pinned DA3 并写不可变 split manifest |
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
| `model/evaluate_stanford_pano.py` | Area_1 regular-to-pano 融合；可选 nested14 Route P/pano-only/regular+pano 评测 |
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
| `analysis/ablate_deterministic_bim_direct.py` | 对 non-learning BIM-direct 做逐因素 validation 消融 |
| `analysis/ablate_oracle_semantic_bim.py` | train/val-only 官方语义 + BIM ray category 的逐帧全局尺度 oracle；禁止 test |
| `analysis/analyze_scale_residual_distribution.py` | train/val-only 尺度后像素残差诊断；比较固定米制、距离比例和混合误差模型，禁止 test |
| `analysis/search_stanford_scale_quantile.py` | official-all-valid train 搜索固定 BIM/DA3 分位数；test 强制读取冻结 receipt |
| `analysis/evaluate_stanford_pano_single_plus_tangent.py` | val-only 严格单张 regular + tangent6/14 对照；无 test/learned/BIM |
| `analysis/evaluate_stanford_pano_regular_roundtrip.py` | val-only 全部 regular→ERP 联合→原 regular 回投评测；固定完整 regular GT support |
| `analysis/evaluate_stanford_bim_scale_roundtrip.py` | val-only 同一逐帧 BIM-scale 与 regular→ERP joint→regular 的严格配对评测 |
| `analysis/generate_pano_evaluation_assets.py` | 从冻结 pano 标量产物生成 PNG/SVG/PDF 论文图 |
| `analysis/export_stanford_pano_panels.py` | 按固定 val 规则导出三套本地 pano panel；不加载 checkpoint |

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

5. `outputs/` 不进入 Git。把 `results/manifest.json` 中 `publish=true` 的三份 checkpoint
   上传到 Release/Hugging Face，并将真实 URL 补入清单；大型 E2E 和负实验 checkpoint
   不应直接提交 Git。
6. 所有者必须在公开发布前选择软件 `LICENSE`；第三方数据许可需单独遵守，不能随代码
   LICENSE 自动继承。
