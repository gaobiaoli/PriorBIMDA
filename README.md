# BIM-PriorDA3

BIM-PriorDA3 用固定 BIM 几何先验细化单帧度量深度。仓库目前提供两条完整、可审计的
数据路径：SLABIM，以及 2D-3D-S `Area_1` + BIMSyn。原始数据不随仓库分发；下载、校验、
位姿/配准、DA3 缓存、样本制备、训练和固定支持集评测均有脚本。

```text
RGB ──> pinned DA3 metric depth ──> one frozen universal scale ──> depth anchor
                         BIM ─────> geometry condition + deterministic comparator
RGB + DA3 geometry + BIM features ─> bounded multi-scale residual ─> depth
```

当前主模型不是旧版 BIM/V1 加权融合。SLABIM 与 Area_1 使用完全相同的尺度规则，网络以
尺度矫正后的 DA3 为锚点，学习帧级、低频和细节 log-residual；BIM-direct 只作为共享的强
非学习比较器。Area_1 跨域初始化会把残差头归零。推理不读取 GT、语义标签或家具 mask。
公式、参数来源与适用边界见[统一尺度协议](docs/UNIVERSAL_SCALE_PROTOCOL.md)。

## 统一协议结果

统一深度协议为 `0.2–5.0 m`，下表是 pixel-micro 指标。原始 DA3 的 AbsRel 在 SLABIM
test 为 `0.19935`，在 Area_1 blind test 为 `0.30123`。

| 数据集 / 方法 | 帧数 | AbsRel | MAE (m) | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|---:|
| SLABIM raw DA3 | 108 | 0.19935 | 0.31109 | 0.42167 | 0.76328 |
| SLABIM universal scale | 108 | 0.06361 | 0.10316 | 0.24305 | 0.96457 |
| SLABIM universal BIM-direct | 108 | 0.06263 | 0.10334 | 0.24761 | 0.96320 |
| SLABIM learned refiner | 108 | **0.05601** | **0.09210** | **0.22725** | **0.97759** |
| Area_1 raw DA3 | 1641 | 0.30123 | 0.67323 | 0.83485 | 0.26437 |
| Area_1 universal scale | 1641 | 0.07752 | 0.13938 | 0.31541 | 0.93710 |
| Area_1 universal BIM-direct | 1641 | 0.07815 | 0.13891 | 0.31350 | 0.93740 |
| Area_1 learned refiner | 1641 | **0.06689** | **0.11761** | **0.30823** | **0.94295** |

相对同一 `universal BIM-direct`，学习模型在 SLABIM/Area_1 test 的 AbsRel 分别改善
`10.56%/14.41%`。Area_1 家具子集也改善；BIM 冲突子集的点估计改善，但其房间级 AbsRel
bootstrap 95% CI 跨 0，不能宣称该子集已有显著优势。机器可读汇总、逐帧结果和训练审计见
[results/](results/)。

SLABIM 的 108 帧三维融合评测也优于 direct：Chamfer-L1 0.09109 m vs. 0.10515 m，
F-score@10 cm 0.79622 vs. 0.74003。

## 安装

推荐 Python 3.10+；Stanford 路径需要 IfcOpenShell。

```bash
git clone https://github.com/gaobiaoli/PriorBIMDA.git
cd PriorBIMDA
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[slabim,stanford,da3,dev]'
```

若 `depth-anything-3` 需要从源码安装：

```bash
pip install -e '.[slabim,stanford,dev]'
pip install -e /path/to/depth-anything-3
```

所有正式配置固定 DA3 Metric Large revision
`4010e39f3634a45bc60553321fb49fb760bd594e`。首次联网制备会缓存该 revision；E2E 训练从
本地缓存加载，避免运行中模型漂移。

## SLABIM 快速复现

数据默认放在仓库同级 `../SLABIM`，也可设置 `BIM_PRIORDA3_SLABIM_ROOT`。先查看完整命令：

```bash
.venv/bin/python scripts/pipelines/run_slabim_experiments.py \
  --slabim-root ../SLABIM --stages all --dry-run
```

确认磁盘、数据条款和 GPU 后执行：

```bash
.venv/bin/python scripts/pipelines/run_slabim_experiments.py \
  --slabim-root ../SLABIM --stages all
```

流程依次执行 download → poses → verify → prepare → audit → pretrain →
finetune → evaluate → reconstruct；状态写入 `outputs/pipeline_state_slabim.json`。固定 annotation
在引入数据时排除 `ignore.txt` 的 90 个坏帧，并额外隔离 13 个共享 fused-LiDAR 帧；有效
train/val/test 为 `496/104/108`。

## Area_1 + BIMSyn 快速复现

2D-3D-S 要求用户先接受 Stanford/Matterport 数据许可。BIMSyn 未提供清晰的再分发许可证；
下载开关只表示用户已自行确认有权使用发布者链接，不表示本项目授予数据权利。

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

IFC 已足够计算；RVT 仅用于来源审计，可用 `--include-rvt` 下载。随后缓存 DA3、制备 10,327
帧，并按 room 严格划分 `7013/1673/1641`：

```bash
.venv/bin/python scripts/data/cache_stanford_da3.py \
  --config configs/stanford_area1_transfer.yaml
.venv/bin/python scripts/data/prepare_stanford_area1.py \
  --config configs/stanford_area1_transfer.yaml
```

仓库内 annotation、alignment 和 robust-scale receipt 用于核验已发布结果。若新制备的 manifest
产生了新的 preparation fingerprint，请用
`scripts/data/materialize_runtime_config.py` 生成本机 child config，
不要手工删除 provenance 校验。完整的 split、scale-selection、zero-shot、训练与评测命令见
[使用与脚本手册](docs/USER_GUIDE.md)。

## 配置

| 配置 | 用途 |
|---|---|
| `slabim_base.yaml` | 公共模型与数据默认值 |
| `slabim_pretrain.yaml` / `slabim.yaml` | pooled-clean frozen 两阶段训练 |
| `slabim_e2e.yaml` | SLABIM DA3 decoder/last-stage 联合微调 |
| `slabim_inference_example.yaml` | 新区域、无 GT 输入示例 |
| `stanford_area1_transfer*.yaml` | SLABIM checkpoint 零样本迁移评测 |
| `stanford_area1.yaml` | Area_1 最终 frozen target 模型 |
| `stanford_area1_e2e.yaml` | 未晋级的 E2E challenger 协议 |

配置只保留当前语义版本；历史 V1–V6 过程配置已移除。checkpoint 会严格验证模型配置和数据
provenance，跨数据集初始化必须显式 opt-in，resume 仍要求完全一致。

## 仓库结构

```text
configs/                 当前可执行协议
data/annotations/        固定 split（不含原始图像/深度）
data/provenance/         固定 alignment 与 train-only scale receipt
docs/                    技术说明、工作流和使用手册
results/                 可提交的小型正式结果与图表
scripts/data/            数据下载、校验、配准、split 与样本制备
scripts/model/           模型训练、推理与评测
scripts/pipelines/       可续跑实验编排与环境验证
scripts/analysis/        离线分析、bootstrap 与可视化
src/bim_priorda3/        数据适配、基线、模型、损失和 provenance
tests/                   单元与协议回归测试
outputs/                 本机 checkpoint/运行记录；默认不进入 Git
```

发布 checkpoint 的角色、大小和 SHA-256 见 [results/manifest.json](results/manifest.json)。大型
E2E 权重应放 GitHub Release 或 Hugging Face，不应直接提交 Git 历史。

## 验证

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
.venv/bin/python scripts/data/audit_dataset.py --config configs/slabim.yaml
```

## 文档

- [使用与脚本手册](docs/USER_GUIDE.md)
- [训练前数据操作手册](docs/DATA_PREPARATION.md)
- [实验与数据流水线](docs/EXPERIMENT_PIPELINE.md)
- [三种深度改进方法](docs/THREE_DEPTH_REFINEMENT_METHODS.md)
- [统一尺度估计与模型锚点协议](docs/UNIVERSAL_SCALE_PROTOCOL.md)
- [Area_1 + BIMSyn 技术与结果](docs/STANFORD_BIMSYNC_EVALUATION.md)
- [科研评测协议、消融与敏感性设计](docs/EVALUATION_PROTOCOL.md)
- [论文与 PPT 素材目录](docs/assets/paper_evaluation/README.md)
- [Codex 对话要点与项目演化](docs/CODEX_WORKFLOW.md)

## 许可与研究边界

- 第三方数据和 DA3 权重不随本仓库的软件许可重新授权；用户必须遵守各发布者条款。
- SLABIM 位姿由 rosbag/官方 SLAM 点云离线 ICP 恢复，不应表述为无配准系统。
- Area_1 的固定 BIM→Area 变换由发布的 semantic structural mesh 辅助估计，属于
  scan-calibrated/oracle-style registration；实际部署应替换为测量或定位系统提供的变换。
- 软件仓库在公开发布前仍需由所有者选择并添加 `LICENSE`。没有 LICENSE 时，默认并不等于
  允许他人复制、修改或再分发代码。
