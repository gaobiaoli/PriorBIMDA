# BIM-PriorDA3

BIM-PriorDA3 用固定 BIM 几何先验细化单帧度量深度。仓库目前提供两条完整、可审计的
数据路径：SLABIM，以及 2D-3D-S `Area_1` + BIMSyn。原始数据不随仓库分发；下载、校验、
位姿/配准、DA3 缓存、样本制备、训练和固定支持集评测均有脚本。

```text
RGB ──> pinned DA3 canonical cache ──> per-camera focal metric conversion ──> depth anchor
                         BIM ─────> geometry condition + deterministic comparator
RGB + DA3 geometry + BIM features ─> bounded multi-scale residual ─> depth
```

当前发布包包含两个不可混表的协议。SLABIM 与 Area_1 的兼容基线继续使用同一冻结 universal
scale，在 `0.2–5.0 m` 上学习 frame/low/detail log-residual；Area_1 当前推荐发布模型则使用
官方全部有效深度、逐帧 focal-corrected DA3、hit-only BIM、冻结 DA3 第 11/23 层 feature、attention global scale 与
low/detail refiner。两者都不是旧版 BIM/V1 加权融合，BIM-direct 只作为非学习比较器，推理
均不读取 GT、语义标签或家具 mask。支持域和 checkpoint 必须成对使用。

## 统一协议结果

统一深度协议为 `0.2–5.0 m`，下表是 pixel-micro 指标。2026-08 的焦距审计发现，历史
`raw_da3` 行直接评测了 DA3METRIC-LARGE 的 canonical-focal 输出，遗漏官方定义的
`mean(fx,fy)/300` 米制换算。为保留审计链，旧数值不删除，但明确改名为 canonical；真正不使用
BIM/GT 的 metric DA3 baseline 是新增的 focal-corrected 行。

| 数据集 / 方法 | 帧数 | AbsRel | MAE (m) | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|---:|
| SLABIM DA3 cached canonical（历史误标 raw） | 108 | 0.19935 | 0.31109 | 0.42167 | 0.76328 |
| SLABIM raw DA3 metric（focal-corrected） | 108 | 0.07373 | 0.11559 | 0.25568 | 0.93891 |
| SLABIM universal scale | 108 | 0.06361 | 0.10316 | 0.24305 | 0.96457 |
| SLABIM universal BIM-direct | 108 | 0.06263 | 0.10334 | 0.24761 | 0.96320 |
| SLABIM learned refiner | 108 | **0.05601** | **0.09210** | **0.22725** | **0.97759** |
| Area_1 DA3 cached canonical（历史误标 raw） | 1641 | 0.30123 | 0.67323 | 0.83485 | 0.26437 |
| Area_1 raw DA3 metric（focal-corrected） | 1641 | 0.08443 | 0.16107 | 0.33620 | 0.94065 |
| Area_1 universal scale | 1641 | 0.07752 | 0.13938 | 0.31541 | 0.93710 |
| Area_1 universal BIM-direct | 1641 | 0.07815 | 0.13891 | 0.31350 | 0.93740 |
| Area_1 learned refiner | 1641 | **0.06689** | **0.11761** | **0.30823** | **0.94295** |

相对同一 `universal BIM-direct`，学习模型在 SLABIM/Area_1 test 的 AbsRel 分别改善
`10.56%/14.41%`。Area_1 家具子集也改善；BIM 冲突子集的点估计改善，但其房间级 AbsRel
bootstrap 95% CI 跨 0，不能宣称该子集已有显著优势。机器可读汇总、逐帧结果和训练审计见
[results/](results/)。

focal correction 不读取 GT：SLABIM 的 504 px focal 固定为 252 px，系数为 0.84；Area_1
则必须使用逐帧 resize 后内参，test 系数范围为 1.09477--2.02612。旧 scale/BIM/learned
checkpoint 仍以 canonical cache 为输入，不能只改输入后直接复用；其尺度阶段已经学习/估计了
米制恢复。当前 Area_1 全深度发布版已经完成一致迁移并从头重训。独立审计入口为
`scripts/analysis/audit_da3_focal_scaling.py`。

SLABIM 的 108 帧三维融合评测也优于 direct：Chamfer-L1 0.09109 m vs. 0.10515 m，
F-score@10 cm 0.79622 vs. 0.74003。

### Area_1 冻结 DA3 feature 与混合残差历史候选

在不更新 DA3 权重的前提下，最新候选缓存并融合 DA3 Metric-Large 第 11/23 层 encoder
token；任务网络仍从头按 scale/refiner/joint 三阶段训练。checkpoint 只由 validation 最终
AbsRel 选择。在完全相同的 room split、`0.2–5.0 m` support 和 pixel-micro evaluator 下：

| Split / 输出 | AbsRel | MAE (m) | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|
| Validation，上一版 scratch final | 0.06387 | 0.12881 | 0.32380 | 0.94435 |
| Validation，DA3-feature final | 0.06209 | 0.12401 | **0.30842** | 0.94615 |
| Validation，hybrid 比例分支 | 0.06131 | 0.12391 | 0.31444 | **0.95080** |
| Validation，hybrid 比例+加法 | **0.06131** | **0.12390** | 0.31443 | **0.95080** |
| Test，上一版 scratch final | 0.06442 | 0.11498 | 0.29802 | 0.94380 |
| Test，DA3-feature final | 0.06244 | **0.10780** | **0.28166** | 0.94391 |
| Test，hybrid 比例分支 | 0.06242 | 0.10961 | 0.28668 | 0.94520 |
| Test，hybrid 比例+加法 | **0.06235** | 0.10953 | 0.28663 | **0.94525** |

hybrid 将 DA3 token 只送入 refiner，尺度头退回无 token 版本，并学习
`D_scale*exp(r_prop)+Delta_D_add`。其 test AbsRel 相对上一版 scratch、公开 refiner 和
BIM-direct 分别下降 `3.21%/6.78%/20.21%`，相对 DA3-feature final 只下降 `0.14%`，同时
MAE/RMSE 较后者更差。更重要的是，加法头相对同检查点比例分支只改善 validation/test
AbsRel `0.007%/0.119%`，没有实质证据表明它值得增加复杂度；主要收益来自 feature 路由与
联合训练。Area_1 test 在这些迭代前已经揭盲，本表属于 post-hoc 诊断；公开主表暂不替换，
后续应在新区域/数据集上盲测确认。结构、训练轨迹、负结果、哈希和完整结果见
[attention-scale 实验记录](docs/ATTENTIVE_SCALE_EXPERIMENT.md)。

该 `0.2–5.0 m` 研究线按模型复杂度和多指标表现回退到**无加法的 DA3-feature 版本**；
hybrid 只保留为负诊断。对该无加法 checkpoint 的顺序消融显示，validation/test 上
`scale → +r_low → +r_detail` 的 AbsRel 分别为
`0.06960→0.06217→0.06209` 和 `0.07144→0.06285→0.06244`。`r_low` 占尺度后总改善的
98.98%/95.44%，`r_detail` 只占 1.02%/4.56%，属于小幅补充而非主要来源。

### Area_1 hit-only BIM prior 负结果

另按“固定围护结构只要射线正向命中即有效”的规则重制了 BIM prior：保留
door/window，仍排除 furniture/proxy/MEP，且不再用 `0.2–5.0 m` 截断 BIM；GT、loss 与
评测 support 仍为 `0.2–5.0 m`。BIM 覆盖率由 88.82% 增至 99.94%，但新增命中并不等价于
可靠对应。冻结 DA3、保持同一 3/9/3 从头训练后：

| Split / 输出 | 旧 bounded-core prior | 新 hit-only prior | 相对变化 |
|---|---:|---:|---:|
| Validation BIM-direct AbsRel | 0.08710 | 0.12411 | +42.49% |
| Validation learned final AbsRel | **0.06209** | 0.06306 | +1.56% |
| Test BIM-direct AbsRel | **0.07815** | 0.10745 | +37.50% |
| Test learned final AbsRel | **0.06244** | 0.06585 | +5.47% |

新 learned final 仍比自身 BIM-direct 在 validation/test 上低 49.19%/38.71%，说明注意力
尺度与 low/detail refiner 能恢复大部分污染；但它没有超过旧 prior，test 也仅 2/7 房间改善。
跨协议 room-bootstrap 95% CI 均跨 0，因此该差异尚不能提升为房间层面的显著结论。项目不把
hit-only 规则晋升为推荐协议，而将其保留为“覆盖率不等于先验精度”的可复现负诊断。完整
stage 指标、命中分布、命令与哈希见
[attention-scale 实验记录](docs/ATTENTIVE_SCALE_EXPERIMENT.md#unbounded-hit-only-bim-prior-retraining)。

### Area_1 三轮尺度条件 attention 官方全深度模型（当前推荐）

当前 Area_1 模型保持无 cross-attention、无加法头的 `scale + r_low + r_detail`，但把静态
scale attention 改为从 `s=1` 开始的三轮共享权重迭代。每轮根据当前 BIM/DA3 ratio residual
重算可靠性；三轮仅步长独立。每帧缓存深度先乘
`mean(fx,fy)/300`，再进入 BIM 比值、尺度头、loss 与 refiner；全流程不读取 test GT 定尺度。
GT 使用官方 regular depth 的全部有效值，仅排除原始 `0` 和 `65535`，输出上限为 128 m。
human epoch 12 仅由 validation final AbsRel 选中，pixel-micro 为：

| Split | Raw DA3 | BIM-direct | Static scale/final | Round 1 | Round 2 | Round 3 | + low | + detail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.09070 | 0.11815 | 0.07531 / 0.06445 | 0.07607 | 0.07093 | 0.06985 | 0.06476 | **0.06421** |
| Test | 0.08545 | 0.11072 | 0.07847 / 0.06614 | 0.07360 | 0.06972 | 0.06889 | 0.06351 | **0.06305** |

非学习分支已重新扫描 q05--q95。validation pixel-micro 下，纯 scale 与完整
consistency-gated/Gaussian BIM-direct 都在 q45 最优，分别为 `0.10568/0.10426`；但冻结到 test
为 `0.11422/0.11437`。当前跨域 robust log-cap 规则的 scale/direct test 为
`0.11033/0.11072`。它们都差于 corrected raw DA3 `0.08545`，因此 focal 修正后 BIM-direct
只作为确定性比较器，不再宣称能增强 standalone DA3。train-only q52 在 test 退化到
`0.12512`，已判定为跨房间过拟合。

相对静态 attention，三轮 scale 在 validation/test 改善 7.25%/12.21%，final 改善
0.36%/4.68%。scale 的 paired-room 95% CI 在两个 split 都不跨 0；final 的 validation CI
跨 0，而 test CI 为 `[-0.00914,-0.00035]`。家具子集仍退化，不能声称所有场景都改善。
Area_1 test 早已揭盲，因此 test 仍只能标为 post-hoc，不能包装成新的 blind claim。

复现配置为
`configs/stanford_area1_iterative_scale_3round_full_depth_metric_da3.yaml`，checkpoint 为
`outputs/stanford_area1_iterative_scale_3round_full_depth_metric_da3/accepted.pt`（SHA256
`74f2797dc42a4e7e8359440ea9d305a073e7ec0d2fe0850fb1ab79877bb7ae6d`）。静态 attention 与旧
canonical-input 全深度权重只保留作审计。

后续 reliability-gated 版本加入 attention-token 监督、RGB-aware BIM adapter gate 和
`r_detail` 可靠性门，但 final val/test AbsRel 为 `0.06928/0.06884`，较发布模型退化
`0.98%/2.12%`，因此已回退且该版本不发布。配置、命令、MAE/RMSE、负结果与审计哈希见
[完整记录](docs/ATTENTIVE_SCALE_EXPERIMENT.md#full-depth-supervised-retraining)。

为选择 residual 形式，另在 validation 的 685 万采样像素上检查尺度后误差。无 DA3-feature
attention scale 的 `mean |GT-scaled|` 随深度的幂指数为 `0.404`（纯加法为 0、纯比例为 1）；
混合模型约为 `0.0566 m + 0.0402×depth`，拟合明显更好。BIM-consistent/furniture/no-hit 的
指数分别为 `0.328/0.762/1.061`，说明不能把全体像素统一假设为固定米制或固定百分比误差。
完整分箱、重尾统计和图件见[同一实验记录](docs/ATTENTIVE_SCALE_EXPERIMENT.md#pixel-residual-distribution-after-scale-correction)。

### Area_1 pano 无训练联合评测

> 焦距审计后的状态：下述 pano 历史实验同样读取 canonical cache，并曾将其误称为 raw metric
> DA3。由于不同 regular/tangent 视图的 focal factor 不同，相关 absolute raw 数值和相对 raw
> 改善暂时只保留为历史诊断，不能进入修正后的论文主结果；需要按每个视图的处理分辨率内参重跑。

全景实验单独采用 exact ERP solid-angle、equal-station macro，不与上表的 pixel-micro 混合。
融合方法按 validation 的 **regular-only raw DA3** 目标选择一次，冻结为 `joint_huber`，随后
一次性评测 31 个 test station / 7 个房间。主结果不使用 learned refiner；表中
`regular-only` 指同站多张 regular 的融合，并非单张图：

| Test 输入 / 方法 | AbsRel | MAE (m) | 球面覆盖率 |
|---|---:|---:|---:|
| regular-only raw DA3 | 0.26985 | 0.59144 | 66.92% |
| regular + tangent6 raw DA3 | 0.23501 | 0.51684 | 99.75% |
| regular + tangent14 raw DA3 | **0.22865** | **0.48947** | **99.96%** |
| regular universal scale | 0.10824 | 0.18045 | 66.92% |
| regular BIM-direct | 0.11006 | 0.18276 | 66.92% |

在完全相同的 regular support 上，加入 pano-derived tangent14 后 AbsRel 下降 15.27%，7/7
房间均改善，room-cluster bootstrap 95% CI（candidate−reference）为
`[-0.05293,-0.03021]`。BIM 提供的统一尺度使 raw AbsRel 下降 59.89%，但进一步的局部
BIM-direct 比 scale-only 退化 1.68% 且 CI 跨 0；这是保留的负消融，不宣称 local correction
有效。完整结果、严格单图负结果和 view-count 消融见[全景评测文档](docs/PANO_DEPTH_EVALUATION.md)。

直接“同一单图 + pano tangent”的补充实验仅在 validation 上执行：AbsRel 从单图的 0.11586
变为 +tangent6 的 0.16454、+tangent14 的 0.31859；覆盖率虽从 10.87% 增至
99.38%/99.85%，但相同单图 support 上精度显著变差。因此当前可确认的是 pano 对
**多-regular 基线**的增益与整球覆盖价值，不能声称朴素 pano 融合优于单图。下一版需先解决
跨切平面尺度/上下文漂移，并按 pano 联合目标预注册融合规则。

上面的 strict-single 实验并不直接对应 regular-view benchmark。为回答“全景联合后，原始
regular 图是否真的更准”，项目另提供更严格的 **regular round-trip** validation 协议：同站
全部 regular 的 DA3 z-depth 先依据原始内外参转成 ERP radial range，在球面上联合；可选加入
从原始 pano RGB 产生的 6/14 个 tangent 预测；最终把球面深度反投影回每一张原始 regular，
并在该帧原有的完整 0.2–5.0 m GT mask 上评测。

| Area_1 validation，原始 regular 支持域 | Pixel-micro AbsRel | 相对 raw |
|---|---:|---:|
| 原始逐帧 DA3 | 0.27710 | reference |
| regular-only ERP joint（最佳 regular-only：Huber） | 0.25954 | −6.34% |
| regular + tangent6（weighted-log） | 0.22399 | −19.17% |
| **regular + tangent14（weighted-log）** | **0.18654** | **−32.68%** |

在相同 weighted-log 融合器下，tangent14 相对 regular-only 从 0.26682 降至 0.18654，下降
30.09%；7/7 房间改善，room-cluster paired 95% CI 为
`[-0.09133,-0.07649]`。单帧投影再回投的恒等控制仅改善 0.25%，说明主要收益不是插值伪影。
该结果只用于 validation 方法设计，尚未在新的盲测集上确认；不能事后替换已经揭盲的旧 test
协议。机器结果见
[`pano_val_regular_roundtrip/summary.json`](results/stanford_area1/pano_val_regular_roundtrip/summary.json)。

为避免把 BIM 的尺度恢复收益误算成 ERP 联合收益，又做了一个**同基线**实验：比较每张 regular
先独立执行 universal BIM scale，与这些完全相同的 scaled depth 经同站
`regular→ERP joint Huber→regular` 后的结果。两侧使用相同 1,673 帧和 364,913,264 个
regular GT 像素，不读取 pano RGB/GT，不使用 tangent、checkpoint 或 learned refiner：

| Area_1 validation，同一 BIM-scale 基线 | Pixel-micro AbsRel | MAE (m) | δ1 |
|---|---:|---:|---:|
| 每帧 regular universal scale | 0.08644 | 0.17422 | 0.91533 |
| 投影→回投控制 | 0.08608 | 0.17336 | 0.91543 |
| **同站 ERP joint Huber 后回投** | **0.07595** | **0.15254** | **0.93156** |
| + overlap residual calibration（sync-Huber） | 0.07446 | 0.15887 | 0.94402 |

主对比 AbsRel 下降 12.14%，7/7 房间改善，room-cluster paired 95% CI 为
`[-0.01231,-0.00700]`。扣除单纯投影/插值控制后，AbsRel 仍下降约 11.77%。因此在目前的
validation 上，结论是：**BIM scale 后的多-regular ERP 联合确实优于同一逐帧 BIM-scale
基线**；这是 validation-only 证据，尚不能替换 blind-test 主表。机器结果见
[`pano_val_bim_scale_regular_roundtrip/summary.json`](results/stanford_area1/pano_val_bim_scale_regular_roundtrip/summary.json)。

残差校准的点估计并非全面更好：相对普通 Huber，AbsRel 再降 1.96%、RMSE 降 5.29%、δ1
提高 1.25 个百分点，但 MAE 增加 4.15%，room-macro AbsRel 增加 0.93%；仅 827/1673 帧、
14/30 站、4/7 房间改善，room-cluster AbsRel 差值 95% CI `[−0.00682,0.00408]` 跨 0。
因此普通 Huber 仍是默认方法，残差校准只保留为敏感性分析。

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
  --acknowledge-bimsyn-license \
  --include-pano

.venv/bin/python scripts/data/verify_stanford_bimsyn_sources.py \
  --area-root ../Stanford2D3DS/no_xyz \
  --area-tar ../Stanford2D3DS/no_xyz/area_1_no_xyz.tar \
  --ifc-root ../BIMSyn/BIM_model/ifc \
  --require-pano \
  --output data/provenance/stanford_area1_sources.local.json
```

`--include-pano` 会额外解出全景 RGB/depth/pose/semantic，`--require-pano` 会在来源回执中
严格核验它们；只复现 regular-view 任务时可同时省略这两个开关。IFC 已足够计算；RVT 仅
用于来源审计，可用 `--include-rvt` 下载。随后缓存 DA3、制备 10,327 帧，并按 room 严格
划分 `7013/1673/1641`：

```bash
.venv/bin/python scripts/data/cache_stanford_da3.py \
  --config configs/stanford_area1_transfer.yaml
.venv/bin/python scripts/data/prepare_stanford_area1.py \
  --config configs/stanford_area1_transfer.yaml
```

可选的 pano 路径先把 ERP RGB 切成冻结的 `nested14` tangent views，并缓存同一 pinned DA3；
该缓存阶段不读取 pano depth。脚本结束时会打印不可变 manifest 的实际路径：

```bash
.venv/bin/python scripts/data/cache_stanford_pano_da3.py \
  --config configs/stanford_area1.yaml \
  --split val --preset nested14 --face-resolution 504 --log-every 1

.venv/bin/python scripts/model/evaluate_stanford_pano.py \
  --config configs/stanford_area1.yaml \
  --split val --output outputs/stanford_area1/pano_val_training_free \
  --device cuda --batch-size 8 --pano-height 512 \
  --bootstrap-repetitions 10000 --seed 42

# 推荐：联合后回投到全部原始 regular，并在原始 regular GT mask 上评测
.venv/bin/python scripts/analysis/evaluate_stanford_pano_regular_roundtrip.py \
  --config configs/stanford_area1.yaml \
  --tangent-manifest data/processed/stanford_area1_504/pano_da3/nested14_r504_737a5fa1b07a/manifests/val_full.json \
  --output outputs/stanford_area1/pano_val_regular_roundtrip_reproduction
```

上面的 evaluator 命令执行 regular-to-pano Route R；加入
`--tangent-manifest <缓存脚本打印的 val_full.json>` 才启用完整球面 Route P 和
regular+pano 分析。test 的缓存与评测都必须额外带 `--confirm-test`，并且只能在 validation
协议冻结后执行。fresh-data 运行若生成本机 config，应把两条命令中的 config 替换为对应
本机路径。当前 pano 主协议是无训练评测，因此命令故意不传 checkpoint；只有复核
归档的 learned 分支时才可选传入 `--checkpoint`，它不参与 pano 方法选择与主结论。

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
| `stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml` | 旧 canonical-input 全深度基配置与审计模型 |
| `stanford_area1_attentive_scale_da3_features_hit_only_full_depth_metric_da3.yaml` | Area_1 冻结静态 attention 全深度对照 |
| `stanford_area1_iterative_scale_3round_full_depth_metric_da3.yaml` | **Area_1 当前推荐的三轮 scale-conditioned attention 模型** |
| `stanford_area1_reliability_gated_full_depth.yaml` | 已回退、禁止发布的负实验配置 |

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

三份发布 checkpoint 的角色、大小和 SHA-256 见 [results/manifest.json](results/manifest.json)。大型
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
- [可学习 attention scale 与冻结 DA3 feature 实验](docs/ATTENTIVE_SCALE_EXPERIMENT.md)
- [BIM early-fusion DAv2 dense metric-depth 实验](docs/BIM_EARLY_FUSION_DENSE_EXPERIMENT.md)
- [Area_1 全景深度联合估计与 BIM 增强评测](docs/PANO_DEPTH_EVALUATION.md)
- [Matterport3D + BIMNet 零样本尺度评测](docs/MATTERPORT_BIMNET_ZERO_SHOT.md)
- [Area_1 全景论文/PPT 图件与三组定性备选](docs/assets/pano_evaluation/README.md)
- [科研评测协议、消融与敏感性设计](docs/EVALUATION_PROTOCOL.md)
- [非学习 BIM-direct 逐因素消融](docs/DETERMINISTIC_BASELINE_ABLATION.md)
- [论文与 PPT 素材目录](docs/assets/paper_evaluation/README.md)
- [Codex 对话要点与项目演化](docs/CODEX_WORKFLOW.md)

## 许可与研究边界

- 第三方数据和 DA3 权重不随本仓库的软件许可重新授权；用户必须遵守各发布者条款。
- SLABIM 位姿由 rosbag/官方 SLAM 点云离线 ICP 恢复，不应表述为无配准系统。
- Area_1 的固定 BIM→Area 变换由发布的 semantic structural mesh 辅助估计，属于
  scan-calibrated/oracle-style registration；实际部署应替换为测量或定位系统提供的变换。
- 软件仓库在公开发布前仍需由所有者选择并添加 `LICENSE`。没有 LICENSE 时，默认并不等于
  允许他人复制、修改或再分发代码。
