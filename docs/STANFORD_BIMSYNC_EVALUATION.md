# 2D-3D-S Area 1 + BIMSyn 适配与评测

本文记录 Area 1 的公开复现协议。旧 universal `0.2–5.0 m` 链与 SLABIM 完全一致；当前
推荐发布模型另外采用 Area_1 官方全部有效深度、逐帧 focal-corrected DA3、hit-only BIM、
attention scale 和冻结 DA3 feature。两条结果的支持域必须明确区分。

## 1. 数据来源与许可

- 2D-3D-S 官方项目：<https://github.com/alexsax/2D-3D-Semantics>
- Area 1 `no_xyz`：RGB、regular-view depth、pose、语义网格；下载前需接受官方许可。
- BIMSyn 论文：<https://doi.org/10.1016/j.autcon.2023.105076>
- BIMSyn 发布目录提供与 Area 1 的 44 个房间对应的 IFC/RVT；计算仅需要 IFC，RVT 可选。

第三方数据不由本项目的软件许可证重新授权，也不随仓库分发。下载器要求用户显式确认
已经阅读并接受提供方条款。

## 2. 数据规模和坐标约定

- 10,327 个 1080×1080 regular views；
- 44 个房间、186 个 camera UUID；
- depth 为 z-depth：`raw / 512 m`，`65535` 无效；
- pose JSON 的 `camera_rt_matrix` 实际为 3×4 世界到相机 `[R|t]`；
- 每帧内参随视角变化，不能写死。

IFC 以毫米为单位。44 个房间按冻结的 `T_area_from_bim` 合并成一个 Area 1 全局 BIM，
每帧只依据公开 pose 渲染，不做逐帧 ICP 或 GT 深度调参。需要区分两份不可混用的 prior：

- 正式旧 benchmark 使用 bounded core envelope：只保留 wall、floor、ceiling、column、beam，
  排除 furniture/proxy/MEP/door/window，并以 0.2–5.0 m 限定 BIM 命中；
- hit-only prior 保留 door/window，仍排除 furniture/proxy/MEP，只要射线获得有限正值命中
  就有效，不以距离过滤 BIM。它在 `0.2–5.0 m` 监督实验中是负结果，但与官方全深度监督配对
  后用于当前推荐发布模型。

二者使用不同 processed root、manifest 和 preparation fingerprint。旧正式结果不会被新版
制备覆盖。

冻结配准 receipt：
`data/provenance/stanford_area1_bimsyn_alignment.json`。该变换由 Area 1 语义结构网格辅助
标定，因此论文中必须称为 **scan-calibrated/oracle-style registration protocol**；工程
部署应由测量或定位系统提供等价变换。

## 3. 下载、验证与制备

```bash
python scripts/data/download_stanford_area1.py \
  --stanford-root ../Stanford2D3DS \
  --bimsyn-root ../BIMSyn \
  --accept-stanford-license --accept-bimsyn-terms

python scripts/data/verify_stanford_bimsyn_sources.py \
  --area-root ../Stanford2D3DS/no_xyz \
  --area-tar ../Stanford2D3DS/no_xyz/area_1_no_xyz.tar \
  --ifc-root ../BIMSyn/BIM_model/ifc \
  --output data/provenance/stanford_area1_sources.json

python scripts/data/cache_stanford_da3.py \
  --config configs/stanford_area1_transfer.yaml
python scripts/data/prepare_stanford_area1.py \
  --config configs/stanford_area1_transfer.yaml --overwrite
python scripts/data/build_stanford_split.py \
  --config configs/stanford_area1_transfer.yaml
```

下载器用固定 manifest 校验 44 个 IFC（可选再校验 44 个 RVT），并核对 Area 1 四模态
basename 与固定 OBJ/MTL/labels hash。完整 fresh-data 命令和 runtime config 物化见
[DATA_PREPARATION.md](DATA_PREPARATION.md)。

## 4. 划分和防泄漏

固定 seed 42，按房间整组划分：

| split | 房间 | 帧 |
|---|---:|---:|
| train | 30 | 7013 |
| validation | 7 | 1673 |
| test | 7 | 1641 |

同一房间和 camera UUID 不跨 split。尺度 cap 只读取 7013 个训练样本；receipt 明确记录
validation/test opened=0。模型和超参数在 validation 冻结后才运行一次 test。

## 5. 通用尺度与模型

Area 1 不再使用数据集专属的直接修正方法。与 SLABIM 相同：

```text
s = exp(min(Q45(log(BIM/DA3)), Q25(log(BIM/DA3)) + 0.05))
anchor = s * DA3
prediction = anchor * exp(bounded learned residual)
```

`universal_bim_direct` 在同一 anchor 上应用固定局部 BIM correction，只作为强确定性基线
与验收 comparator。参数来自 Area 1 train-only 的固定 48 候选选择，之后作为跨数据集
协议冻结；SLABIM validation 只作诊断，没有反向改参数。详见
[UNIVERSAL_SCALE_PROTOCOL.md](UNIVERSAL_SCALE_PROTOCOL.md)。

## 6. 分层评测

所有方法使用完全相同的 0.2–5.0 m GT support，并报告：

- all、furniture、structural、BIM-foreground conflict、BIM-consistent、BIM-no-hit；
- pixel-micro、frame-macro、room-macro；
- AbsRel、MAE、RMSE、δ；
- 10,000 次 paired room bootstrap；
- raw BIM envelope 单独报告 coverage，不与稠密方法共用可变支持。

任何 support 内的非有限或非正预测会直接报错，而不是从分母中删除。

## 7. 正式结果

Area 1 blind test（1641 帧，pixel-micro）：

| 方法 | AbsRel | MAE (m) | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|
| Raw DA3 | 0.30123 | 0.67323 | 0.83485 | 0.26437 |
| Universal scale | 0.07752 | 0.13938 | 0.31541 | 0.93710 |
| Universal BIM direct | 0.07815 | 0.13891 | 0.31350 | 0.93740 |
| Learned refiner | **0.06689** | **0.11761** | **0.30823** | **0.94295** |

学习方法相对 direct 的 AbsRel 降低 14.41%。all 的 room bootstrap AbsRel 差为
-0.01323，95% CI [-0.01754, -0.00822]，7/7 房间改善。家具也改善；conflict 像素点估计
改善，但其房间级 AbsRel CI [-0.00692, +0.00064] 跨 0，不能声称房间层面显著优越。

正式机器结果：

- [validation summary](../results/stanford_area1/val_summary.json)
- [test summary](../results/stanford_area1/test_summary.json)
- [compact result index](../results/README.md)

### 7.1 Hit-only prior 重训练诊断

使用与冻结 DA3-feature 候选相同的网络、房间划分和 3/9/3 从头训练策略，只替换 BIM prior
规则。全 10,327 帧的 RGB、DA3、GT 与 GT mask 逐数组一致。BIM coverage 从 88.82% 增至
99.94%，其中新 prior 有效像素的 4.82% 大于 5 m、0.75% 小于 0.2 m。

| Split / 方法 | 旧 bounded core | 新 hit-only |
|---|---:|---:|
| validation BIM-direct AbsRel | 0.08710 | 0.12411 |
| validation learned final AbsRel | **0.06209** | 0.06306 |
| test BIM-direct AbsRel | **0.07815** | 0.10745 |
| test learned final AbsRel | **0.06244** | 0.06585 |

结论是负面的：更稠密的命中降低了对应精度，非学习 direct 尤其明显。learned final 仍比新
BIM-direct 在 validation/test 上改善 49.19%/38.71%，但相对旧 learned final 退化
1.56%/5.47%。跨协议 room-bootstrap CI 均跨 0，只有 2/7 validation 和 2/7 test 房间改善。
因此“只更换 prior、仍限制 `0.2–5.0 m`”的实验仅保留作诊断；它不否定下一节采用全深度
监督的发布模型。详见
[ATTENTIVE_SCALE_EXPERIMENT.md](ATTENTIVE_SCALE_EXPERIMENT.md#unbounded-hit-only-bim-prior-retraining)。

### 7.2 当前发布模型：focal-corrected 官方全深度 attention-scale

当前 Area_1 推荐 checkpoint 使用同一房间划分和 hit-only prior，从头训练冻结 DA3
layer-11/layer-23 feature、三轮 scale-conditioned attention 与 low/detail refiner。GT 支持域为官方
regular-view z-depth 的全部正值，只排除原始 `0/65535`；模型输出上限为 128 m。缓存 DA3
先逐帧乘 `mean(fx,fy)/300`，然后才进入 BIM ratio、尺度头和 refiner。

| Split / 输出 | Raw DA3 | BIM-direct | Round 1 | Round 2 | Round 3 | + low | + detail |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation AbsRel | 0.09070 | 0.11815 | 0.07607 | 0.07093 | 0.06985 | 0.06476 | **0.06421** |
| Test AbsRel | 0.08545 | 0.11072 | 0.07360 | 0.06972 | 0.06889 | 0.06351 | **0.06305** |

发布 checkpoint 为
`outputs/stanford_area1_iterative_scale_3round_full_depth_metric_da3/accepted.pt`，SHA256
为 `74f2797dc42a4e7e8359440ea9d305a073e7ec0d2fe0850fb1ab79877bb7ae6d`。此前
reliability-gated 模型在相同 benchmark 上得到 `0.06928/0.06884`，因此已回退且不发布。
checkpoint 可公开复现和部署，但 Area_1 test 此前已经揭盲，test 数字必须标记为 post-hoc。

## 8. 训练与评测命令

```bash
python scripts/model/train.py \
  --config configs/stanford_area1.yaml \
  --init-checkpoint outputs/slabim/accepted.pt \
  --allow-cross-dataset-initialization --device cuda

python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1.yaml \
  --checkpoint outputs/stanford_area1/accepted.pt \
  --split val --output outputs/stanford_area1/formal_val \
  --batch-size 8 --bootstrap-repetitions 10000 \
  --bootstrap-seed 42 --inference-seed 42 --device cuda

python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1.yaml \
  --checkpoint outputs/stanford_area1/accepted.pt \
  --split test --output outputs/stanford_area1/formal_test \
  --batch-size 8 --bootstrap-repetitions 10000 \
  --bootstrap-seed 42 --inference-seed 42 --device cuda
```

上面的 universal 同数据集 checkpoint 不加 `--cross-dataset`，不使用 test 选择模型，也不
需要 `--allow-unverified-robust-comparator`。

官方全深度发布模型使用其专用配置；由于旧 robust comparator receipt 绑定到 bounded
manifest，评测器要求显式记录 comparator 为 unverified，但不会关闭数据或 checkpoint
provenance：

```bash
python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth_metric_da3.yaml \
  --device cuda

python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth_metric_da3.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth_metric_da3/accepted.pt \
  --split val --depth-support all-valid \
  --output results/stanford_area1/attentive_scale_da3_features_hit_only_full_depth_metric_da3_val \
  --batch-size 8 --bootstrap-repetitions 10000 \
  --bootstrap-seed 42 --inference-seed 42 --device cuda \
  --allow-unverified-robust-comparator
```
