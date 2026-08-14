# 2D-3D-S Area 1 + BIMSyn 适配与评测

本文记录 Area 1 的公开复现协议。模型方法与 SLABIM 完全一致；差异只在原始数据、BIM
来源、坐标配准和房间隔离划分。

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

IFC 以毫米为单位。围护先验只保留 wall、floor、ceiling、column、beam；排除家具、proxy、
MEP、door 和 window。44 个房间按冻结的 `T_area_from_bim` 合并成一个 Area 1 全局 BIM，
每帧只依据公开 pose 渲染，不做逐帧 ICP 或 GT 深度调参。

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

不要为同数据集 checkpoint 加 `--cross-dataset`，不要用 test 选择模型，也不要在正式结果
中使用 `--allow-unverified-robust-comparator`。
