# Matterport3D + BIMNet 零样本尺度评测

## 1. 目的与边界

本协议检验在 Stanford 2D-3D-S Area_1 上训练的 scale estimator，能否在
Matterport3D 与 BIMNet 的新场景上直接迁移，并对比 full regression、fixed-attention
pseudo-Huber 和 iterative-attention pseudo-Huber。首个场景为 Matterport
`HxpKQynjfin` / BIMNet `train/hxp`。不使用 Matterport 图像或深度训练、微调、选 checkpoint
或调整网络参数。

本实验只评测 learned frame scale：

\[
D_{\mathrm{pred}}=s_{\mathrm{learned}}D_{\mathrm{DA3}}.
\]

不运行 low/detail refiner。冻结的尺度网络输入为 8 通道：RGB 3 通道、焦距修正后的 raw DA3
深度、BIM 深度、BIM hit mask、BIM/DA3 signed log disagreement 及其绝对值。网络从
`log(s_0)=0` 开始，使用共享 updater 迭代 3 轮。

## 2. 数据、配准与 BIM 先验

- Matterport RGB-D 和相机由 `S3-SAM3D-ToolKit` 的 `Matterport3DDataset` 解析；
- BIMNet `hxp` 自动映射到 Matterport scan `HxpKQynjfin`；
- BIMNet wall-filled component OBJ 通过 `inverse(mat_pc2obj)` 变换到 Matterport world；
- BIM 深度使用同一相机内外参射线投影到 DA3 的 `406 x 504` 处理尺寸；
- `hxp` 的 46 个 BIM 构件仅包含墙、门、窗、楼板和 covering，没有家具设备；
- DA3 canonical 输出乘以 `mean(fx_processed, fy_processed) / 300` 转为 metric depth。

`HxpKQynjfin` 包含 44 个采集位置，每个位置 18 个 perspective frames，共 792 帧。

## 3. 为什么不能只报告未筛选全集

该场景同时存在两种与算法无关的问题：

1. 部分 Matterport depth PNG 为全零或有效深度极少；
2. BIMNet 只建模了建筑的一部分，例如部分阳台相机位于 BIM 包围范围外。

把这些帧全部纳入 BIM-prior 方法，会把“GT 损坏 / BIM 根本不存在”混入尺度估计误差。反过来，
如果按最终预测误差挑帧，又会造成有利于模型的 selection bias。因此评测器采用预先声明的几何与
GT/BIM 可见表面规则，并同时报告未筛选全集和拒绝集。

## 4. GT-assisted benchmark 筛选

> **当前正式有效帧定义（后续澄清）**：只使用三条、与模型预测无关的规则：GT 正深度
> 比例 `>10%`、BIM ray-hit 比例 `>20%`、相机中心位于**严格 BIM AABB** 内。AABB margin
> 固定为 0，不再使用 BIM/GT agreement 或 BIM/DA3 ratio support。按该定义得到 624 帧；
> 本节以下旧子集只保留为历史诊断，不能再把旧 `gt_quality` 或 `gt_verified` 称作当前正式结果。
> 四种已完成的 zero-shot 预测按新规则的统一汇总见[第 9 节](#9-三规则正式有效帧汇总)。

GT/BIM 一致性诊断子集 `gt_verified`（兼容旧名 `effective`）同时满足：

1. GT 正深度像素占整图至少 10%；
2. 相机中心位于 BIM axis-aligned bounding box 的 0.25 m 扩展范围内；
3. BIM ray hit 占处理图像至少 20%；
4. 至少整图 10% 的像素同时有 GT/BIM，且满足

   \[
   |D_{\mathrm{BIM}}-D_{\mathrm{GT}}|
   \leq \max(0.10\ \mathrm{m},0.05D_{\mathrm{GT}});
   \]

5. 网络定义的有效 BIM/DA3 ratio support 不少于 100 像素。

这里第 1、4 条使用 GT，所以该规则只能用于说明 benchmark 是否合理，**不能作为部署时的
test-time selector，也不作为论文 headline**。本项目将不依赖 BIM/GT 一致性的 `gt_quality`
作为主结果，并报告仅使用相机/BIM/模型输入信息的 `operational_no_gt` 子集。逐帧 CSV 记录每条
筛除原因，`summary.json` 报告：

- `all_gt_valid`：只要求 GT 至少存在一个正深度像素；
- `gt_quality`：只应用 10% GT 有效率，作为主 benchmark；
- `operational_no_gt`：GT-quality 计分帧中，只按 AABB、BIM hit 和 ratio support 筛选，
  不读取 BIM/GT agreement；
- `gt_verified`：再要求 BIM/GT 一致的诊断上限；
- `bim_applicable`、`effective`：为兼容旧分析保留的 GT-assisted 名称；
- `rejected_from_effective`：可计分但不满足主有效集的帧。

此外，`summary.json` 固定输出 hit threshold `{0.10, 0.20, 0.30, 0.50}` 与 agreement
threshold `{0.05, 0.10, 0.20}` 的 12 组敏感性结果，避免结论依赖单一阈值。

## 5. 复现命令

```bash
.venv/bin/python scripts/model/evaluate_matterport_bimnet_full_regression.py \
  --matterport-root /path/to/Matterport3D \
  --bimnet-root /path/to/BIMNet_release \
  --toolkit-root /path/to/S3-SAM3D-ToolKit \
  --bimnet-scene hxp \
  --config configs/stanford_area1_full_regression_scale_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch_full_depth_metric_da3.yaml \
  --checkpoint outputs/stanford_area1_full_regression_scale_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch/accepted.pt \
  --output-dir results/matterport3d/hxp_full_regression_scale_zero_shot \
  --process-res 504 \
  --device cuda \
  --progress-every 100
```

复现 Huber 对照时保持其他参数不变，仅替换下列 config/checkpoint/output-dir：

| 结构 | config 前缀 | checkpoint 目录 | output-dir |
|---|---|---|---|
| Fixed Huber | `stanford_area1_fixed_attention_huber_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch_full_depth_metric_da3.yaml` | `stanford_area1_fixed_attention_huber_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch` | `hxp_fixed_attention_huber_zero_shot` |
| Iterative Huber | `stanford_area1_iterative_attention_huber_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch_full_depth_metric_da3.yaml` | `stanford_area1_iterative_attention_huber_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch` | `hxp_iterative_attention_huber_zero_shot` |

输出：

- `per_frame.csv`：逐帧筛选证据、三轮 log-scale、raw/learned/oracle 指标；
- `summary.json`：按子集的 pixel-micro 与 frame-macro 汇总、threshold sensitivity；
- `assets/`：代表性有效视角、BIM 缺失视角和室内但 BIM 不一致视角。

## 6. Full-regression 首场景结果

所有数字为全深度、pixel-micro 聚合；同一行的三种预测使用完全相同的 GT support。
`oracle frame scale` 仅作为 DA3 形状误差的诊断上限，不是可部署方法。

| 子集 | 帧数 | Raw DA3 AbsRel | Learned scale AbsRel | Oracle frame-scale AbsRel | Learned 相对改善 | 逐帧胜率 |
|---|---:|---:|---:|---:|---:|---:|
| all GT-valid | 777 | 0.16409 | 0.11377 | 0.06634 | 30.66% | 70.14% |
| GT quality | 740 | 0.16403 | 0.11360 | 0.06630 | 30.75% | 72.84% |
| operational no-GT | 620 | 0.16664 | 0.11133 | 0.06514 | 33.19% | 81.77% |
| GT-verified（诊断） | 550 | 0.16505 | 0.10492 | 0.06226 | 36.43% | 83.82% |
| rejected from effective | 227 | 0.15907 | 0.15981 | 0.08757 | -0.46% | 37.00% |

GT-verified 上 RMSE 从 0.33214 m 降至 0.26080 m，delta1 从 0.81441 增至
0.91875。尺度网络在未参与训练的 Matterport/BIMNet 场景上明显优于 raw DA3；但在 BIM
缺失或错配的拒绝集上没有收益，甚至轻微退化。因此不筛选直接报告 777 帧全集，会把方法本身的
zero-shot 能力与先验不可用混在一起；只报告 GT-verified 而隐去全集则会造成条件性偏优。论文
headline 应使用 GT-quality，并同时给出 all、operational no-GT 和 GT-verified 诊断结果。

### 6.1 数据与筛除统计

- 792 个唯一视角中，777 帧可计分，15 帧 GT 全零，推理/渲染错误为 0；
- 740 帧通过 GT 10% 有效率，620 帧通过 operational no-GT 筛选，550 帧通过
  GT/BIM 一致性诊断筛选；
- 筛除原因可重叠：`camera_outside_bim_aabb` 131 帧、`low_bim_gt_agreement` 227 帧、
  `low_bim_hit` 86 帧、`insufficient_bim_da3_ratio_support` 71 帧、`sparse_gt` 37 帧；
- 旧 raw DA3 CSV 与本次 777 个共同帧逐帧 AbsRel 的最大差为 `1.46e-5`，整体 AbsRel 均为
  `0.16409`，证明新评测没有重新遗漏 focal correction。

### 6.2 三轮尺度迭代

GT-verified 子集的 frame-oracle mean absolute log-scale error 随轮次为：

| round | 1 | 2 | 3 |
|---|---:|---:|---:|
| mean absolute log-scale error | 0.11418 | 0.09367 | **0.07602** |

round 2/3 的更新方向没有符号振荡。仍有 21.64% 的 GT-verified 帧在 round 3 比 round 2 离
oracle 更远，说明三轮整体有利但不是逐帧单调。拒绝集则从 `0.13848 → 0.14124 → 0.14610`
持续变差：网络在无可靠 BIM 证据时会反复强化错误更新，这与 rejected 子集无总体收益一致。

### 6.3 阈值敏感性

在已满足 AABB、GT quality 与 network support 的条件下，hit threshold 从 10% 提高到 50%
不改变入选集，说明 agreement 条件已经排除了低覆盖帧。BIM/GT agreement image fraction
阈值的结果为：

| agreement threshold | 帧数 | Raw AbsRel | Learned AbsRel | 相对改善 |
|---:|---:|---:|---:|---:|
| 5% | 572 | 0.16503 | 0.10674 | 35.32% |
| **10%** | **550** | **0.16505** | **0.10492** | **36.43%** |
| 20% | 520 | 0.16383 | 0.10215 | 37.65% |

三个阈值下结论一致。20% 得分最好但筛帧更严格，因此 GT-verified 诊断仍保留预先声明的 10%，
不按性能反选阈值；主结果始终是不用 agreement 筛选的 GT-quality。

### 6.4 示例素材

- `assets/filter_example_effective.png`：正常室内，BIM hit 100%，整图 agreement 56.5%；
- `assets/filter_example_outside_bim.png`：阳台/外部视角，虽有 87.5% 射线误命中延伸 BIM，
  但整图 agreement 仅 1.8%；
- `assets/filter_example_inside_aabb_mismatch.png`：相机仍在粗 AABB 内，BIM hit 100%，但
  BIM/GT agreement 为 0，说明只用 AABB 或 hit rate 都不足以判定 BIM 有效。

由此可见，`BIM hit fraction` 不能单独作为有效帧标准：未建模阳台仍可能命中模型外壳，而室内
错误区域甚至可达到 100% hit。GT-assisted 几何一致性是本次 benchmark 清洗中真正区分
“有 BIM”与“BIM 对当前视野正确”的关键。

## 7. 三种尺度结构的 zero-shot 对照

三种 checkpoint 都只在 Area_1 训练前三个 scale epochs，且严格使用相同的 8 通道输入、
三轮共享 updater、`c0=0`、无 DA3 latent feature、无 confidence、无 BIM normal/edge、无
deterministic scale 输入和无 fallback gate。区别仅为尺度更新机制：

- full regression：神经网络直接回归每轮 log-scale 增量；
- fixed Huber：在第 1 轮计算一次 attention，三轮复用，并由 pseudo-Huber 聚合 ratio；
- iterative Huber：每轮依据当前 ratio residual 重新计算 attention，再做 pseudo-Huber 聚合。

主结果使用不按 BIM/GT 一致性挑帧的 GT-quality 740 帧：

| 方法 | AbsRel | 相对 Raw 改善 | frame-oracle log-scale MAE | 优于 Raw 的帧占比 |
|---|---:|---:|---:|---:|
| Raw DA3 | 0.16403 | - | 0.13053 | - |
| Full regression | 0.11360 | 30.75% | 0.08865 | 72.84% |
| Fixed Huber | 0.10635 | 35.16% | 0.08085 | 75.95% |
| **Iterative Huber** | **0.10532** | **35.79%** | **0.07957** | **76.62%** |
| Oracle frame scale | 0.06630 | 59.58% | 0 | - |

不同筛选口径下，fixed 与 iterative 的结论保持一致：

| 子集 | 帧数 | Fixed Huber | Iterative Huber | Iterative 相对 Fixed 改善 |
|---|---:|---:|---:|---:|
| all GT-valid | 777 | 0.10654 | **0.10550** | 0.97% |
| GT quality（主结果） | 740 | 0.10635 | **0.10532** | 0.98% |
| operational no-GT | 620 | 0.10323 | **0.10222** | 0.99% |
| GT-verified（诊断） | 550 | 0.09745 | **0.09692** | 0.55% |

GT-quality 上的三轮 frame-oracle log-scale MAE 为：

| 方法 | round 1 | round 2 | round 3 |
|---|---:|---:|---:|
| Fixed Huber | 0.10336 | 0.08732 | 0.08085 |
| Iterative Huber | **0.10283** | **0.08629** | **0.07957** |

两者三轮更新均没有出现 update 符号翻转；iterative 在 57.70% 的逐帧配对和 79.55% 的采集位置
配对上优于 fixed。以 44 个 panorama/采集位置为 cluster 的 10,000 次 paired bootstrap，
`Iterative - Fixed` 的 pixel-micro AbsRel 差值为 `-0.00104`，95% CI
`[-0.00209, -0.00041]`。这说明每轮刷新 attention 在该 zero-shot 场景有小而一致的收益，但只验证
了一个建筑，尚不能将约 1% 的提升外推为跨数据集普遍结论。

完整逐帧证据与汇总位于：

- `results/matterport3d/hxp_fixed_attention_huber_zero_shot/`；
- `results/matterport3d/hxp_iterative_attention_huber_zero_shot/`；
- `results/matterport3d/hxp_three_scale_estimators_zero_shot_comparison.json`。

## 8. BIM Early-Fusion Dense 模型的 zero-shot 结果

在 Area_1 上训练 10 epochs 的 `BIMEarlyFusionDepthAnythingV2` epoch-9 best checkpoint
也按同一场景、同一 504 process resolution、同一焦距修正 DA3 和同一 wall-filled BIMNet
mesh 进行 zero-shot。网络在 Matterport 上没有训练、微调、选模或 scale alignment，直接输出
dense absolute metric depth。评测使用旧 full-regression `per_frame.csv` 作为不可变筛选
receipt；792 帧筛选重算 mismatch 为 0，Raw DA3 汇总也与旧评测逐值一致。

运行命令：

```bash
.venv/bin/python scripts/model/evaluate_matterport_bimnet_early_fusion.py \
  --matterport-root /path/to/Matterport3D \
  --bimnet-root /path/to/BIMNet_release \
  --toolkit-root /path/to/S3-SAM3D-ToolKit \
  --bimnet-scene hxp \
  --config configs/stanford_area1_bim_early_fusion_dense.yaml \
  --checkpoint outputs/stanford_area1_bim_early_fusion_dense/best.pt \
  --benchmark-reference-csv \
    results/matterport3d/hxp_full_regression_scale_zero_shot/per_frame.csv \
  --output-dir results/matterport3d/hxp_bim_early_fusion_dense_zero_shot \
  --process-res 504 \
  --device cuda
```

全深度 pixel-micro AbsRel 如下：

| 子集 | 帧数 | Raw DA3 | Dense early fusion | 相对 Raw 变化 | 逐帧胜率 |
|---|---:|---:|---:|---:|---:|
| all GT-valid | 777 | 0.16409 | 0.18070 | **退化 10.12%** | 56.37% |
| GT quality（主结果） | 740 | 0.16403 | 0.18021 | **退化 9.87%** | 57.57% |
| operational no-GT | 620 | 0.16664 | 0.15876 | 改善 4.73% | 66.45% |
| GT-verified（诊断） | 550 | 0.16505 | 0.10934 | 改善 33.76% | 73.45% |
| rejected from effective | 227 | 0.15907 | 0.55188 | **退化 246.94%** | 14.98% |

GT-quality 主结果的完整指标为：

| 方法 | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE_log |
|---|---:|---:|---:|---:|---:|---:|
| Raw DA3 | **0.16403** | **0.33540** | 0.23333 | 0.81351 | **0.96662** | **0.20536** |
| Dense early fusion | 0.18021 | 0.44238 | **0.23181** | **0.83023** | 0.92171 | 0.26183 |

这不是“模型完全不能迁移”：在已确认 BIM 覆盖且与真实可见结构一致的 550 帧上，AbsRel 从
`0.16505` 降到 `0.10934`，RMSE 从 `0.33214 m` 降到 `0.26861 m`，说明模型确实能利用
跨域有效 BIM。但是它在 BIM 未覆盖、错配或命中错误围护结构的 227 帧上没有可靠回退，AbsRel
升至 `0.55188`，最终使不按 BIM/GT 一致性挑帧的正式 GT-quality 主结果差于 Raw DA3。

与三种只修改一帧全局尺度的结构比较：

| 方法 | GT-quality AbsRel | operational no-GT | GT-verified | rejected |
|---|---:|---:|---:|---:|
| Full regression scale | 0.11360 | 0.11133 | 0.10492 | 0.15981 |
| Fixed Huber scale | 0.10635 | 0.10323 | 0.09745 | 0.15378 |
| **Iterative Huber scale** | **0.10532** | **0.10222** | **0.09692** | **0.15015** |
| Dense early fusion | 0.18021 | 0.15876 | 0.10934 | 0.55188 |

因此在该单场景 zero-shot benchmark 上，dense early fusion 的主结果明显不如三个 scale-only
模型。最直接的诊断是 domain shift 下的 BIM reliability / fallback 失败：全网络 fine-tuning
允许错误 BIM 改写 dense token 与深度，而 scale-only 模型的输出空间受一个标量约束，错误先验
造成的破坏更有限。GT-verified 只能作为“BIM 正确时的条件性能”诊断，不能用其 `0.10934`
替代主 benchmark 的 `0.18021`。

机器可读结果位于：

- `results/matterport3d/hxp_bim_early_fusion_dense_zero_shot/per_frame.csv`；
- `results/matterport3d/hxp_bim_early_fusion_dense_zero_shot/summary.json`。

## 9. 三规则正式有效帧汇总

正式帧集合只由以下条件的交集确定：

1. `gt_valid_fraction > 0.10`；
2. `bim_hit_fraction > 0.20`；
3. `AABB_min <= camera_center <= AABB_max`，不使用 margin。

明确排除的筛选因素包括 BIM/GT agreement、BIM/DA3 ratio support、模型 confidence、预测
误差和任何模型特定条件。四份逐帧 CSV 均选出相同的 624 个 frame ID，集合 SHA256 为
`e6639e7bd16eb7b666a6f22f41ee17ec50a2ce8d8b841427ad489126013bd18b`。所有指标使用完整
Matterport 正深度 GT、pixel-micro 聚合，预测不做 scale/affine alignment。

| 方法 | AbsRel | RMSE (m) | MAE (m) | delta1 | delta2 | RMSE_log | 相对 Raw AbsRel | 逐帧胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw DA3 | 0.165732 | 0.331608 | 0.233062 | 0.812649 | 0.969444 | 0.202418 | - | - |
| Full regression scale | 0.111086 | 0.266687 | 0.154178 | 0.907183 | **0.973256** | 0.170223 | 32.97% | 80.77% |
| Fixed-attention Huber scale | 0.103019 | 0.262662 | 0.145563 | 0.914014 | 0.971836 | 0.166771 | 37.84% | 83.81% |
| **Iterative-attention Huber scale** | **0.102018** | **0.261359** | **0.144137** | **0.915987** | 0.971799 | **0.166042** | **38.44%** | **84.13%** |
| BIM early-fusion dense | 0.170670 | 0.418547 | 0.213292 | 0.855974 | 0.933837 | 0.250093 | **-2.98%** | 65.87% |
| Area_1 iterative scale+refiner SOTA（final） | 0.105769 | 0.257265 | 0.149570 | 0.911615 | **0.978996** | 0.160549 | 36.18% | 73.88% |

新定义没有改变结构排序：iterative Huber 仍是三种 scale estimator 和全部当前候选中的最优
AbsRel/RMSE/MAE/delta1/RMSE-log 方法。Dense early fusion 在 65.87% 的帧上逐帧 AbsRel
优于 Raw，且 MAE、delta1 也优于 Raw，但少数严重错误使总体 AbsRel 退化 2.98%、RMSE
显著增大；这仍支持“跨域错误 BIM 缺乏安全回退”的诊断。

作为边界敏感性检查，若仅把 AABB 放宽为旧协议的 0.25 m margin，会选出 633 帧；对应
AbsRel 为 Raw `0.166886`、full regression `0.112633`、fixed Huber `0.104598`、iterative
Huber `0.103599`、dense early fusion `0.172560`。方法排序和结论均不变。正式数字仍固定使用
margin 0 的 624 帧，不依据性能选择边界。

复现：

```bash
.venv/bin/python \
  scripts/analysis/reaggregate_matterport_zero_shot_three_rule.py
```

机器可读汇总为
`results/matterport3d/hxp_three_rule_zero_shot_comparison.json`。

## 10. Area_1 scale+refiner SOTA 的 zero-shot 结果

Area_1 全深度协议下的 scale+refiner 发布版 SOTA 为：

- config：`configs/stanford_area1_iterative_scale_3round_full_depth_metric_da3.yaml`；
- checkpoint：`outputs/stanford_area1_iterative_scale_3round_full_depth_metric_da3/accepted.pt`；
- checkpoint SHA256：`74f2797dc42a4e7e8359440ea9d305a073e7ec0d2fe0850fb1ab79877bb7ae6d`；
- Area_1 validation/test final AbsRel：`0.06421 / 0.06305`。

该模型使用三轮 iterative scale-conditioned attention、`r_low+r_detail` refiner，以及冻结的
DA3 layer 11/23 features；`r_frame` 关闭。下表是在 HxpKQynjfin 上不训练、不调参的严格
zero-shot 结果。有效帧仍为同一三规则选出的 624 帧，集合 SHA256 仍为
`e6639e7bd16eb7b666a6f22f41ee17ec50a2ce8d8b841427ad489126013bd18b`。

| 本 checkpoint 的推理阶段 | AbsRel | RMSE (m) | MAE (m) | delta1 | delta2 | RMSE_log | 相对 Raw AbsRel | 逐帧胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw DA3 | 0.165732 | 0.331608 | 0.233062 | 0.812649 | 0.969444 | 0.202418 | - | - |
| learned scale | 0.120093 | 0.277114 | 0.168279 | 0.898258 | 0.973548 | 0.174682 | 27.54% | 77.40% |
| scale + `r_low` | **0.103467** | **0.255510** | **0.146866** | **0.913863** | 0.978921 | **0.159488** | **37.57%** | **75.00%** |
| scale + `r_low+r_detail`（final） | 0.105769 | 0.257265 | 0.149570 | 0.911615 | **0.978996** | 0.160549 | 36.18% | 73.88% |
| GT oracle frame scale（诊断） | 0.063964 | 0.236421 | 0.096782 | 0.944613 | 0.975823 | 0.142802 | 61.40% | - |

结论需要区分“Area_1 SOTA”和“MP3D zero-shot SOTA”：该完整模型的 final 相对 Raw DA3
改善 36.18%，说明它能迁移；但 `r_detail` 使 AbsRel 相对 `scale+r_low` 从 `0.103467`
退化到 `0.105769`（约 2.22%）。此前单独训练的 iterative-attention Huber scale-only
在同一 624 帧 benchmark 上为 `0.102018`，仍略优于本模型的最佳中间阶段和 final。因此，
Area_1 上最优的 dense refiner 并未在该跨域场景成为 zero-shot SOTA，主要负迁移来自细节残差，
其次该完整 checkpoint 自身的 scale head（`0.120093`）也弱于专门的跨结构 scale-only 对照。

复现命令：

```bash
.venv/bin/python scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
  --matterport-root /path/to/Matterport3D \
  --bimnet-root /path/to/BIMNet_release \
  --toolkit-root /path/to/S3-SAM3D-ToolKit \
  --bimnet-scene hxp \
  --config configs/stanford_area1_iterative_scale_3round_full_depth_metric_da3.yaml \
  --checkpoint outputs/stanford_area1_iterative_scale_3round_full_depth_metric_da3/accepted.pt \
  --output-dir results/matterport3d/hxp_iterative_scale_refiner_sota_zero_shot \
  --process-res 504 \
  --device cuda \
  --progress-every 100
```

逐帧与机器可读结果位于：

- `results/matterport3d/hxp_iterative_scale_refiner_sota_zero_shot/per_frame.csv`；
- `results/matterport3d/hxp_iterative_scale_refiner_sota_zero_shot/summary.json`。

## 11. Reduced-input Huber + refiner continuation

为与三种 scale estimator 的八通道对照保持一致，Fixed-attention Huber 与
Iterative-attention Huber 的三轮 scale-only checkpoint 又分别续训了 refiner。续训时不重复
前三个 scale-only epoch：先加载已训练的 `attention_scale.*` 及其命名 Adam 状态，训练 9 个
refiner-only epoch，再训练 3 个 joint epoch；正式评测使用 validation AbsRel 最优的 `best.pt`。
两个最佳 checkpoint 都出现在第 5 个 continuation epoch（refiner-only），而不是 joint 阶段。

scale head 仍使用相同八通道输入：RGB、focal-corrected DA3 depth、BIM depth、BIM hit mask、
有符号及绝对 BIM/DA3 log-ratio。Reduced refiner 使用 RGB、raw/scaled DA3 geometry、learned
scale channel、BIM depth/hit/disagreement；不使用 DA3 confidence、DA3 layer features、BIM
normal、BIM edge 或 deterministic-scale feature。

Area_1 official-all-valid test（1,641 帧，pixel-micro）结果为：

| 模型 | Raw | Round 1 | Round 2 | Round 3 scale | +`r_low` | Final | Final RMSE | Final delta1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Huber + reduced refiner | 0.085453 | 0.073757 | 0.069118 | 0.068114 | **0.063705** | **0.063595** | **0.414976** | **0.947431** |
| Iterative Huber + reduced refiner | 0.085453 | 0.073734 | 0.069099 | **0.068092** | 0.064061 | 0.063988 | 0.415054 | 0.946991 |

在 Area_1 test 上，Fixed final 相对 raw 改善 25.58%，相对其 Round-3 scale 改善 6.64%；
Iterative 对应改善 25.12% 和 6.03%。Fixed final AbsRel 比 Iterative 低 `0.000394`，但该 test
此前已经揭盲，因此不能据此声称盲测选模优势。

两者随后在 HxpKQynjfin 上冻结权重进行 zero-shot。仍严格使用 624 帧三规则集合，frame-ID
SHA256 为 `e6639e7bd16eb7b666a6f22f41ee17ec50a2ce8d8b841427ad489126013bd18b`；
两个运行均为 792 个源帧、777 个正常 GT、15 个坏 GT、0 error，且未执行 test-time alignment。

| 模型/阶段 | AbsRel | RMSE (m) | MAE (m) | delta1 | Frame-macro AbsRel |
|---|---:|---:|---:|---:|---:|
| Raw DA3 | 0.165732 | 0.331608 | 0.233062 | 0.812649 | 0.173679 |
| Fixed Round-3 scale | 0.103039 | 0.262681 | 0.145594 | 0.914000 | 0.109813 |
| Fixed + `r_low` | 0.104254 | 0.267771 | 0.148898 | 0.910786 | 0.111511 |
| Fixed final | 0.103608 | 0.267383 | 0.148071 | 0.911300 | 0.110873 |
| **Iterative Round-3 scale** | 0.102020 | **0.261352** | **0.144135** | **0.915967** | **0.108860** |
| Iterative + `r_low` | 0.102158 | 0.266590 | 0.146615 | 0.913081 | 0.109387 |
| **Iterative final** | **0.101781** | 0.266364 | 0.146135 | 0.913364 | 0.109016 |

Iterative final 相对 raw 改善 38.59%，并比 Fixed final 低 `0.001826` AbsRel；同帧比较中，
Iterative 的 scale/final 分别在 60.10%/60.58% 帧上优于 Fixed。其 final 相对自身 scale 仅改善
0.234%，而 Fixed final 相对自身 scale 退化 0.551%。因此这次实验支持两点：reduced Huber
scale 具有稳定的跨域泛化；Area_1 上的 refiner 增益大多不能迁移到 Hxp，动态 attention 主要
保住并小幅改善 scale，而不是带来同等幅度的跨域 dense refinement 收益。

Artifacts：

- `results/stanford_area1/fixed_attention_huber_reduced_refiner_continuation_test/summary.json`；
- `results/stanford_area1/iterative_attention_huber_reduced_refiner_continuation_test/summary.json`；
- `results/matterport3d/hxp_fixed_attention_huber_reduced_refiner_continuation_zero_shot/summary.json`；
- `results/matterport3d/hxp_iterative_attention_huber_reduced_refiner_continuation_zero_shot/summary.json`。
