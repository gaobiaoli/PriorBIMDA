# 非学习 BIM-direct 逐因素消融

## 1. 目的与边界

本实验回答确定性 BIM-direct 中每个因素是否有效。实验是在正式 blind test 已完成后追加的
**post-hoc validation diagnostic**，因此只读取 SLABIM validation（104 帧、6 区域）和
Area 1 validation（1673 帧、7 房间），没有重新读取 test，也没有修改冻结的 v1 协议。

所有变体使用相同 cached DA3、BIM、GT support 和 0.2–5.0 m 范围。每次只移除或替换一个
因素，报告 pixel/frame/group 聚合，并对 region/room 做 10,000 次配对 bootstrap。

机器结果：[summary.json](../results/deterministic_baseline_ablation/summary.json)。

## 2. 非学习方法实际使用的因素

正式 `universal_bim_direct` 包含：

1. `0.2 < BIM/DA3 < 5.0` 比例过滤；
2. 至少 100 个有效比例，否则尺度回退为 1；
3. `Q25(log-ratio)+0.05` 对 log-Q45 的单侧稳健 cap；
4. `|log(BIM/scaled DA3)| <= 0.10` 一致性门；
5. 从 BIM depth 重新计算的 Sobel 梯度门 `<0.25`；
6. `sigma=64` 的 normalized Gaussian 空间传播；
7. smoothed support `<0.05` 时将局部场归零；
8. 局部 log correction 乘数 `alpha=1.25`。

容易混淆的一点是：存储在 NPZ 中的 `bim_normals` 和 `bim_edge` **不参与非学习输出**；
它们是学习网络的条件输入。这里被消融的“边缘”是直接从 BIM depth 计算的 Sobel gate。

## 3. 总体结果

下表为 all-pixel validation。`Δ pixel` 是变体减去完整方法；正值表示移除该因素后变差，
即该因素有帮助。CI 是 group-macro AbsRel 差的 95% bootstrap 区间。

| 变体 | SLABIM AbsRel | Δ pixel | SLABIM group Δ 95% CI | Area 1 AbsRel | Δ pixel | Area 1 group Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.078483 | — | — | 0.087100 | — | — |
| Scale only / 去掉全部局部修正 | 0.083293 | +0.004809 | [-0.00229, +0.01012] | **0.086442** | -0.000658 | [-0.00261, +0.00081] |
| 去 Q25 cap | 0.084765 | +0.006281 | [-0.00231, +0.01840] | 0.091201 | +0.004101 | [-0.00158, +0.01075] |
| 放宽 ratio bounds | 0.078535 | +0.000052 | [-0.000005, +0.000138] | 0.087130 | +0.000030 | [-0.000307, +0.000049] |
| min samples 100→1 | 0.078483 | 0 | [0, 0] | 0.087100 | 0 | [0, 0] |
| 去一致性门 | 0.102099 | **+0.023616** | **[+0.01500, +0.03387]** | 0.102019 | **+0.014919** | **[+0.00177, +0.01710]** |
| 去 Sobel edge gate | 0.078501 | +0.000018 | [-0.000297, +0.000267] | **0.086873** | **-0.000227** | **[-0.000300, -0.000040]** |
| 去 Gaussian propagation | 0.082286 | **+0.003803** | **[+0.00015, +0.00674]** | 0.089787 | **+0.002686** | **[+0.00187, +0.00341]** |
| 去 support cutoff | 0.080365 | +0.001882 | [-0.00004, +0.00746] | 0.087412 | +0.000312 | [-0.00047, +0.00057] |
| alpha 1.25→1.0 | **0.078097** | -0.000386 | [-0.00118, +0.00027] | **0.085957** | **-0.001143** | **[-0.00146, -0.00054]** |

## 4. 结论

- **明确有效且跨域一致**：DA3–BIM 一致性门、Gaussian 空间传播。
- **方向上有效但 group CI 不充分**：Q25 cap。它在两个数据集都改善点估计，并在 Area 1
  conflict 子集有明显收益，但 all 的房间/区域数较少，CI 仍跨 0。
- **运行安全因素、当前数据无法证明精度收益**：ratio bounds 和 min-samples fallback。
  所有 validation 帧都有至少 11,528 个比例，故 min-samples 消融完全相同；它的作用是保护
  BIM 覆盖极低的新场景。
- **弱有效/子集依赖**：support cutoff。在 SLABIM 点估计有帮助，Area 1 all 接近中性，但
  Area 1 furniture 的 removal CI 为正，仍建议保留。
- **没有得到有效性支持**：Sobel edge gate。SLABIM 基本中性；Area 1 去掉后反而在 7 个房间
  上获得小但一致的改善。
- **当前设置偏强**：`alpha=1.25`。改成 1.0 在两个数据集的点估计都更好，并在 Area 1
  room-bootstrap 上明确改善。
- **整个局部修正不是无条件通用增益**：它改善 SLABIM 点估计，但 Area 1 的 scale-only
  略好；两者 group CI 都跨 0。真正稳定、通用的核心是尺度估计、一致性筛选和空间传播。

## 5. 对下一版本的建议

冻结 v1 不能依据这次 post-hoc validation 直接改写，否则原 blind-test claim 失效。建议将
下一版本预注册为：保留稳健尺度、一致性门、Gaussian propagation 和 support cutoff；移除
Sobel edge gate；把 `alpha` 从 1.25 改为 1.0。先在新的 validation protocol 验证该组合交互，
再重训/验收 refiner，最终用新的未见建筑或新冻结 test 确认。

复现命令：

```bash
python scripts/analysis/ablate_deterministic_bim_direct.py \
  --slabim-config configs/slabim.yaml \
  --stanford-config configs/stanford_area1.yaml \
  --split val --workers 8 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --output results/deterministic_baseline_ablation/summary.json
```
