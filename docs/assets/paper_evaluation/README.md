# 论文评测与 PPT 素材目录

本目录由真实冻结结果和验证集推理生成，不含生成式图片。量化图以最终结果或明确标注的
历史/训练集证据为数据源；定性样例只从 Area_1 validation 按固定规则选择，不能当作 blind
test 证据。所有深度图固定显示范围为 0.2--5.0 m。

由于 Area_1 的 RGB、GT 和稠密派生数组受第三方数据许可约束，`qualitative/` 只在本地生成并
被 Git 忽略，不随公开仓库再分发；下列定性链接会在执行第 4 节命令后生效。量化统计图、流程
图及本目录说明可正常公开。

完整评测设计见 [`docs/EVALUATION_PROTOCOL.md`](../../EVALUATION_PROTOCOL.md)，机器可读来源、
数值和 SHA-256 见 [`manifest.json`](manifest.json) 与
[`qualitative/manifest.json`](qualitative/manifest.json)。

## 1. 过程图三选一

每张均提供 PNG、SVG、PDF，建议在 PPT 中优先使用 SVG：

| 备选 | 用途 | 预览 |
|---|---|---|
| A：简洁方法流水线 | 论文主图 / 摘要页；突出先尺度矫正、再学习残差 | [`candidate_a_method_pipeline.png`](process/candidate_a_method_pipeline.png) |
| B：多条件网络结构 | 方法章节；展示 RGB/DA3、BIM adapters 和三类 residual heads | [`candidate_b_dual_stream_architecture.png`](process/candidate_b_dual_stream_architecture.png) |
| C：无泄漏评测协议 | 实验设置 / 可复现性页；展示 train-only 选择、validation 晋级、test-once | [`candidate_c_evaluation_protocol.png`](process/candidate_c_evaluation_protocol.png) |

推荐：正文方法总览选 A，结构细节或补充材料选 B，答辩的实验可信度说明选 C。

## 2. 定性示例三选一

总预览见
[`three_options_preview_contact_sheet.png`](qualitative/three_options_preview_contact_sheet.png)。
预览页只用于选择；正式 PPT 请从各目录取独立的 504×504 无标题 PNG，并配合
[`shared_colorbars/`](qualitative/shared_colorbars/) 中的统一色条。

### A：典型案例

- 目录：[`option_a_typical/`](qualitative/option_a_typical/)
- 样本：`office_31/camera_613d201e991744799556e737bdb89c89_office_31_frame_67`
- 规则：在有效像素大于 100,000 的 validation 帧中，取 learned-vs-direct gain 最接近
  全体中位数者。
- all AbsRel：raw DA3 `0.43887`，robust BIM-direct `0.06515`，refined `0.05600`。
- 适合：正文中说明常规场景下模型在强 BIM 基线之上继续改善。

### B：家具与 BIM 冲突恢复

- 目录：[`option_b_furniture_conflict_success/`](qualitative/option_b_furniture_conflict_success/)
- 样本：`office_6/camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_20`
- 规则：家具像素大于 30,000 后，最大化“家具像素占比 × 家具 AbsRel 改善”。
- support：家具 155,708 px，BIM 前景冲突 192,346 px。
- all AbsRel：direct `0.17859` → refined `0.07086`；家具 `0.21690` → `0.06120`；
  conflict `0.17875` → `0.05080`。
- 适合：解释固定围护 BIM 不含家具时，RGB/DA3 条件如何恢复真实前景。

### C：诚实失败案例

- 目录：[`option_c_failure/`](qualitative/option_c_failure/)
- 样本：`office_31/camera_613d201e991744799556e737bdb89c89_office_31_frame_9`
- 规则：在有效像素大于 100,000 的 validation 帧中选择 learned-vs-direct gain 最负者。
- all AbsRel：direct `0.05462` → refined `0.23615`；模型发生明显过矫正。
- 适合：limitations、答辩问答或 failure analysis；不得隐藏后只展示成功案例。

每个样本目录均包含 `rgb`、`gt`、`raw_da3`、`bim_depth`、`robust_global_scale`、
`robust_bim_direct`、`refined`、三种 error map、改善图、家具/冲突/BIM coverage mask、
reliability、routing gate、frame/low/detail/total residual，以及 `arrays.npz`、`metrics.json`、
`manifest.json`。

## 3. 已生成的量化图

| 图 | 证据范围 | 文件 |
|---|---|---|
| 两数据集主结果 | 正式 blind test，pixel-micro | [`main_blind_test_absrel.png`](quantitative/main_blind_test_absrel.png) |
| Area_1 all/furniture/conflict | 正式 blind test；显式保留 conflict 负结果 | [`area1_subset_absrel.png`](quantitative/area1_subset_absrel.png) |
| 房间配对与 95% CI | 7 个 test 房间，10,000 次 paired-room bootstrap | [`area1_room_pairs_and_bootstrap.png`](quantitative/area1_room_pairs_and_bootstrap.png) |
| robust scale 热图 | 7,013 个 train 样本、48 个候选；val/test 打开数为 0 | [`area1_train_only_scale_heatmap.png`](quantitative/area1_train_only_scale_heatmap.png) |
| robust scale Pareto | train-only all/furniture room-macro trade-off | [`area1_train_only_scale_pareto.png`](quantitative/area1_train_only_scale_pareto.png) |
| 训练曲线 | validation；frozen 与 E2E 各自 epoch | [`registered_training_curves.png`](quantitative/registered_training_curves.png) |
| 历史消融 | 旧 validation、single seed；只能作附录 | [`historical_slabim_validation_ablation.png`](quantitative/historical_slabim_validation_ablation.png) |

当前 Area_1 conflict blind-test 的 pixel AbsRel 为 direct `0.14064`、refined `0.14173`，
room-bootstrap 区间跨 0，因此不能写成稳定改善。图中已原样显示。

## 4. 一键重生成

从仓库根目录执行：

```bash
.venv/bin/python scripts/analysis/generate_paper_assets.py \
  --output docs/assets/paper_evaluation

.venv/bin/python scripts/analysis/export_stanford_qualitative_panels.py \
  --preset three-options \
  --device cuda \
  --output-dir docs/assets/paper_evaluation/qualitative
```

定量生成器只读取紧凑 JSON 结果；定性生成器严格验证 checkpoint/config/dataset provenance、
固定 seed 42，并要求 selection CSV 与运行时 validation population 完全相同。它不会读取 test
CSV 或 test 样本。
