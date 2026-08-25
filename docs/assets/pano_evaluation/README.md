# Area_1 全景评测素材说明

本目录只服务于 **training-free 全景联合估计** 与确定性 BIM 基线。主图不使用 learned
refiner；正式 evaluator 中遗留的 learned 字段仅作历史审计，不应拼入本轮论文图。

## 推荐选图

定量图位于 [`quantitative/`](quantitative/)，每张均提供 PNG、SVG 和 PDF；SVG/PDF 适合论文
排版，PNG 适合快速预览。图中数值来自冻结后的 validation/test CSV 与 summary，完整来源
哈希见 [`quantitative/manifest.json`](quantitative/manifest.json)。

- 主结果：`test_main_pano_gain`。展示 regular-only、+tangent6、+tangent14 的 AbsRel 与球面覆盖率。
- BIM 基线：`deterministic_bim_chain`。如实展示 universal scale 有效，而局部 BIM-direct 没有
  带来额外增益。
- 统计稳定性：`room_level_paired_pano_claim` 与 `tangent_view_count_ablation`。
- 负结果/敏感性：`strict_single_vs_joint` 与 `fusion_sensitivity`。
- 直接单图对照：`strict_single_plus_pano_validation`，明确标为 validation-only；展示覆盖率
  大幅增加但共同单图 support 精度退化。
- 三个过程图备选：
  - `process_candidate_a_erp_tangent_pipeline`：推荐用于方法主图，突出 ERP→nested14→DA3→稳健融合；
  - `process_candidate_b_orthogonal_matrix`：推荐用于实验章节，突出正交对照矩阵；
  - `process_candidate_c_deployment_coverage`：推荐用于工程/部署说明，突出覆盖率与计算路径。

本版冻结的 F*=Huber 是按 validation 的 `regular-only` 误差选择，而不是按 pano 联合误差
选择；因此 `fusion_sensitivity` 中 weighted-log 的更低 pano 误差只能作为探索性结果，不能
在 test 已揭盲后替换正式主方法。下一版应先按 pano 目标预注册选择规则，再使用新盲测。

## 三组定性备选

定性素材运行后位于 `qualitative/`。其中含 Stanford RGB/GT 的派生图，因此按第三方数据许可
**只保存在本机并由 `.gitignore` 排除，不随公开仓库再分发**；公开仓库提供导出脚本与固定
选样规则。每个 panel 都是不含标题的 1024×512 PNG，便于后续在 PPT 中自由拼接；共享色条
同时提供 PNG、SVG 和 PDF。选择仅使用 formal validation CSV，在查看 RGB/GT 前按固定规则
完成，未用 test 图像挑样：

导出器只保存可直接排版的图像、指标与 manifest，不落盘包含原始 RGB/GT 的稠密 NPZ。

- A，中位代表：`office_6/2439b3f7bc184152bf89e6012a511aa6`，AbsRel 增益 0.03361
  （12.24%）。适合作为主文常规案例。
- B，最大增益：`hallway_2/2fb7a2dcf8004578a541ee750dda0ed6`，增益 0.07447
  （29.23%）。适合展示全景补充视角的上限收益。
- C，最困难：`office_31/a77fba57be66411c908077c69d803797`，增益 0.00618
  （3.23%）。适合 limitations 或补充材料。

本机快速总览为 `qualitative/preview_sheet.png`，完整选择规则、运行身份和面板 SHA 位于
`qualitative/manifest.json`。推荐拼版顺序为：
`pano_rgb`、`gt_range`、`regular_only_raw`、`regular_plus_tangent14_raw`、两者误差图、
`joint_minus_regular_signed_absrel`。若讨论 BIM，再追加 `universal_scale` 与 `bim_direct`；不要
把它们误写成 learned 方法。

## 视觉约定

- 深度/径向距离：`turbo`，固定 0.2–5.0 m；
- 逐像素 AbsRel：`magma`，固定 0–0.5；
- `joint − regular` 有符号误差变化：`coolwarm`，固定 −0.25–0.25，负值表示改进；
- 无效或不在共同 support 的像素：深灰色；
- 所有方法比较使用同一固定 support，不能按各自有效预测重新筛像素。

重新生成：

```bash
.venv/bin/python scripts/analysis/generate_pano_evaluation_assets.py

.venv/bin/python scripts/analysis/export_stanford_pano_panels.py \
  --device cuda \
  --preview-sheet \
  --output-dir docs/assets/pano_evaluation/qualitative
```

第二条命令只访问 validation，且不接受 checkpoint。正式实验协议与结果解释见
[`../../PANO_DEPTH_EVALUATION.md`](../../PANO_DEPTH_EVALUATION.md)。
