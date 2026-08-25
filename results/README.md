# Results

这里只保留统一尺度协议的正式小型产物和能够解释当前结论的关键诊断产物：

- `metrics.json`：论文/README 主表的紧凑机器可读版本；
- `manifest.json`：两份生产 checkpoint、协议和结果文件的 SHA-256；
- `slabim/`：108 帧 pooled-clean test 的 summary、逐帧 CSV、三维重建 summary、history 与
  run-state；
- `stanford_area1/`：room-disjoint regular-view validation/test 的 summary、逐帧 CSV、history 与
  run-state；`pano_val/` 与 `pano_test/` 另存 same-station regular/pano 的固定支持域评测，
  `pano_val_single_plus_tangent/` 是揭盲后补做的 validation-only 单图对照；
  `pano_val_regular_roundtrip/` 把同站 regular/pano 预测投到 ERP 联合后反投影回全部原始
  regular，并在每帧原始 GT mask 上评测；`pano_val_bim_scale_regular_roundtrip/` 则在两侧
  都使用同一逐帧 universal BIM scale，只隔离同站 ERP 联合的增益；
  `oracle_semantic_global_scale_{train,val}/` 使用官方 semantic annotation 选择同类
  BIM/DA3 对应并估计逐帧单一全局尺度；前者选择 q65，后者冻结评测。旧
  `oracle_semantic_val/` 是被推翻的像素替换试验，仅保留为试错记录；两者都不是可部署或
  blind-test 结果；`attentive_scale_mlp_scratch_{val,test}/` 记录冻结 DA3、任务网络从头
  三阶段训练和 bounded-MLP scale 修正；`attentive_scale_da3_features_{val,test}/` 在同一
  协议上增加冻结 DA3 第 11/23 层 feature 融合；`hybrid_additive_{val,test}/` 将 feature
  仅送入 refiner，并评测 `比例残差` 与 `比例+米制加法残差`。hybrid final 是当前 AbsRel
  最低的 Area_1 候选（val/test `0.06131/0.06235`），但相对其比例分支的加法收益仅
  `0.007%/0.119%`，且相对前一 feature 模型的 MAE/RMSE 退化。三者的 test 都因此前已揭盲
  而只属于诊断结果，不能替代新区域 blind confirmation。
  `attentive_scale_da3_features_stage_ablation_{val,test}/` 对回退后的无加法 checkpoint
  顺序评测 `scale`、`scale+r_low`、`scale+r_low+r_detail`；结果显示 low 占总 AbsRel 改善的
  validation/test `98.98%/95.44%`。当前推荐该无加法版本，hybrid 仅作归档诊断。
  `attentive_scale_da3_features_hit_only_{val,test}/` 使用相同网络和划分，改为无 BIM 距离
  截断、保留 door/window 的正向命中 prior。覆盖率从 88.82% 增至 99.94%，但 retrained
  final 的 val/test AbsRel 为 `0.06306/0.06585`，较旧 bounded-core prior 退化
  `1.56%/5.47%`。机器可读的新旧协议汇总位于
  `attentive_scale_da3_features_hit_only_protocol_comparison.json`；这是负诊断，不替代推荐模型。
  `attentive_scale_da3_features_hit_only_full_depth_{val,test}/` 则不再只做事后全图评测，而是
  用官方全部有效 regular GT 从头训练、validation 选点并评测；仅排除 raw `0/65535`，模型
  输出上限为 128 m。final val/test AbsRel 为 `0.06861/0.06741`。相对旧 hit-only checkpoint
  在完全相同全深度 support 上改善 `4.90%/0.054%`，说明 validation 收益明确而 test AbsRel
  基本持平；该 test 已揭盲，仍属于诊断结果。
- `deterministic_baseline_ablation/`：两个 validation split 上的非学习 BIM-direct 逐因素
  post-hoc 消融；不属于 blind-test 主结果。
- `stanford_area1/scale_residual_distribution_val/`：在 1,673 张 validation 图上对无 DA3-feature
  attention scale 和 universal scale 的尺度后残差做 685 万像素诊断。learned scale 的距离幂
  指数为 0.404；`|error| ≈ 0.0566 m + 0.0402 depth` 明显优于纯加法或纯比例模型。该目录包含
  JSON、逐深度 bin CSV 和两张可直接用于论文/PPT 的 PNG。

旧 q45-only SLABIM、BIM-direct 网络锚点、E2E challenger、region-CV、旧消融和事后区域分析
均已从活动项目删除，避免与当前公共方法混用。历史 summary 内的绝对路径只是审计字段；公开
复现应生成本机 child config/receipt，不能手工关闭 provenance 校验。

本机 checkpoint 校验：

```bash
sha256sum -c results/checkpoints.sha256
```

`outputs/` 默认不进入 Git。发布者应把两份约 30 MB 的 checkpoint 上传至 Release/Hugging
Face，并在 `manifest.json` 补充 URL；不得把第三方数据或大型旧 E2E 权重提交到 Git 历史。
冻结 DA3-feature 与 hybrid 候选 checkpoint 本机分别位于
`outputs/stanford_area1_attentive_scale_da3_features/accepted.pt` 和
`outputs/stanford_area1_hybrid_additive/accepted.pt`；它们在 `manifest.json` 中标记为
非公开诊断权重，应先经新区域盲测再决定是否晋升发布。

Area_1 pano 主结果是无训练联合估计：冻结 `joint_huber` 后，test 上
`regular+tangent14` 相对 `regular-only` 的 exact-solid-angle equal-station AbsRel 从
0.26985 降至 0.22865（下降 15.27%，7/7 房间改善），球面覆盖率从 66.92% 提高至
99.96%。确定性 BIM 分支只报告统一尺度与 BIM-direct；learned pano 输出仅保留在原始 evaluator
产物中作审计，不进入 pano 主表。选择与一次性 test 回执位于 `data/provenance/`。

必须区分“多 regular + pano”与“单图 + pano”：后者的探索性 validation AbsRel 为
0.11586（single）、0.16454（+tangent6）和 0.31859（+tangent14），虽然原生球面覆盖率从
10.87% 增至 99.38%/99.85%，但相同单图 support 上精度显著下降。它不是 test claim，作用是
约束下一版协议先解决跨切平面尺度/上下文漂移。当前 `joint_huber` 又是按 regular-only
validation 目标冻结，不能在 test 已揭盲后依据 pano sensitivity 改选方法。

对 regular benchmark 更直接的 validation-only round-trip 结果覆盖 1,673 帧、30 站、7 房间和
364,913,264 个固定 GT 像素。raw DA3 为 0.27710；最佳 regular-only joint 为 0.25954；
`regular+tangent14` weighted-log 为 0.18654。与相同 weighted-log regular-only 的 0.26682
相比下降 30.09%，7/7 房间改善，95% CI 为 [−0.09133,−0.07649]。该协议不读取 pano GT、
不使用 BIM/checkpoint/learned 模型；由于方法在 validation 上选定，它是后续盲测的候选协议，
不是新的 test claim。

同基线 BIM-scale round-trip 也只在 validation 上运行。逐帧 universal scale 与原 regular
评测完全一致（AbsRel 0.08644），同站 joint Huber 回投为 0.07595，下降 12.14%；7/7 房间
改善，room-cluster paired 95% CI 为 [−0.01231,−0.00700]。该协议覆盖相同的 1,673 帧和
364,913,264 个 regular GT 像素，不读取 pano RGB/GT、tangent 或 learned checkpoint；它回答
的是已有多 regular 的联合收益，不是新增 pano 图像内容的收益。

同一输出中的 overlap residual calibration（`joint_synchronized_huber`）相对普通 Huber 将
pixel-micro AbsRel 从 0.07595 降至 0.07446，但 MAE 从 0.15254 增至 0.15887，且只有
4/7 房间改善；房间聚类 AbsRel 差值 95% CI [−0.00682,0.00408] 跨 0。它是混合的敏感性
结果，不替代默认 `joint_huber`。

Area_1 official-semantic global-scale oracle 只用图像/BIM 同类结构像素估尺度，并把每帧
唯一尺度应用到全图。q65 由 30 个 train rooms 选择；冻结到 validation 后 AbsRel 从
all-hit scale 的 0.08644 降到 0.07725（-10.63%），MAE 从 0.17422 降到 0.15464；
7/7 rooms 改善，room-cluster 95% CI `[-0.01404,-0.00273]`。家具也因全局尺度变化从
0.10979 降到 0.10097。使用官方语义使其仍只是 privileged 上限，不是部署方法。

SLABIM 三维融合评测在 BIM 坐标中使用冻结恢复位姿、4 像素采样步长和 5 cm voxel：
learned 的 Chamfer-L1 为 0.09109 m（direct 0.10515 m），F-score@10 cm 为 0.79622
（direct 0.74003）。详见 `slabim/reconstruction_test/summary.json`。
