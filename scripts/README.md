# Script layout

CLI 按职责分组，所有命令从仓库根目录执行：

| 目录 | 边界 |
|---|---|
| `data/` | 数据下载、来源校验、位姿/配准、DA3 缓存、样本制备、split 与审计 |
| `model/` | 训练、无 GT 推理、2D 深度评测与 3D 重建评测 |
| `pipelines/` | 可续跑的端到端编排和环境 smoke check |
| `analysis/` | 不修改训练状态的统计、bootstrap 和可视化 |

数据实现的共享逻辑位于 `src/bim_priorda3/data/`，不应通过从一个 CLI 导入
另一个 CLI 来复用。新的共享功能应先放入 `src/`，CLI 只负责参数解析和结果
落盘。

- 训练前数据流程：[`docs/DATA_PREPARATION.md`](../docs/DATA_PREPARATION.md)
- 全部脚本索引和模型命令：[`docs/USER_GUIDE.md`](../docs/USER_GUIDE.md)
- 端到端实验顺序：[`docs/EXPERIMENT_PIPELINE.md`](../docs/EXPERIMENT_PIPELINE.md)
- 科研评测、消融和图件协议：[`docs/EVALUATION_PROTOCOL.md`](../docs/EVALUATION_PROTOCOL.md)
- Area_1 全景深度协议：[`docs/PANO_DEPTH_EVALUATION.md`](../docs/PANO_DEPTH_EVALUATION.md)

`scripts/` 不是 wheel CLI 包；公开复现模式是 clone 仓库、执行
`pip install -e ...`，然后从仓库根目录运行 `python scripts/<group>/<tool>.py`。

冻结 DA3 feature 融合实验额外使用
`data/cache_stanford_da3_features.py`：它在数据制备阶段导出固定 revision 的中/深层
encoder token，并写完整性 manifest；训练脚本只读缓存，不在线运行或更新 DA3。

## Area_1 panorama 入口

| 脚本 | 职责 | 关键开关 |
|---|---|---|
| `data/download_stanford_area1.py` | 选择性解压 Area_1 regular 数据和可选 pano 四模态 | `--include-pano`；可选 `--include-rvt` |
| `data/verify_stanford_bimsyn_sources.py` | 验证 Area_1/BIMSyn 来源并写 receipt | 全景实验使用 `--require-pano` |
| `data/cache_stanford_pano_da3.py` | ERP→tangent RGB，缓存 pinned DA3 z-depth/confidence，写不可变 split manifest | `--preset {cubemap6,nested14}`；正式 Route P 使用 `nested14` |
| `model/evaluate_stanford_pano.py` | regular-to-pano Route R；可选 tangent Route P、pano-only 与 regular+pano 分析 | Route P 传 `--tangent-manifest` |
| `analysis/evaluate_stanford_pano_single_plus_tangent.py` | 同一 GT-free 单帧与 +tangent6/14 的 val-only 固定支持域诊断 | 无 test/checkpoint/BIM 或可调 fusion 参数 |
| `analysis/evaluate_stanford_pano_regular_roundtrip.py` | 全部 regular→ERP 联合→反投影回原 regular 的 val-only 主诊断 | 在每张原始 regular 的完整固定 GT support 上比较；无 test/checkpoint/BIM |
| `analysis/evaluate_stanford_bim_scale_roundtrip.py` | 同一 universal BIM-scale 基线的 regular 与 regular→ERP→regular 配对评测 | val-only；无 pano RGB/GT、tangent、checkpoint 或 learned 模型 |
| `analysis/ablate_oracle_semantic_bim.py` | 官方 regular semantic + BIM ray category 的逐帧单一全局尺度上限消融 | 仅 train/val；语义只筛尺度对应，禁止 test |
| `analysis/analyze_scale_residual_distribution.py` | 分析 scale 输出之后的像素级米制/log 残差随距离和内容的变化 | 仅 train/val；GT 只在无 GT 尺度推理后用于诊断 |
| `analysis/generate_pano_evaluation_assets.py` | 从冻结 summary/CSV 生成 pano 定量图与三套流程图 | 不读取原始图像/GT |
| `analysis/export_stanford_pano_panels.py` | 从固定 val 规则导出三套本地定性素材 | 不加载 checkpoint；不访问 test |
| `analysis/audit_da3_focal_scaling.py` | 对比 cached canonical DA3 与按处理分辨率内参乘 `mean(fx,fy)/300` 的 metric DA3 | 不运行 checkpoint、不用 GT 定尺度、不改缓存 |
| `analysis/compare_stanford_evaluations.py` | 对两个同 support 的 Area_1 summary 做逐子集差值与配对 room-bootstrap | 要求 split、房间、样本数、GT support 和像素数一致 |

焦距审计示例：

```bash
.venv/bin/python scripts/analysis/audit_da3_focal_scaling.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --split val --split test --workers 8 \
  --output results/stanford_area1/da3_focal_scaling_full_depth_audit/summary.json
```

脚本中的 focal 必须是 DA3 实际 processing resolution 下的 `fx/fy`，不能直接使用未 resize 的
原图内参。输出同时保留旧 canonical 指标，便于定位历史表格错误。

validation 数据输入和 Route R 的最小入口为：

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

.venv/bin/python scripts/model/evaluate_stanford_pano.py \
  --config configs/stanford_area1.yaml \
  --split val --output outputs/stanford_area1/pano_val_training_free \
  --device cuda --batch-size 8 --pano-height 512 \
  --bootstrap-repetitions 10000 --seed 42
```

全景主协议不加载 checkpoint：BIM 增强只比较确定性的 universal scale 与 BIM-direct。
`--checkpoint` 仅保留给历史 learned 产物的审计复核，不进入本轮方法选择或主结果。

如需 Route P，先运行 `data/cache_stanford_pano_da3.py --split val --preset nested14
--face-resolution 504`，再把它打印的 `val_full.json` 路径传给 evaluator 的
`--tangent-manifest`。`--max-stations` 只用于 exploratory smoke。缓存和 evaluator 访问 test
时都要求同时提供 `--split test --confirm-test`。`single_best_view` 是逐 ERP 像素选择一个来源
的 no-fusion mosaic，不是整站只输入一张图。

推荐用下面的 round-trip 协议直接衡量全景联合对原始 regular benchmark 的收益。它先冻结每站
球面预测，再读取 regular GT；不读取 pano GT：

```bash
.venv/bin/python scripts/analysis/evaluate_stanford_pano_regular_roundtrip.py \
  --config configs/local/stanford_area1.yaml \
  --tangent-manifest "$PANO_TANGENT_MANIFEST" \
  --output outputs/stanford_area1/pano_val_regular_roundtrip_reproduction
```

该入口固定为 validation-only；没有 test、checkpoint、BIM 或融合调参开关。

隔离“BIM scale 后多 regular ERP 联合”的增益时，使用同基线入口；它不需要 tangent cache：

```bash
.venv/bin/python scripts/analysis/evaluate_stanford_bim_scale_roundtrip.py \
  --config configs/local/stanford_area1.yaml \
  --output outputs/stanford_area1/pano_val_bim_scale_regular_roundtrip
```

reference 与 candidate 都先做完全相同的逐帧 universal BIM scale，随后只改变是否进行同站
ERP 联合；输出逐 frame/station/room 指标与 room-cluster paired bootstrap。

官方语义全局尺度先在 train 选择分位数，再在 validation 评测：

```bash
.venv/bin/python scripts/analysis/ablate_oracle_semantic_bim.py \
  --config configs/local/stanford_area1.yaml --split train \
  --study semantic-scale --selection-only --workers 8 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --output-dir results/stanford_area1/oracle_semantic_global_scale_train_reproduction

.venv/bin/python scripts/analysis/ablate_oracle_semantic_bim.py \
  --config configs/local/stanford_area1.yaml --split val \
  --study semantic-scale --workers 8 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --output-dir results/stanford_area1/oracle_semantic_global_scale_val_reproduction
```

该脚本在预测时读取发布的 semantic annotation，所以只能衡量方法上限；不得在
test 运行。每帧只输出一个应用于全图的尺度；不得把数字当成可部署 RGB-only semantic
predictor 的性能。
