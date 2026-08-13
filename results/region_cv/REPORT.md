# Region Cross-Validation Report: region_cv_v5_resource_v2

Status: **incomplete**

Completed runs: 8/18; summarized regions: 6/6.

Checkpoint selection: pretrain=`best`, final=`best`.

Aggregation order: average seeds within each held-out region, then average regions with equal weight.

> WARNING: This is an incomplete result. Missing or invalid runs were recorded as failures and were not silently discarded.

## Region-macro metrics

| Method | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ1 ↑ | Regions |
|---|---:|---:|---:|---:|---:|
| base | 0.250945 | 0.551968 | 0.373363 | 0.634854 | 6 |
| scaled | 0.107667 | 0.424967 | 0.169451 | 0.902967 | 6 |
| direct_bim | 0.103351 | 0.423027 | 0.162732 | 0.904881 | 6 |
| refined | 0.091834 | 0.396147 | 0.145611 | 0.921531 | 6 |

## Learned versus direct BIM

- Regions won: 6/6 (3F_Region2, 3F_Region3, 4F_Region2, 4F_Region3, 5F_Region2, 5F_Region3).
- Region-macro AbsRel relative improvement: 11.144%.
- Worst region by refined AbsRel: 3F_Region2 (0.142157).

## Per-region AbsRel

| Region | Seeds | Base | Scaled | Direct BIM | Refined | Relative improvement |
|---|---:|---:|---:|---:|---:|---:|
| 3F_Region2 | 3 | 0.244393 | 0.160261 | 0.158621 | 0.142157 | 10.379% |
| 3F_Region3 | 1 | 0.191034 | 0.117619 | 0.118425 | 0.111831 | 5.567% |
| 4F_Region2 | 1 | 0.279297 | 0.088056 | 0.082515 | 0.077654 | 5.892% |
| 4F_Region3 | 1 | 0.268834 | 0.082004 | 0.079762 | 0.073408 | 7.967% |
| 5F_Region2 | 1 | 0.244916 | 0.067480 | 0.054114 | 0.051704 | 4.453% |
| 5F_Region3 | 1 | 0.277193 | 0.130583 | 0.126671 | 0.094252 | 25.594% |

## Failures

| Fold | Region | Seed | Code | Detail |
|---|---|---:|---|---|
| fold_01_3F_Region3 | 3F_Region3 | 41 | missing_artifact | evaluation summary: historical_outputs/slabim_region_cv/folds/fold_01_3F_Region3/seed_41/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_01_3F_Region3/seed_41/evaluation_test/per_frame.csv |
| fold_01_3F_Region3 | 3F_Region3 | 43 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_01_3F_Region3/seed_43/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_01_3F_Region3/seed_43/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_01_3F_Region3/seed_43/evaluation_test/per_frame.csv |
| fold_02_4F_Region2 | 4F_Region2 | 41 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_02_4F_Region2/seed_41/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_02_4F_Region2/seed_41/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_02_4F_Region2/seed_41/evaluation_test/per_frame.csv |
| fold_02_4F_Region2 | 4F_Region2 | 43 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_02_4F_Region2/seed_43/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_02_4F_Region2/seed_43/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_02_4F_Region2/seed_43/evaluation_test/per_frame.csv |
| fold_03_4F_Region3 | 4F_Region3 | 41 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_03_4F_Region3/seed_41/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_03_4F_Region3/seed_41/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_03_4F_Region3/seed_41/evaluation_test/per_frame.csv |
| fold_03_4F_Region3 | 4F_Region3 | 43 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_03_4F_Region3/seed_43/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_03_4F_Region3/seed_43/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_03_4F_Region3/seed_43/evaluation_test/per_frame.csv |
| fold_04_5F_Region2 | 5F_Region2 | 41 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_04_5F_Region2/seed_41/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_04_5F_Region2/seed_41/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_04_5F_Region2/seed_41/evaluation_test/per_frame.csv |
| fold_04_5F_Region2 | 5F_Region2 | 43 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_04_5F_Region2/seed_43/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_04_5F_Region2/seed_43/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_04_5F_Region2/seed_43/evaluation_test/per_frame.csv |
| fold_05_5F_Region3 | 5F_Region3 | 41 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_05_5F_Region3/seed_41/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_05_5F_Region3/seed_41/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_05_5F_Region3/seed_41/evaluation_test/per_frame.csv |
| fold_05_5F_Region3 | 5F_Region3 | 43 | missing_artifact | selected checkpoint: historical_outputs/slabim_region_cv/folds/fold_05_5F_Region3/seed_43/best.pt; evaluation summary: historical_outputs/slabim_region_cv/folds/fold_05_5F_Region3/seed_43/evaluation_test/summary.json; per-frame evaluation: historical_outputs/slabim_region_cv/folds/fold_05_5F_Region3/seed_43/evaluation_test/per_frame.csv |
