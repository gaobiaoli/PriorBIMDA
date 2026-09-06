# Reproducible F36 anchor variants: frame-macro reevaluation

Protocol: use the frozen Matterport3D three-rule frame set; compute every metric
inside each frame first, then take an unweighted arithmetic mean across all 1,935
frames. No scale or affine alignment is applied. Values were recomputed from each
run's saved `per_frame.csv`; inference was not rerun. The unreproducible original
peak checkpoint (`ca54375...`) is intentionally excluded.

| Variant | Checkpoint SHA-256 (prefix) | hxp final AbsRel | 759 final AbsRel | 1px final AbsRel | Old pixel-micro aggregate | Frame-macro aggregate |
|---|---|---:|---:|---:|---:|---:|
| 1c07d65 seed42 batch16 retrain | `8bfe160c` | 0.105236 | 0.078324 | 0.094447 | 0.088164 | **0.093610** |
| cbd3cef reproduction | `61df0a08` | 0.101268 | 0.080050 | 0.096549 | 0.088407 | **0.093654** |
| batch32 reproduced anchor | `061fca9b` | 0.104297 | 0.081230 | 0.094574 | 0.088634 | **0.094137** |
| 1c07d65 reproduction | `77028caa` | 0.107088 | 0.077924 | 0.095761 | 0.088869 | **0.094639** |
| main-tree anchor retrain | `a2ecb04d` | 0.103792 | 0.081574 | 0.096465 | 0.089285 | **0.094842** |
| scale-gradient best/epoch6 | `19dc2cf5` | 0.112526 | 0.080595 | 0.094417 | 0.090918 | **0.096557** |
| uncentered r36 teacher | `4e1d2345` | 0.113003 | 0.080889 | 0.094522 | 0.091310 | **0.096832** |
| scale-gradient epoch5 | `64e5fc12` | 0.114767 | 0.080578 | 0.095585 | 0.091980 | **0.097753** |
| scale-gradient epoch3 | `a6fd54db` | 0.114487 | 0.082854 | 0.095059 | 0.092251 | **0.098057** |

## Aggregate frame-macro metrics

| Variant | Raw AbsRel | Scale AbsRel | Final AbsRel | Final RMSE (m) | Final MAE (m) | Final delta1 | Final delta2 | Final RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1c07d65 seed42 batch16 retrain | 0.133884 | 0.097641 | **0.093610** | 0.252159 | 0.151742 | 0.913377 | 0.973073 | 0.133703 |
| cbd3cef reproduction | 0.133884 | 0.097385 | **0.093654** | 0.253559 | 0.152694 | 0.916581 | 0.972521 | 0.134759 |
| batch32 reproduced anchor | 0.133884 | 0.099973 | **0.094137** | 0.252876 | 0.154176 | 0.916651 | 0.973527 | 0.133454 |
| 1c07d65 reproduction | 0.133884 | 0.099337 | **0.094639** | 0.252948 | 0.153291 | 0.914771 | 0.973238 | 0.134054 |
| main-tree anchor retrain | 0.133884 | 0.099701 | **0.094842** | 0.254376 | 0.154372 | 0.912389 | 0.972075 | 0.135113 |
| scale-gradient best/epoch6 | 0.133884 | 0.101101 | **0.096557** | 0.255294 | 0.157171 | 0.915130 | 0.975154 | 0.133970 |
| uncentered r36 teacher | 0.133884 | 0.100403 | **0.096832** | 0.254418 | 0.157038 | 0.916340 | 0.973738 | 0.134154 |
| scale-gradient epoch5 | 0.133884 | 0.101503 | **0.097753** | 0.256801 | 0.158476 | 0.913718 | 0.974899 | 0.134779 |
| scale-gradient epoch3 | 0.133884 | 0.102167 | **0.098057** | 0.256894 | 0.159436 | 0.913856 | 0.973581 | 0.135149 |

Under frame-macro aggregation, the strongest reproducible checkpoint is the
1c07d65 seed42 batch16 retrain (`8bfe160c...`) at 0.093610 AbsRel, narrowly ahead
of the cbd3cef reproduction at 0.093654. Allowing the disagreement branch to send
gradients into the scale head is consistently negative: its best epoch6 result is
0.096557, 2.57% worse than the batch32 detached-scale anchor at 0.094137.

## Pre-batch-retraining anchor-derived experiments

These are the accessible anchor-derived checkpoints trained before the later
reproduction/batch-size audit. The unavailable-to-reproduce original peak anchor
is still excluded, but its derived architectural experiments are included.

| Variant | Checkpoint SHA-256 (prefix) | hxp final AbsRel | 759 final AbsRel | 1px final AbsRel | Old pixel-micro aggregate | Frame-macro aggregate |
|---|---|---:|---:|---:|---:|---:|
| Detached second-pass r36 shortcut | `e58e12bc` | 0.104846 | 0.079608 | 0.093126 | 0.087628 | **0.093287** |
| Adapter + 3 residual blocks | `e25227bf` | 0.106218 | 0.077746 | 0.094490 | 0.087861 | **0.093790** |
| Iterative independent r18/r36 geometry | `e31d70c5` | 0.105349 | 0.078618 | 0.094754 | 0.088131 | **0.093851** |
| Detached second-pass all-DINO features | `36072928` | 0.108080 | 0.078399 | 0.095250 | 0.089244 | **0.094877** |
| Simple calibrated adapter 3-32-128 | `924ecf98` | 0.107024 | 0.079473 | 0.095615 | 0.089260 | **0.094973** |
| Projected-P36 injection | `a3402458` | 0.108535 | 0.079107 | 0.095271 | 0.089552 | **0.095221** |
| Shared geometry + dynamic r36 teacher | `56cf01fa` | 0.108288 | 0.079848 | 0.095766 | 0.089667 | **0.095543** |
| Shared r18/r36 geometry | `82b1d2fa` | 0.108899 | 0.078705 | 0.096176 | 0.089680 | **0.095602** |
| Shared geometry, non-detached iteration | `da98bb40` | 0.114743 | 0.078657 | 0.099028 | 0.092418 | **0.098642** |
| RGB6 disagreement adapter | `8ebea5e3` | 0.119394 | 0.084785 | 0.095372 | 0.094259 | **0.100285** |

### Pre-batch aggregate frame-macro metrics

| Variant | Raw AbsRel | Scale AbsRel | Scale+r18 AbsRel | Final AbsRel | Final RMSE (m) | Final MAE (m) | Final delta1 | Final delta2 | Final RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Detached second-pass r36 shortcut | 0.133884 | 0.097847 | 0.097847 | **0.093287** | 0.252096 | 0.152483 | 0.916207 | 0.973746 | 0.132754 |
| Adapter + 3 residual blocks | 0.133884 | 0.098849 | 0.098849 | **0.093790** | 0.250869 | 0.150749 | 0.914685 | 0.973237 | 0.133294 |
| Iterative independent r18/r36 geometry | 0.133884 | 0.098506 | 0.094036 | **0.093851** | 0.252250 | 0.152559 | 0.917254 | 0.972689 | 0.133596 |
| Detached second-pass all-DINO features | 0.133884 | 0.100858 | 0.100858 | **0.094877** | 0.253961 | 0.154088 | 0.914325 | 0.973573 | 0.134474 |
| Simple calibrated adapter 3-32-128 | 0.133884 | 0.100071 | 0.100071 | **0.094973** | 0.253513 | 0.153591 | 0.912169 | 0.972998 | 0.134807 |
| Projected-P36 injection | 0.133884 | 0.100436 | 0.100436 | **0.095221** | 0.252591 | 0.153562 | 0.915734 | 0.973336 | 0.133903 |
| Shared geometry + dynamic r36 teacher | 0.133884 | 0.102134 | 0.097862 | **0.095543** | 0.253710 | 0.153906 | 0.912932 | 0.972523 | 0.134820 |
| Shared r18/r36 geometry | 0.133884 | 0.100762 | 0.095862 | **0.095602** | 0.253465 | 0.154118 | 0.915482 | 0.972630 | 0.134786 |
| Shared geometry, non-detached iteration | 0.133884 | 0.102786 | 0.099138 | **0.098642** | 0.256540 | 0.157953 | 0.912017 | 0.972732 | 0.136106 |
| RGB6 disagreement adapter | 0.133884 | 0.105130 | 0.105130 | **0.100285** | 0.257519 | 0.159526 | 0.907199 | 0.973060 | 0.137399 |

Across every accessible anchor-derived experiment except the excluded original
peak checkpoint, the best frame-macro AbsRel is 0.093287 from the detached
second-pass r36-shortcut adapter. It narrowly beats the best later reproduction,
the 1c07d65 seed42 batch16 retrain at 0.093610, by 0.35%.
