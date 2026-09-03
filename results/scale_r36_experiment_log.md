# Continuous scale+r36 experiment log

更新日期：2026-09-03（Pacific/Auckland）

## 固定协议

- Stanford Area_1，continuous training 6 epochs；validation 选最佳 epoch，test 仅作最终评测。
- DA3 输入使用 metric focal correction；深度评测不做 GT scale/affine alignment。
- Matterport3D zero-shot 使用冻结的三个 three-rule frame set：`hxp`、`759`、`1px`；
  汇总为全部场景像素的 pixel-micro 指标。
- 当前显式 disagreement 为
  `z_s = log(D_BIM) - log(D_DA3) - stopgrad(c)`；输入 adapter 前将 `z_s/1.5`
  截断到 `[-1,1]`、`|z_s|/1.5` 截断到 `[0,1]`，再与 BIM hit fraction `M`
  一起 mask-aware pool 到 36x36。

## 关键对照

| 版本 | Area_1 val AbsRel | Area_1 test AbsRel | MP3D 3-scene zero-shot AbsRel |
|---|---:|---:|---:|
| 原始 continuous `scale+r36` baseline | 0.064553 | 0.064848 | 0.088506 |
| calibrated adapter，`3→32→128` | 0.065350 | 0.063944 | 0.089260 |
| adapter 加 3 个 ResBlock，decoder `128→64→1` | 0.065268 | 0.062829 | 0.087861 |
| **adapter `32→64→128`，decoder `128→64→32→1`** | **0.064915** | **0.063646** | **0.085511** |
| + detached second-pass DINO feature adapter | 0.064381 | 0.062143 | 0.089244 |

当前锚点相对原始 continuous baseline 的 MP3D zero-shot：AbsRel 改善 3.38%，MAE 改善
1.78%，RMSE 改善 0.21%，RMSE-log 改善 1.32%，delta1 提高 0.00384。

## 当前锚点：progressive adapter + decoder

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32/best.pt`
- checkpoint SHA-256：
  `ca54375af20c68821f837d00949aff6abef51a3d3bb26bbef78d7b7c13e5b43a`
- 最佳 epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：3689.92 s。
- 结构：`A(C36): 3→32→(3×ResBlock32)→64→128`，最后一层 zero-init；
  `F36' = F36 + A(C36)`；`R(F36'): 128→64→32→1`，最后一层 zero-init。

Area_1 pixel-micro：

| split | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|
| validation（1673 帧） | 0.064915 | 0.638481 | 0.203297 | 0.941430 | 0.976911 | 0.145796 |
| test（1641 帧） | 0.063646 | 0.414871 | 0.128084 | 0.941512 | 0.975323 | 0.167089 |

Matterport3D zero-shot：

| scene | frames | valid pixels | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.093048 | 0.260582 | 0.132060 | 0.917248 | 0.972956 | 0.165726 |
| 759 | 518 | 595,254,869 | 0.076744 | 0.302756 | 0.151934 | 0.933956 | 0.989762 | 0.128801 |
| 1px | 793 | 953,135,152 | 0.085580 | 0.424084 | 0.151271 | 0.931860 | 0.970462 | 0.187189 |
| **aggregate** | **1935** | **2,232,131,543** | **0.085511** | **0.349342** | **0.145563** | **0.927943** | **0.976373** | **0.166731** |

对应 aggregate scale-only AbsRel 为 0.091618；当前 r36 将其改善到 0.085511。

## Detached second-pass DINO feature residual

实验定义：第一遍使用现有 RGB+BIM early-fusion DINO，得到原始 DINO feature 并预测
global log-scale `c`；使用 `stopgrad(c)` 校准 DA3 depth，仅重构 BIM condition 的 disagreement
channel，然后以同一 RGB 和新 condition 重跑同一个 early-fusion DINO。第二遍四层 feature
完全 detach，经共享的 `768→64→768` zero-init token adapter 后，分别与第一遍对应 feature
相加，再进入原 DPT。最终尺度仍使用第一遍的 `c`。其余 adapter、decoder、loss、训练轮数、
数据和评测协议不变。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_detached_second_pass_dino_adapter_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_detached_second_pass_dino_adapter/best.pt`
- checkpoint SHA-256：
  `36072928c5aee66b19545d4b02fb9d68e9dbf42c699fe710896704bda48f222d`
- best epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：4232.56 s。

Area_1 pixel-micro：

| split | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|
| validation（1673 帧） | 0.064381 | 0.635265 | 0.200549 | 0.942664 | 0.977298 | 0.144671 |
| test（1641 帧） | 0.062143 | 0.413792 | 0.125784 | 0.944776 | 0.976282 | 0.165983 |

Matterport3D zero-shot：

| scene | frames | valid pixels | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.099859 | 0.264405 | 0.140500 | 0.914099 | 0.974178 | 0.167448 |
| 759 | 518 | 595,254,869 | 0.075480 | 0.300629 | 0.150483 | 0.930990 | 0.989570 | 0.128007 |
| 1px | 793 | 953,135,152 | 0.090225 | 0.423625 | 0.155984 | 0.927192 | 0.970197 | 0.188709 |
| **aggregate** | **1935** | **2,232,131,543** | **0.089244** | **0.349494** | **0.149774** | **0.924194** | **0.976583** | **0.167823** |

相对当前锚点：Area_1 val/test AbsRel 分别改善 0.82%/2.36%，但 MP3D aggregate AbsRel
退化 4.37%，MAE 退化 2.89%，RMSE-log 退化 0.65%，delta1 下降 0.00375。仅 `759`
AbsRel 改善 1.65%；`hxp` 与 `1px` 分别退化 7.32%/5.43%。因此该结构是明显的
Area_1 拟合增益、跨域泛化负结果，不替代上一锚点。
