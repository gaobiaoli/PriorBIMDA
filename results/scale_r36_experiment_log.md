# Continuous scale+r36 experiment log

更新日期：2026-09-04（Pacific/Auckland）

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
| + detached second-pass DINO all-layer token adapter（过宽消融） | 0.064381 | **0.062143** | 0.089244 |
| + detached second-pass r36-shortcut adapter | **0.063343** | 0.062976 | 0.087628 |
| calibrated disagreement 改注入 `projected P36` | 0.063831 | 0.062332 | 0.089552 |

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

该实验实际把第二遍 adapter 加在全部四层 DINO token 上，会连带改变 F18，并不对应“只增强
r36 lateral shortcut”的原始假设；保留为过宽融合负消融。

## Detached second-pass r36-shortcut adapter

严格实验定义：F18 及其 `projected[3]` 输入完全保持第一遍原路径。第二遍仍使用同一个
RGB+BIM early-fusion DINO；`stopgrad(c)` 校准 DA3 后只重构 condition 的 disagreement
channel。仅将第二遍 DINO stage 2 映射到 36x36 DPT lateral shortcut，整个第二遍
DINO、reassemble、projection 和 `fusion36.residual_layer1` 坡度路径均 detach。然后通过
spatial `Conv3x3(128→64)→GELU→Conv1x1(64→128)` zero-init adapter，与第一遍 r36
shortcut 相加。原 calibrated-disagreement adapter、r36 decoder 与其他协议不变。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_detached_second_pass_r36_shortcut_adapter_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_detached_second_pass_r36_shortcut_adapter/best.pt`
- checkpoint SHA-256：
  `e58e12bc0990caabe88195b6dfb9e4c06031caeb2ef84ec3aeea136b8705175f`
- best epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：4204.42 s。
- 验证：人为注入非零 r36 shortcut delta 后，F18 bitwise 不变、F36 改变；zero-init
  adapter 初始输出精确为零。

Area_1 pixel-micro：

| split | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|
| validation（1673 帧） | 0.063343 | 0.633401 | 0.198097 | 0.943124 | 0.977181 | 0.143636 |
| test（1641 帧） | 0.062976 | 0.413753 | 0.126360 | 0.942540 | 0.975564 | 0.166735 |

Matterport3D zero-shot：

| scene | frames | valid pixels | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.096680 | 0.260969 | 0.137873 | 0.917737 | 0.974833 | 0.164342 |
| 759 | 518 | 595,254,869 | 0.076569 | 0.302188 | 0.152801 | 0.930860 | 0.989950 | 0.127650 |
| 1px | 793 | 953,135,152 | 0.088042 | 0.421315 | 0.152687 | 0.928297 | 0.970353 | 0.187393 |
| **aggregate** | **1935** | **2,232,131,543** | **0.087628** | **0.347866** | **0.148179** | **0.925746** | **0.976951** | **0.166173** |

相对当前锚点，Area_1 val/test AbsRel 改善 2.42%/1.05%；MP3D aggregate RMSE 改善
0.42%、RMSE-log 改善 0.34%、delta2 提高 0.00058，但 AbsRel 退化 2.48%、MAE 退化
1.80%、delta1 下降 0.00220。`759` AbsRel 小幅改善 0.23%，`hxp` 与 `1px` 分别退化
3.90%/2.88%。严格 scope 比过宽消融的 aggregate AbsRel 改善 1.81%，说明限制 shortcut
范围有效，但仍不足以替代 zero-shot AbsRel 最优的当前锚点。

## Calibrated disagreement 注入 projected P36

该实验保持当前锚点的 `C36=[z_s,|z_s|,M]`、`3→32→3×ResBlock→64→128`
zero-init adapter、`128→64→32→1` r36 decoder、loss、数据与 6-epoch schedule 不变，
只把 adapter delta 的注入位置从后融合

`F36' = F36 + A(C36)`

移动到 DPT lateral input：

`P36' = projected[2] + A(C36)`，再执行
`Fuse36(Up(F18), residual_layer1(P36'))`。F18 路径完全不变。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_projected_p36_injection_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_projected_p36_injection/best.pt`
- checkpoint SHA-256：
  `a34024584a362361fbdb765d48d77b9e3fe6de10ee8ff2556b3070a7d19708f0`
- best epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：3691.61 s。
- 验证：zero-init delta 精确为零；人为注入非零 P36 delta 后 F18 bitwise 不变、F36 改变。

Area_1 pixel-micro：

| split | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|
| validation（1673 帧） | 0.063831 | 0.632590 | 0.199805 | 0.941989 | 0.977692 | 0.143720 |
| test（1641 帧） | 0.062332 | 0.413547 | 0.125796 | 0.944654 | 0.975580 | 0.165874 |

Matterport3D zero-shot：

| scene | frames | valid pixels | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.100569 | 0.262016 | 0.141508 | 0.915976 | 0.974758 | 0.165915 |
| 759 | 518 | 595,254,869 | 0.076148 | 0.301725 | 0.151364 | 0.934043 | 0.989214 | 0.127612 |
| 1px | 793 | 953,135,152 | 0.090020 | 0.421058 | 0.153789 | 0.926356 | 0.969441 | 0.187880 |
| **aggregate** | **1935** | **2,232,131,543** | **0.089552** | **0.347866** | **0.149381** | **0.925226** | **0.976343** | **0.166876** |

相对后融合 F36 锚点，Area_1 val/test AbsRel 改善 1.67%/2.06%，MP3D aggregate RMSE
改善 0.42%；但 zero-shot AbsRel 退化 4.73%、MAE 退化 2.62%、RMSE-log 退化 0.09%，
delta1 下降 0.00272。仅 `759` AbsRel 改善 0.78%，`hxp` 与 `1px` 分别退化
8.08%/5.19%。因此 P36 注入是训练域收益、跨域负结果，不替代 F36 后融合锚点。

## 后融合 F36 calibrated-disagreement + RGB 六通道 adapter

保持当前锚点的后融合位置、3 个 residual blocks、`32→64→128` zero-init adapter、
`128→64→32→1` r36 decoder、loss、数据与 6-epoch schedule 不变。只将 residual
condition 从

`C36=[z_s,|z_s|,M]`

扩展为

`C36=[z_s,|z_s|,M,R,G,B]`。

其中 RGB 保持 `[0,1]` 并以 adaptive average pooling 降到 36×36；adapter 第一层相应由
`Conv3x3(3→32)` 改为 `Conv3x3(6→32)`，delta 仍注入融合后的 F36。真实 CUDA 前向确认
condition 为 `B×6×36×36`，adapter 输入为 6 通道，zero-init delta 精确为零。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_rgb6_adapter_32_64_128_decoder_128_64_32_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_rgb6_adapter_32_64_128_decoder_128_64_32/best.pt`
- checkpoint SHA-256：
  `8ebea5e33f85be96e5243b459fb8dbfcc8b481f26b74d0f8254b79f9d32470da`
- best epoch：3；optimizer steps：2634；skipped steps：0；训练耗时：3685.33 s。

Area_1 pixel-micro：

| split | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|
| validation（1673 帧） | 0.063771 | 0.635074 | 0.198717 | 0.942345 | 0.977018 | 0.145168 |
| test（1641 帧） | 0.063869 | 0.416355 | 0.129774 | 0.945394 | 0.975961 | 0.167407 |

Matterport3D zero-shot：

| scene | frames | valid pixels | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.110143 | 0.267022 | 0.152044 | 0.900374 | 0.973309 | 0.169827 |
| 759 | 518 | 595,254,869 | 0.082000 | 0.308434 | 0.159554 | 0.925147 | 0.989322 | 0.133970 |
| 1px | 793 | 953,135,152 | 0.090521 | 0.424245 | 0.155061 | 0.925611 | 0.969419 | 0.187161 |
| **aggregate** | **1935** | **2,232,131,543** | **0.094259** | **0.352228** | **0.155335** | **0.917757** | **0.975918** | **0.169052** |

相对当前三通道后融合 F36 锚点，Area_1 validation AbsRel 改善 1.76%，但 test AbsRel
退化 0.35%。MP3D aggregate AbsRel 退化 10.23%、RMSE 退化 0.83%、MAE 退化
6.71%、RMSE-log 退化 1.39%，delta1 下降 0.01019，delta2 下降 0.00045。三个场景
AbsRel 均没有超过锚点，因此该 RGB 六通道方案是明确的 zero-shot 负结果，不替代当前锚点。

## Iterative scale→r18→r36 双独立 geometry encoder

在后融合思路上扩展为显式两阶段空间修正：

`D_s = D_DA3 * exp(c)`

`C18 = [z_s, |z_s|, M] → A18 → F18' → r18`

`D_s18 = D_s * exp(r18)`

`C36 = [z_s18, |z_s18|, M] → A36 → F36' → r36`

`D_final = D_s * exp(r18 + r36)`

其中 A18、A36 是参数完全独立的
`3→32→3×ResBlock(32)→64→128` zero-init geometry encoder，分别后融合到 F18、F36；
两个 residual decoder 均为 `128→64→32→1`。C18 对 global scale stop-gradient，C36
同时对 global scale 和上一级 r18 stop-gradient，避免 r36 condition 分支反向改写 coarse
decomposition；最终深度损失和 native teacher 仍通过直接输出训练 c、r18、r36。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_iterative_geometry_r18_r36_6epoch_continuous_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_iterative_geometry_r18_r36_6epoch_continuous/best.pt`
- checkpoint SHA-256：
  `e31d70c5efd38e3c52ff053f9ad152b91988211ccf7ed0ad98c6f98a1e4c66a0`
- best epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：3724.62 s。
- 验证：C18/C36 分别为 `B×3×18×18`、`B×3×36×36`；两个 adapter 参数不共享，
  zero-init delta 均精确为零。

Area_1 pixel-micro：

| split | stage | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | scale | 0.069651 | 0.658213 | 0.224149 | 0.938886 | 0.976206 | 0.148964 |
| validation | scale+r18 | 0.065271 | 0.637074 | 0.203544 | 0.941698 | 0.977112 | 0.145431 |
| validation | scale+r18+r36 | 0.063862 | 0.635756 | 0.198895 | 0.941706 | 0.977230 | 0.144695 |
| test | scale | 0.067992 | 0.418656 | 0.140387 | 0.943709 | 0.975769 | 0.168467 |
| test | scale+r18 | 0.063880 | 0.414378 | 0.129672 | 0.943080 | 0.975323 | 0.166454 |
| test | scale+r18+r36 | 0.062839 | 0.413095 | 0.126641 | 0.942941 | 0.975399 | 0.165983 |

Matterport3D zero-shot：

| scene | frames | valid pixels | scale AbsRel | scale+r18 AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.103202 | 0.097247 | 0.097052 | 0.261197 | 0.137726 | 0.916646 | 0.973832 | 0.165299 |
| 759 | 518 | 595,254,869 | 0.079641 | 0.076480 | 0.076046 | 0.301841 | 0.152135 | 0.932606 | 0.989640 | 0.127369 |
| 1px | 793 | 953,135,152 | 0.093713 | 0.089313 | 0.089279 | 0.423393 | 0.153753 | 0.928188 | 0.969116 | 0.189067 |
| **aggregate** | **1935** | **2,232,131,543** | **0.092867** | **0.088321** | **0.088131** | **0.348913** | **0.148412** | **0.925831** | **0.976034** | **0.167212** |

相对当前三通道后融合 F36 锚点，Area_1 validation/test AbsRel 分别改善 1.62%/1.27%；
MP3D aggregate RMSE 改善 0.12%，但 AbsRel 退化 3.06%、MAE 退化 1.96%、RMSE-log
退化 0.29%，delta1/delta2 分别下降 0.00211/0.00034，因此仍不替代当前锚点。
跨域 aggregate 从 scale 到 scale+r18 改善 4.89%，而 r36 仅在其上再改善 0.22%，说明该
迭代结构的主要有效信息集中在 coarse r18，第二级 geometry/r36 的边际贡献很小。

## Iterative r18/r36 共享 geometry trunk + 独立 stage heads

沿用上一版的 `scale→r18→r36` 顺序、condition、stop-gradient、Laplacian teacher、两个
`128→64→32→1` residual decoder 及全部训练协议。把两个原本独立的 geometry encoder
重构为：

`shared trunk: 3→32→3×ResBlock(32)→64`

`r18 head: 1×1 Conv(64→128), zero-init`

`r36 head: 1×1 Conv(64→128), zero-init`

共享 trunk 完全卷积，因此可同时处理 18×18 和 36×36 condition；两个 stage head 参数
独立，保留尺度/频段适配能力。geometry 部分参数量由独立版 166,400 降为 91,520，减少
45.0%。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_iterative_shared_geometry_r18_r36_heads_6epoch_continuous_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_iterative_shared_geometry_r18_r36_heads_6epoch_continuous/best.pt`
- checkpoint SHA-256：
  `82b1d2faa74fade1c395ec298d033c6e0c4d10ee91ab2622f09614d7db85f2be`
- best epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：3726.12 s。
- 验证：共享 trunk 只有一份参数；r18/r36 heads 参数不共享；C18/C36 分别为
  `B×3×18×18`、`B×3×36×36`；两个 zero-init delta 均精确为零。

Area_1 pixel-micro：

| split | stage | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | scale | 0.069171 | 0.659427 | 0.224256 | 0.937552 | 0.976150 | 0.148953 |
| validation | scale+r18 | 0.065761 | 0.638246 | 0.205845 | 0.942696 | 0.977048 | 0.145742 |
| validation | scale+r18+r36 | 0.064382 | 0.637119 | 0.201267 | 0.942568 | 0.977198 | 0.144923 |
| test | scale | 0.066617 | 0.417897 | 0.137854 | 0.945879 | 0.975825 | 0.167929 |
| test | scale+r18 | 0.062683 | 0.414767 | 0.127930 | 0.946191 | 0.975816 | 0.166322 |
| test | scale+r18+r36 | 0.061717 | 0.413520 | 0.125460 | 0.946333 | 0.975931 | 0.165779 |

Matterport3D zero-shot：

| scene | frames | valid pixels | scale AbsRel | scale+r18 AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.107191 | 0.100739 | 0.100206 | 0.264795 | 0.141013 | 0.911240 | 0.973097 | 0.168744 |
| 759 | 518 | 595,254,869 | 0.079075 | 0.076033 | 0.075661 | 0.296992 | 0.149073 | 0.937373 | 0.990880 | 0.125229 |
| 1px | 793 | 953,135,152 | 0.095875 | 0.090955 | 0.090884 | 0.424480 | 0.156250 | 0.928179 | 0.969210 | 0.189557 |
| **aggregate** | **1935** | **2,232,131,543** | **0.094861** | **0.089972** | **0.089680** | **0.349198** | **0.149668** | **0.925442** | **0.976179** | **0.168070** |

相对独立 encoder 迭代版，共享版 Area_1 test AbsRel 改善 1.79%，但 validation 退化
0.81%，MP3D aggregate AbsRel 退化 1.76%。相对当前 F36 锚点，Area_1 validation/test
AbsRel 改善 0.82%/3.03%，MP3D RMSE 改善 0.04%，但 zero-shot AbsRel 退化 4.88%、
MAE 退化 2.82%、RMSE-log 退化 0.80%，delta1/delta2 分别下降 0.00250/0.00019。
因此共享 trunk 提高了 Area_1 test 且显著压缩 geometry 参数，但跨域表现不如独立 encoder，
仍不替代当前锚点。共享版中 r36 在 r18 后带来的 aggregate AbsRel 改善为 0.33%。

## Shared iterative geometry：C36 中 c+r18 不 detach

该实验严格继承共享 geometry trunk 版本，只改变第二阶段 iteration condition 的梯度：

- C18 仍使用 `stopgrad(c)`，因此 r18 condition 不反向调整 global scale；
- C36 从 `geometry(stopgrad(c+r18))` 改为 `geometry(c+r18)`；
- 恢复 `L_r36 → A36(C36) → c,r18` 路径；
- F18/F36 DPT backbone、共享 `3→32→3×ResBlock→64` trunk、两个独立
  `64→128` stage heads、residual decoders、loss、数据和 schedule 均不变。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_iterative_shared_geometry_r18_r36_heads_nondetached_6epoch_continuous_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_iterative_shared_geometry_r18_r36_heads_nondetached_6epoch_continuous/best.pt`
- checkpoint SHA-256：
  `da98bb40b45708cc36acb4c9e1fd420b6774aaf6d0fd74412852fde18de521a0`
- best epoch：5；optimizer steps：2634；skipped steps：0；训练耗时：3731.75 s。
- 真实前向梯度审计：C18 `requires_grad=False`，C36 `requires_grad=True`；C36 signed
  channel 对 c/r18 的梯度绝对和分别为 0.66650/0.66667；两个 adapter delta 初始仍精确为零。

Area_1 pixel-micro：

| split | stage | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | scale | 0.069353 | 0.660006 | 0.224302 | 0.938104 | 0.976437 | 0.148690 |
| validation | scale+r18 | 0.064772 | 0.634657 | 0.202664 | 0.942848 | 0.977531 | 0.144106 |
| validation | scale+r18+r36 | 0.063651 | 0.633403 | 0.198881 | 0.942752 | 0.977731 | 0.143376 |
| test | scale | 0.067370 | 0.418146 | 0.139008 | 0.944619 | 0.975500 | 0.168072 |
| test | scale+r18 | 0.063637 | 0.413813 | 0.129819 | 0.945284 | 0.975309 | 0.165433 |
| test | scale+r18+r36 | 0.062744 | 0.412256 | 0.127183 | 0.945189 | 0.975261 | 0.164850 |

Matterport3D zero-shot：

| scene | frames | valid pixels | scale AbsRel | scale+r18 AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.109592 | 0.106721 | 0.106227 | 0.270872 | 0.151015 | 0.909533 | 0.973416 | 0.170604 |
| 759 | 518 | 595,254,869 | 0.079635 | 0.075748 | 0.075108 | 0.294990 | 0.147965 | 0.935616 | 0.990156 | 0.125071 |
| 1px | 793 | 953,135,152 | 0.098147 | 0.093785 | 0.093322 | 0.423201 | 0.158181 | 0.924417 | 0.969411 | 0.189398 |
| **aggregate** | **1935** | **2,232,131,543** | **0.096716** | **0.092938** | **0.092418** | **0.349510** | **0.153261** | **0.922844** | **0.976170** | **0.168537** |

相对 detach 共享版，不 detach 使 Area_1 validation AbsRel 改善 1.13%，但 test 退化
1.66%，MP3D aggregate AbsRel 退化 3.05%。相对当前 F36 锚点，zero-shot AbsRel 退化
8.08%、MAE 退化 5.29%、RMSE-log 退化 1.08%，delta1 下降 0.00510。虽然 r36 在
r18 后的自身边际改善由 0.33% 增至 0.56%，额外 condition 梯度同时破坏了 global scale
与 coarse residual 的跨域校准；因此该实验为负结果，应保留 detach。

## Shared detached iteration + predicted-r18 dynamic r36 teacher

回到共享 trunk、C36 对 `c+r18` detach 的版本，只修改 native r36 teacher。旧 teacher：

`T_r36 = T36_GT - Up(T18_GT)`

新 teacher：

`T_r36 = T36_GT - Up(stopgrad(r18_pred))`

因此 C36 输入状态和 native teacher 都以当前 `r18_pred` 为第一阶段状态。teacher 中的 r18
显式 detach，low2 teacher 不会经 target 反向调整 r18；final-depth loss、r18 native teacher、
zero-mean regularization、encoder、共享 geometry trunk/head、数据和 schedule 均保持不变。
旧 teacher 仍是默认值，新行为由 loss 配置独立开启。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_iterative_shared_geometry_r18_r36_dynamic_teacher_6epoch_continuous_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_iterative_shared_geometry_r18_r36_dynamic_teacher_6epoch_continuous/best.pt`
- checkpoint SHA-256：
  `56cf01fa7b0894156b19b6bbd001c7b6b68d01548783d270ab872e9f835c8971`
- best epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：3726.71 s。
- 数值测试确认：当 r18 prediction 偏离 T18_GT 时，新 target 精确等于预测状态后的 remainder；
  low2 teacher 对 r18 prediction 的梯度为 None。

Area_1 pixel-micro：

| split | stage | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | scale | 0.069304 | 0.656784 | 0.222968 | 0.937876 | 0.976441 | 0.148580 |
| validation | scale+r18 | 0.064687 | 0.637213 | 0.203538 | 0.941878 | 0.977236 | 0.145078 |
| validation | scale+r18+r36 | 0.063159 | 0.635082 | 0.197654 | 0.942488 | 0.977151 | 0.144543 |
| test | scale | 0.067963 | 0.417630 | 0.139772 | 0.943700 | 0.975670 | 0.168384 |
| test | scale+r18 | 0.064911 | 0.414266 | 0.132077 | 0.943583 | 0.974865 | 0.166912 |
| test | scale+r18+r36 | 0.062691 | 0.412317 | 0.125139 | 0.943079 | 0.974965 | 0.166186 |

Matterport3D zero-shot：

| scene | frames | valid pixels | scale AbsRel | scale+r18 AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.110813 | 0.105841 | 0.100294 | 0.264484 | 0.141384 | 0.913425 | 0.973020 | 0.167606 |
| 759 | 518 | 595,254,869 | 0.081908 | 0.077198 | 0.076892 | 0.302070 | 0.152683 | 0.928989 | 0.989652 | 0.128773 |
| 1px | 793 | 953,135,152 | 0.094573 | 0.090929 | 0.090023 | 0.424279 | 0.153453 | 0.924672 | 0.968854 | 0.189569 |
| **aggregate** | **1935** | **2,232,131,543** | **0.096170** | **0.091835** | **0.089667** | **0.350182** | **0.149551** | **0.922378** | **0.975677** | **0.168441** |

新 teacher 达成了预期的局部效果：MP3D 上 r36 对 scale+r18 的边际 AbsRel 改善由旧
teacher 的 0.33% 增至 2.36%，证明理想 T18_GT decomposition 确实压制了第二阶段的补偿
能力。但 r18 阶段本身从 0.089972 退化到 0.091835，最终 aggregate AbsRel 仅从
0.089680 改善到 0.089667（0.014%，可视为持平），同时 RMSE 退化 0.28%、delta1 下降
0.00306。Area_1 validation 改善 1.90%，test 退化 1.58%。相对当前 F36 锚点，MP3D
AbsRel 仍退化 4.86%。因此假设得到验证，但固定 0.5 权重的动态 teacher 没有产生整体
zero-shot 净收益，暂不替代当前锚点或旧共享 teacher。

## F36 后融合锚点：oracle r36 teacher 不减均值

回到当前 F36 后融合锚点，仅移除 oracle-scale spatial teacher 的显式中心化：

`T_r36 = Pool36(log(D_GT) - log(exp(c_oracle) D_DA3))`

不再执行 `T_r36 -= masked_mean(T_r36)`。本实验没有启用 predicted-scale dynamic
teacher；原有 global-scale oracle loss、final-depth loss、r36 teacher 权重 0.5、r36
零均值软正则权重 0.10、adapter/decoder、数据与 6-epoch schedule 均不变。实现开关默认
仍为中心化，避免改变旧实验语义。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_uncentered_r36_teacher_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_uncentered_r36_teacher/best.pt`
- checkpoint SHA-256：
  `4e1d23456edc77c645dd6728dbc1c6d9d0f91c5f44cb8c2b83437470fb3acb46`
- best epoch：5；optimizer steps：2634；skipped steps：0；训练耗时：3689.07 s。
- 数值测试确认：未中心化 teacher 保留 oracle scale 后的 DC residual；中心化默认路径会
  消去该常量分量。相关测试共 34 项通过。

Area_1 pixel-micro：

| split | stage | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | scale | 0.068834 | 0.652950 | 0.219829 | 0.938786 | 0.976510 | 0.147978 |
| validation | scale+r36 | 0.062987 | 0.631518 | 0.197337 | 0.942495 | 0.977685 | 0.142931 |
| test | scale | 0.066679 | 0.417785 | 0.137613 | 0.944633 | 0.975800 | 0.167741 |
| test | scale+r36 | 0.062356 | 0.412484 | 0.125386 | 0.943328 | 0.975505 | 0.165206 |

Matterport3D zero-shot：

| scene | frames | valid pixels | scale AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.107430 | 0.105043 | 0.263980 | 0.148680 | 0.915353 | 0.974751 | 0.165737 |
| 759 | 518 | 595,254,869 | 0.082806 | 0.078276 | 0.304033 | 0.156263 | 0.931290 | 0.989773 | 0.128738 |
| 1px | 793 | 953,135,152 | 0.093552 | 0.089598 | 0.419189 | 0.154121 | 0.929116 | 0.969981 | 0.185603 |
| **aggregate** | **1935** | **2,232,131,543** | — | **0.091310** | **0.347893** | **0.153025** | **0.925480** | **0.976720** | **0.165963** |

相对原 F36 锚点，Area_1 validation/test AbsRel 分别改善 2.97%/2.03%；但 MP3D
aggregate AbsRel 从 0.085511 退化到 0.091310（+6.78%），三个场景均退化，其中 hxp
最明显（+12.89%）。虽然 aggregate RMSE 与 RMSE-log 小幅改善 0.41%/0.46%，AbsRel、
MAE 和 delta1 分别退化 6.78%、5.13% 和 0.00246。预测 residual 的平均绝对均值也从
原锚点 validation/test 的 0.00640/0.00559 增至 0.00831/0.00706。该修改提升域内结果，
但明显损害跨域尺度稳健性，因此不替代原 F36 锚点。

## 当前最优 F36 锚点复训

使用当前代码从官方初始化重新训练原始最优 F36 后融合锚点。复训配置除实验名称与输出
路径外，解析后的 model/loss/train/data 均与原锚点完全一致：seed=42、centered oracle
teacher、r36 零均值软正则 0.10、6 epochs；没有启用 dynamic 或 uncentered teacher。

- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_retrain_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_retrain/best.pt`
- checkpoint SHA-256：
  `a2ecb04d814ff292d2a678a646d954119726e3aa9c084fd825eaedbf50ab5bd8`
- best epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：3686.14 s。
- 官方 encoder/DPT 参数初始化逐值一致，BIM projection、r36 decoder output 与
  calibrated-disagreement adapter 均通过 zero-init 审计。

Area_1 pixel-micro：

| split | stage | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | scale | 0.069142 | 0.656784 | 0.222085 | 0.938156 | 0.976245 | 0.148488 |
| validation | scale+r36 | 0.064177 | 0.635679 | 0.199357 | 0.941637 | 0.977032 | 0.145483 |
| test | scale | 0.067792 | 0.418957 | 0.140191 | 0.944661 | 0.975479 | 0.167934 |
| test | scale+r36 | 0.063260 | 0.415709 | 0.127778 | 0.942884 | 0.975759 | 0.166933 |

Matterport3D zero-shot：

| scene | frames | valid pixels | scale AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.101341 | 0.095634 | 0.263400 | 0.136118 | 0.916106 | 0.973433 | 0.166714 |
| 759 | 518 | 595,254,869 | 0.081117 | 0.078894 | 0.309334 | 0.155678 | 0.922078 | 0.986858 | 0.133472 |
| 1px | 793 | 953,135,152 | 0.097156 | 0.091222 | 0.425401 | 0.157106 | 0.925500 | 0.969127 | 0.190481 |
| **aggregate** | **1935** | **2,232,131,543** | — | **0.089285** | **0.352198** | **0.150296** | **0.921710** | **0.975175** | **0.169580** |

相对第一次锚点训练，本次复训 Area_1 validation/test AbsRel 分别改善 1.14%/0.61%，
但 MP3D aggregate AbsRel 从 0.085511 退化到 0.089285（+4.41%）；hxp、759、1px
分别退化 2.78%、2.80%、6.59%。aggregate RMSE、MAE、RMSE-log 分别退化
0.82%、3.25%、1.71%，delta1/delta2 分别下降 0.00623/0.00120。结果说明锚点的域内
性能可复现，但当前只有一次训练的 zero-shot 峰值没有被本次复训重现；原 checkpoint
仍保持所有已测版本中的最佳 MP3D zero-shot AbsRel。

## 独立历史源码复现（commit cbd3cef）

为排除当前源码后续修改的影响，在独立 worktree
`/home/bgao491/PriorBIMDA_anchor_cbd3cef` 中检出历史提交
`cbd3cefd3f5c8b91fb8cd12143d3cf6deaaaa047`，使用该提交中原始 F36 锚点配置从官方
初始化重新训练 6 epochs。训练和评测均显式设置
`PYTHONPATH=/home/bgao491/PriorBIMDA_anchor_cbd3cef/src`；数据目录通过软链接指向主目录的
同一份 `data/processed`，没有复制或更换数据。

- 历史源码目录：`/home/bgao491/PriorBIMDA_anchor_cbd3cef`
- 配置：
  `configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32_full_depth_metric_da3.yaml`
- checkpoint：
  `outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32/best.pt`
- checkpoint SHA-256：
  `61df0a082f4efe44cb7867196855f8dc782e5b5e7256b3463e56199a2d94ca3c`
- best epoch：6；optimizer steps：2634；skipped steps：0；训练耗时：3684.15 s。
- manifest SHA-256：
  `f78f5cb9c61065c3278f8d93925f9eda8e5a6102787179b0b67a08806864f225`
- split annotation SHA-256：
  `18f4e68838f24ee10feba23f66d4baddd005e5eac5ed5288a459224152c0ed59`

Area_1 pixel-micro：

| split | frames | stage | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| validation | 1673 | scale | 0.069371 | 0.656074 | 0.221315 | 0.937606 | 0.976429 | 0.148733 |
| validation | 1673 | scale+r36 | 0.064581 | 0.639037 | 0.199921 | 0.940155 | 0.976783 | 0.146330 |
| test | 1641 | scale | 0.067600 | 0.419747 | 0.140178 | 0.945047 | 0.975456 | 0.169714 |
| test | 1641 | scale+r36 | 0.063207 | 0.416932 | 0.127694 | 0.943336 | 0.975196 | 0.169014 |

Validation AbsRel 轨迹对比：

| epoch | 原锚点 | 历史源码复训 |
|---:|---:|---:|
| 1 | 0.077651 | 0.067430 |
| 2 | 0.066760 | 0.066807 |
| 3 | 0.066897 | 0.065218 |
| 4 | 0.068644 | 0.064820 |
| 5 | 0.065399 | 0.064991 |
| 6 | 0.064915 | 0.064581 |

因此当前 validation 数据与评测路径一致，但训练轨迹及最终权重不一致；历史源码复训的
最终 validation/test AbsRel 相对原锚点分别改善 0.51%/0.69%。

Matterport3D zero-shot：

| scene | frames | valid pixels | scale AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.098427 | 0.093741 | 0.264353 | 0.133811 | 0.917547 | 0.972823 | 0.169009 |
| 759 | 518 | 595,254,869 | 0.080435 | 0.077245 | 0.301028 | 0.150839 | 0.931047 | 0.988901 | 0.129633 |
| 1px | 793 | 953,135,152 | 0.095079 | 0.091552 | 0.428238 | 0.158205 | 0.928133 | 0.969363 | 0.190953 |
| **aggregate** | **1935** | **2,232,131,543** | **0.092199** | **0.088407** | **0.351965** | **0.148768** | **0.925667** | **0.975633** | **0.169709** |

历史源码复训相对当前源码复训的 MP3D aggregate AbsRel 由 0.089285 改善到 0.088407，
说明后续源码差异确实贡献了一部分退化；但仍未重现原锚点的 0.085511（退化 3.39%）。
差异主要来自 1px：0.091552 对原来的 0.085580，退化 6.98%。原 checkpoint 的 SHA-256
为 `ca54375af20c68821f837d00949aff6abef51a3d3bb26bbef78d7b7c13e5b43a`，与本次复训不同。

需要注意：原锚点训练在 2026-09-03 19:00 左右启动，而 `cbd3cef` 在 19:38 提交，处于
该次训练过程中；checkpoint/配置没有保存启动瞬间的 Git SHA 或未提交 diff。因此
`cbd3cef` 是能够恢复的最近历史提交，但无法证明与 19:00 已载入内存的源码逐字节相同。
另外训练虽固定 seed 并启用 cuDNN deterministic、关闭 benchmark，但没有启用
`torch.use_deterministic_algorithms(True)`，且使用 AMP、gradient checkpointing、GPU
attention/backward 与多 worker augmentation，同 seed 不保证 checkpoint 位级一致。

## 原锚点权重 × cbd3cef 源码交叉评测

为区分历史源码与重训权重的影响，保持 `cbd3cef` worktree、历史配置、DA3
`depth-anything/da3metric-large@4010e39f...`、process resolution 504、MP3D 数据和冻结的
三规则帧集不变，仅将 checkpoint 换回第一次锚点训练得到的原权重：

`/home/bgao491/PriorBIMDA/outputs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_calibrated_disagreement_adapter_32_64_128_decoder_128_64_32/best.pt`

checkpoint SHA-256 为
`ca54375af20c68821f837d00949aff6abef51a3d3bb26bbef78d7b7c13e5b43a`。模型严格加载成功，
三个场景的 frame-set SHA-256 均与原评测相同，1935 帧全部成功且 error=0。

| scene | frames | valid pixels | scale AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.101183 | 0.093048 | 0.260583 | 0.132060 | 0.917247 | 0.972956 | 0.165727 |
| 759 | 518 | 595,254,869 | 0.080000 | 0.076744 | 0.302759 | 0.151933 | 0.933952 | 0.989762 | 0.128805 |
| 1px | 793 | 953,135,152 | 0.092020 | 0.085579 | 0.424084 | 0.151269 | 0.931860 | 0.970459 | 0.187189 |
| **aggregate** | **1935** | **2,232,131,543** | **0.091618** | **0.085511** | **0.349343** | **0.145562** | **0.927941** | **0.976371** | **0.166732** |

交叉评测 aggregate final AbsRel 为 `0.0855108319`，原评测记录为 `0.0855110022`，绝对差
仅 `-1.70e-7`；逐场景 AbsRel 差异均小于 `1e-6`。这证明原权重在 `cbd3cef` 源码下可以
恢复原 zero-shot 性能，历史与当前 evaluator 路径不是此前退化的原因。`cbd3cef` 重训权重
的 aggregate `0.088407` 退化来自训练产生了不同 checkpoint；问题应继续定位训练轨迹的
非确定性或原训练启动时未被提交的源码状态，而不是 MP3D 评测实现。

交叉评测结果目录：

- `results/matterport3d/hxp_original_weights_cbd3cef_zero_shot/`
- `results/matterport3d/759_original_weights_cbd3cef_zero_shot/`
- `results/matterport3d/1px_original_weights_cbd3cef_zero_shot/`

## 直接后继提交 1c07d65 独立复现

`cbd3cef` 在当前主线上的直接子提交是
`1c07d652ff17a8b03306862995fb3ac7394eca9b`（commit `903-5`）。在独立 worktree
`/home/bgao491/PriorBIMDA_anchor_1c07d65` 检出该提交，源码文件保持 Git 原样；只新增一个
继承配置，用独立输出目录防止覆盖提交内已有实验产物：

`configs/stanford_area1_f36_anchor_1c07d65_reproduction.yaml`

数据仍通过 `data/processed` 软链接复用主目录的同一份缓存。训练和评测均显式设置
`PYTHONPATH=/home/bgao491/PriorBIMDA_anchor_1c07d65/src`。

### 提交差异审计

- 原 F36 锚点 YAML 在 `cbd3cef` 与 `1c07d65` 中逐字节相同，SHA-256 均为
  `89f6c48283aaca0409ead6b0f710b98ddbaef5f4da3d6c7c1c97d471f574b8b5`。
- 新的 reproduction 配置除 experiment name/output/results path 外，解析后的 data/model/loss/
  train 核心配置哈希与原 YAML 相同：
  `4133bc7184b1b086df3bce0ef117e1942bc66e0f88cc58a5045f603dc6537709`。
- manifest SHA-256 仍为
  `f78f5cb9c61065c3278f8d93925f9eda8e5a6102787179b0b67a08806864f225`；split annotation
  SHA-256 仍为
  `18f4e68838f24ee10feba23f66d4baddd005e5eac5ed5288a459224152c0ed59`。
- `1c07d65` 新增可选的 detached second-pass DINO adapter，并为此将 `_encoded_neck`
  拆为 `_encode_dino` 与 `_decode_dpt_native_features`。原锚点配置没有该字段，解析后该功能
  为关闭；没有增加激活参数或额外 forward pass。
- 固定 seed 的合成 batch 审计中，两个提交的初始 state、前向输出、total/各分项 loss、DPT、
  scale head、r36 decoder 和 disagreement adapter 梯度均逐字节一致。
- DINO backbone 梯度哈希不逐字节一致，但对同一提交重复运行时哈希同样会变化，梯度总和
  仅在约 `1e-9` 尺度波动。因此这不能归因于源码重构，而是现有 GPU attention/backward
  路径自身没有位级确定性。静态和数值审计均未发现原配置下有语义行为改变。
- 相关单元测试 18 项全部通过。

### 训练与 Area_1

训练从相同官方 DAv2 初始化开始，初始化审计全部通过。第一次进程在 epoch 2 checkpoint
写入时被外部终止，随后从完整的 epoch 1 `latest.pt` 恢复；optimizer、scheduler 与 AMP
scaler 均恢复，epoch 2 按固定 epoch seed 重新完整执行。脚本不保存 DataLoader worker/RNG
快照，因此该恢复不承诺逐 batch 位级重放，但没有缺失或重复 optimizer step。

- checkpoint：`outputs/stanford_area1_f36_anchor_1c07d65_reproduction/best.pt`
- checkpoint SHA-256：
  `77028caab5ed5ce9c2f549ed93b87777db586fa5e68ea1cb0812c4ac4ccca7f6`
- best epoch：6；optimizer steps：2634；skipped steps：0。
- 六个 epoch 的计算时间之和：3610.94 s。

Validation AbsRel 轨迹：

| epoch | 原锚点 | cbd3cef 重训 | 1c07d65 重训 |
|---:|---:|---:|---:|
| 1 | 0.077651 | 0.067430 | 0.077727 |
| 2 | 0.066760 | 0.066807 | 0.068142 |
| 3 | 0.066897 | 0.065218 | 0.065957 |
| 4 | 0.068644 | 0.064820 | 0.064129 |
| 5 | 0.065399 | 0.064991 | 0.064019 |
| 6 | 0.064915 | 0.064581 | 0.063933 |

Area_1 pixel-micro：

| split | frames | stage | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| validation | 1673 | scale | 0.069159 | 0.656422 | 0.222387 | 0.938910 | 0.976365 | 0.148201 |
| validation | 1673 | scale+r36 | 0.063933 | 0.636674 | 0.200671 | 0.943269 | 0.977266 | 0.144074 |
| test | 1641 | scale | 0.067597 | 0.418971 | 0.139575 | 0.943317 | 0.975719 | 0.169012 |
| test | 1641 | scale+r36 | 0.063032 | 0.413559 | 0.126312 | 0.943000 | 0.975258 | 0.167054 |

相对原锚点，validation/test AbsRel 分别改善 1.51%/0.96%；相对 `cbd3cef` 重训分别改善
1.00%/0.28%。

### Matterport3D zero-shot

| scene | frames | valid pixels | scale AbsRel | final AbsRel | RMSE | MAE | delta1 | delta2 | RMSE-log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hxp | 624 | 683,741,522 | 0.104122 | 0.098790 | 0.262697 | 0.139275 | 0.914195 | 0.973862 | 0.166945 |
| 759 | 518 | 595,254,869 | 0.079848 | 0.075081 | 0.300946 | 0.150846 | 0.934759 | 0.990257 | 0.125872 |
| 1px | 793 | 953,135,152 | 0.094705 | 0.090362 | 0.423712 | 0.154712 | 0.925265 | 0.969618 | 0.189571 |
| **aggregate** | **1935** | **2,232,131,543** | **0.093628** | **0.088869** | **0.349217** | **0.148953** | **0.924406** | **0.976422** | **0.167654** |

r36 将 aggregate AbsRel 从 `0.093628` 改善至 `0.088869`，相对改善 5.08%。但相对原锚点
`0.085511` 仍退化 3.93%；其中 hxp/1px 分别退化 6.17%/5.59%，759 则改善 2.17%。相对
`cbd3cef` 重训的 `0.088407`，本次 aggregate 再退化 0.52%。因此 `1c07d65` 能得到更好的
Area_1 val/test，却仍未恢复第一次锚点的 MP3D zero-shot 峰值；该结果继续支持“训练轨迹/
checkpoint 差异导致跨域波动”，而不是 evaluator 或原配置被改动。

结果目录：

- `results/matterport3d/hxp_f36_anchor_1c07d65_reproduction_zero_shot/`
- `results/matterport3d/759_f36_anchor_1c07d65_reproduction_zero_shot/`
- `results/matterport3d/1px_f36_anchor_1c07d65_reproduction_zero_shot/`
