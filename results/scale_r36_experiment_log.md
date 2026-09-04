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
