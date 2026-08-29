# BIM Early-Fusion DAv2 Dense Metric Depth

## 1. 实验目的与边界

本实验验证 dense BIM prior 在 DINOv2 token formation 阶段的 zero-initialized
early fusion 能否提升逐像素绝对公制深度。模型直接输出 metric depth，不使用
GT/median/oracle scale alignment，也不把最终结果限制为 DA3 的全局尺度修正。

本轮只训练一个模型，不包含 shuffled BIM、zero BIM、噪声鲁棒性、zero-shot 或其他
附加消融。训练、选模和测试均已完成，此后不再根据 test 结果修改模型。

## 2. 模型与输入

- 模型类：`BIMEarlyFusionDepthAnythingV2`
- 主干：官方 `Depth-Anything-V2-Metric-Indoor-Base-hf`
- Encoder：DINOv2 ViT-B/14，hidden size 768
- Decoder：随 checkpoint 完整加载的官方 metric DPT neck/head
- 输入/输出分辨率：504 x 504
- 参数量：97,923,137；全部参数参与训练
- RGB patch projection：预训练 `Conv2d(3, 768, 14, 14)`
- BIM projection：独立的 `Conv2d(3, 768, 14, 14)`，weight/bias 全零初始化
- 融合位置：RGB 与 BIM patch tokens 相加后，再进入 class token、位置编码和
  DINOv2 Transformer

BIM condition 固定为三个通道：

1. train-only mean/std 标准化后的 BIM log depth；
2. 二值 BIM ray-hit mask；
3. `clip(log(D_BIM / D_DA3), -1.5, 1.5) / 1.5`。

第三通道使用固定 DA3METRIC-LARGE cache，并依据相机内参执行
`D_DA3_metric = D_cache * mean(fx, fy) / 300`。DA3 不参与训练，也不提供 confidence
或 latent feature。BIM log-depth 统计只由 7,013 帧 train split 的有效 BIM hit 计算：
mean `0.4236631011`、std `0.7573384621`，validation/test 不重新估计。

为同时满足“完整加载官方 metric DPT”及“初始化输出逐值等于官方模型”，最终输出沿用
官方 checkpoint 的 metric head（sigmoid 与 20 m `max_depth`），没有在其后另接随机初始化的
inverse-disparity head。

## 3. 初始化审计

正式训练前的四项自动检查全部通过：

| 检查 | 结果 | max abs diff | mean abs diff |
| --- | ---: | ---: | ---: |
| BIM projection 输出为零 | PASS | 0 | 0 |
| 不同 BIM 输入的初始化输出一致 | PASS | 0 | 0 |
| Early-fusion 初始化输出等于官方 DAv2 | PASS | 0 | 0 |
| DPT 预训练参数逐值一致 | PASS | 0 | 0 |

DPT 审计覆盖 10,890,305 个 parameter values；数值容差为 `atol=1e-6`、
`rtol=1e-5`。完整机器可读记录见
`outputs/stanford_area1_bim_early_fusion_dense/initialization_verification.json`。

## 4. 数据与训练协议

- 固定 Stanford Area_1 room split：train 7,013、validation 1,673、test 1,641 帧
- 固定评价/监督支持：GT valid 且 0.2--5.0 m
- 所有指标为 pixel-micro aggregation；validation/test 使用完全相同的像素支持
- 不做任何 test-time scale 或 affine alignment
- 损失：`L_logdepth + 0.5 L_grad`，gradient 仅在相邻 GT 均有效时计算
- 优化器：AdamW，weight decay 0.01，cosine decay
- Encoder LR：`5e-6`
- DPT decoder LR：`5e-5`
- BIM projection LR：`5e-5`
- 物理/有效 batch size：2/2
- FP16 autocast、AMP initial scale 1024、activation checkpointing
- 随机种子：42；训练 10 epochs
- best checkpoint：只依据最低 validation AbsRel，最终 test 仅运行一次

训练耗时 7,834.7 秒（约 2 小时 10 分 35 秒），完成 35,057 次 optimizer update。
AMP 检测到 13 个非有限梯度步并安全跳过，占计划 35,070 步的 0.0371%；保存的模型和
所有 validation 指标均为有限值。

## 5. 逐轮训练记录

| Epoch | Total | Log depth | Grad | Val AbsRel | Val RMSE | Val MAE | Val delta1 | Encoder LR | Decoder LR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.121369 | 0.117935 | 0.006869 | 0.069630 | 0.276313 | 0.143030 | 0.942767 | 4.878e-6 | 4.878e-5 |
| 2 | 0.091208 | 0.087987 | 0.006442 | 0.068366 | 0.270221 | 0.138707 | 0.941261 | 4.523e-6 | 4.523e-5 |
| 3 | 0.083514 | 0.080326 | 0.006376 | 0.066815 | 0.250251 | 0.132936 | 0.948199 | 3.970e-6 | 3.970e-5 |
| 4 | 0.072326 | 0.069188 | 0.006275 | 0.060219 | 0.242254 | 0.119566 | 0.952666 | 3.273e-6 | 3.273e-5 |
| 5 | 0.063790 | 0.060691 | 0.006199 | 0.063864 | 0.248649 | 0.129229 | 0.952008 | 2.501e-6 | 2.501e-5 |
| 6 | 0.056586 | 0.053501 | 0.006171 | 0.056985 | 0.252387 | 0.113730 | 0.954907 | 1.729e-6 | 1.729e-5 |
| 7 | 0.050075 | 0.047013 | 0.006123 | 0.058373 | 0.233821 | 0.112019 | 0.955555 | 1.032e-6 | 1.032e-5 |
| 8 | 0.044551 | 0.041518 | 0.006065 | 0.055625 | 0.233850 | 0.107357 | 0.955795 | 4.788e-7 | 4.788e-6 |
| **9** | **0.040755** | **0.037736** | **0.006037** | **0.054133** | 0.231837 | **0.105240** | **0.956053** | 1.232e-7 | 1.232e-6 |
| 10 | 0.038887 | 0.035876 | 0.006022 | 0.054159 | **0.231469** | 0.105563 | 0.956016 | 1.695e-12 | 1.695e-11 |

epoch 9 的 validation AbsRel 最低，因此冻结并用于最终 validation/test 报告。epoch 10
没有用于选模。

## 6. Validation 与 Test

### Validation（1,673 帧，364,913,264 像素）

| Method | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE_log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Focal-corrected Raw DA3 | 0.089047 | 0.386177 | 0.182532 | 0.933805 | 0.983064 | 0.143596 |
| **BIM Early Fusion Dense** | **0.054133** | **0.231837** | **0.105240** | **0.956053** | **0.988445** | **0.108731** |

Validation AbsRel 相对改善：`39.21%`。

### Test（1,641 帧，404,367,334 像素）

| Method | AbsRel | RMSE | MAE | delta1 | delta2 | RMSE_log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Focal-corrected Raw DA3 | 0.084433 | 0.336196 | 0.161070 | 0.940648 | 0.977480 | 0.167341 |
| **BIM Early Fusion Dense** | **0.060664** | **0.272202** | **0.107563** | **0.945090** | **0.978171** | **0.147537** |

Test AbsRel 相对改善：`28.15%`。全部数字直接来自冻结的 epoch-9 metric prediction，
没有 GT scale alignment 或后处理尺度修正。

## 7. 训练正确性与结论

训练结束后 BIM projection 的 weight L2 norm 为 `1.132207`，bias L2 norm 为
`0.016460`，证明 condition branch 已离开 zero initialization。

结论如下：

1. 网络成功从完整的官方 DAv2 ViT-B/14 encoder 与 metric DPT checkpoint 初始化；
2. zero-init BIM branch 成功学习到非零参数；
3. best validation AbsRel 为 `0.054133`，比 Raw DA3 改善 `39.21%`；
4. 固定 test AbsRel 为 `0.060664`，比 Raw DA3 改善 `28.15%`；
5. 本实验的 dense absolute metric prediction 已成功训练，并在所有六项 validation/test
   指标上优于相同支持的 focal-corrected Raw DA3。

## 8. 复现与产物

运行：

```bash
python scripts/model/train_bim_early_fusion_dense.py \
  --config configs/stanford_area1_bim_early_fusion_dense.yaml
```

主要产物：

- `outputs/stanford_area1_bim_early_fusion_dense/best.pt`
- `outputs/stanford_area1_bim_early_fusion_dense/latest.pt`
- `outputs/stanford_area1_bim_early_fusion_dense/training_history.csv`
- `results/stanford_area1/bim_early_fusion_dense/val_summary.json`
- `results/stanford_area1/bim_early_fusion_dense/test_summary.json`
- `results/stanford_area1/bim_early_fusion_dense/provenance.json`

Checkpoint 固定使用官方 Hugging Face revision
`9560f57a2f07803ba353bb918d6a6e5e005b9277`，SHA256 为
`8b902f5d9a8c8a9520f3a8c6a00afe442b464eff5aaf0cb405d1da721cd9f79f`。
