# 当前深度矫正与学习模型技术说明

## 1. 文档状态与正式协议

本文档描述 BIM-PriorDA3 当前代码、数据划分、五个正式对比模型以及最终实验结果。
所有主结果统一使用 **0.2–5.0 m** 深度范围，不使用 2026-07-31 生成的全距离
诊断结果。

当前正式协议为：

- 六个区域合并为同一数据集，不再按区域分别训练；
- 使用 annotation 引用源文件，不复制原始 RGB、PCD 或 BIM；
- `ignore.txt` 中 90 个错误帧在划分前排除；
- 额外排除 13 帧，保证融合 LiDAR 来源不跨 train/val/test；
- train/val/test 分别为 496/104/108 帧；
- 测试集包含 19,422,086 个有效 GT 像素；
- 所有方法使用完全相同的测试帧与 GT mask；
- 模型选择只使用验证集，测试集不参与调参；
- 当前正式结果为单次 `seed=42` 训练，不报告多 seed 结论。

正式数据文件：

- [全区域 annotation](../data/annotations/slabim_clean_global_v1.jsonl)
- [划分元数据](../data/annotations/slabim_clean_global_v1.meta.json)
- [划分生成脚本](../scripts/data/build_global_split.py)

### 1.1 深度范围的含义

`data.min_depth=0.2` 和 `data.max_depth=5.0` 在数据准备阶段用于：

1. 过滤投影到相机坐标系的 LiDAR GT；
2. 限制 BIM ray-casting 深度；
3. 定义当前模型的目标工作范围。

因此 0.2–5.0 m 不是评测时临时裁剪，而是训练数据、BIM 输入和正式评测共同遵守的
协议。全距离 GT 只作为范围外诊断，不与主表混合。

---

## 2. 正式比较对象

当前论文主表建议保留以下五项：

| 编号 | 正式名称 | 评测来源 | 当前数据上训练的部分 |
|---|---|---|---|
| A | Frozen pretrained DA3 | `base` | 无；DA3 是外部预训练模型 |
| B | DA3 + BIM global scale | `global_scale` | 无 |
| C | DA3 + BIM scale/local direct correction | `previous_scale_local` | 无 |
| D | Frozen-DA3 learned refiner | frozen checkpoint 的 `refined` | 仅学习 refiner |
| E | Partially fine-tuned DA3 + learned refiner | E2E checkpoint 的 `refined` | DA3 末端 head + refiner |

需要注意：

- A 使用了预训练神经网络，不能称为“纯非学习方法”；准确表述是“本项目内冻结”。
- C 是当前最强的主非学习 BIM baseline。
- D 和 E 分别来自两个 checkpoint 的 `refined`，并不是同一次 forward 的两个输出。
- frozen 模型中的 `coarse` 与 `global_scale` 相同，不重复列。
- E2E 模型中的 `coarse` 与 `live_scale` 相同，代码已将其标记为 alias，不重复列。
- `live_da3` 和 `live_scale` 只作为 E2E 内部诊断，可放入补充材料。

---

## 3. 总体数据流

```text
RGB
 │
 ▼
DA3 Metric-Large ────────────────> 原始深度 D0、置信度 C0
 │                                      │
 │                                      ├── A: Frozen pretrained DA3
 │                                      │
 │                     BIM depth DB ────┴──> 固定鲁棒尺度 s
 │                                                    │
 │                                                    ▼
 │                                                Ds = sD0
 │                                                    │
 │                         ┌──────────────────────────┼─────────────────────┐
 │                         │                          │                     │
 │                         ▼                          ▼                     ▼
 │                  B: global scale          C: fixed local field      D/E: learned
 │                                                Da                  residual refiner
 │                                                                         │
 └────────────────────────── RGB/DA3/BIM 多条件输入 ────────────────────────┘
                                                                           │
                                                                           ▼
                                                              Dhat = Ds·exp(r)
```

D 与 E 使用相同的轻量 refiner。二者的主要差异是：

- D 读取冻结并缓存的 DA3 深度；
- E 在线执行 DA3，只微调 decoder 最后阶段，ViT-L backbone 始终冻结。

---

## 4. 模型 A：Frozen pretrained DA3

### 4.1 结构

基础模型为 `depth-anything/da3metric-large`。其主要组成是：

- ViT-L backbone；
- Metric-Large/DPT 风格深度 head；
- 单帧 RGB 输入；
- 输出原始度量深度 \(D_0\)。

DA3 总参数量约为 334,171,394，其中 backbone 为 304,367,616，head 为
29,803,778。

在 frozen 学习基线中，DA3 不在训练计算图中，预测预先缓存到 `.npz`。因此本项目
训练阶段只优化后续 refiner。

### 4.2 输入与输出

\[
D_0 = F_{\mathrm{DA3}}(I_{\mathrm{RGB}}).
\]

该阶段：

- 使用 RGB；
- 不使用 BIM；
- 不读取 GT；
- 不进行区域调参；
- 是后续固定方法的共同起点。

---

## 5. 模型 B：BIM 全局尺度矫正

### 5.1 鲁棒尺度估计

在 DA3 和 BIM 深度都有限且大于零的位置计算：

\[
r_i=\frac{D_B(i)}{D_0(i)}.
\]

只保留：

\[
0.2 < r_i < 5.0.
\]

如果有效比值不少于 100 个，尺度为：

\[
s=Q_{0.45}\left(\{r_i\}\right);
\]

否则退化为 \(s=1\)。输出为：

\[
D_s=sD_0.
\]

### 5.2 特点

- 无训练参数；
- 不使用 RGB 语义，只使用 \(D_0\) 和 BIM depth；
- 只改变整帧尺度，不改变相对空间结构；
- 分位数可降低 BIM 遮挡、位姿误差和异常比例的影响；
- BIM 支持不足时自动退回 DA3。

实现见 [baselines.py](../src/bim_priorda3/baselines.py) 中的
`estimate_bim_scale`。

---

## 6. 模型 C：固定 BIM 局部矫正

### 6.1 固定局部残差

模型 C 先执行模型 B，然后计算：

\[
e(i)=\log D_B(i)-\log D_s(i).
\]

局部场只使用同时满足以下条件的像素：

1. BIM depth 有限且大于零；
2. \(|e(i)|\le 0.10\)；
3. BIM depth 的 Sobel 梯度模长小于 0.25。

令可信掩码为 \(m\)，使用 \(\sigma=64\) 的归一化高斯平滑：

\[
f(i)=
\frac{G_\sigma*(me)}
{\max(G_\sigma*m,10^{-4})}.
\]

当平滑支持小于 0.05 时令 \(f=0\)，并限制：

\[
f\in[-0.10,0.10].
\]

最终结果为：

\[
D_a=D_s\exp(1.25f).
\]

因此局部阶段相对 \(D_s\) 的乘性变化严格限制在：

\[
\frac{D_a}{D_s}\in
[e^{-0.125},e^{0.125}]
\approx[0.8825,1.1331].
\]

### 6.2 在系统中的角色

模型 C：

- 是最强的确定性 BIM baseline；
- 不需要训练或区域调参；
- 训练时作为防退化 anchor；
- 不是学习网络的输入结果。

学习网络仍从 \(D_s\) 开始，而不是把 \(D_a\) 与网络输出加权融合。

实现见：

- `previous_local_correction_features`
- `bim_scale_and_local_features`

文件：[baselines.py](../src/bim_priorda3/baselines.py)

---

## 7. 模型 D：冻结 DA3 的学习式 refiner

### 7.1 输入

网络融合三类条件。

#### RGB 分支：3 通道

\[
X_{\mathrm{rgb}}=I_{\mathrm{RGB}}.
\]

#### DA3 几何分支：4 通道

\[
X_{\mathrm{geo}}=
\left[
\frac{\log D_0}{3},
\frac{\log D_s}{3},
C_0,
\operatorname{clip}
\left(
\frac{\log D_s-\log D_0}{0.5},
-2,2
\right)
\right].
\]

#### BIM 分支：8 通道

\[
X_{\mathrm{BIM}}=
\left[
\frac{\log D_B}{3}M_B,
M_B,
N_x,N_y,N_z,
E_B,
\Delta,
|\Delta|
\right],
\]

其中：

\[
\Delta=(\log D_B-\log D_s)M_B.
\]

这里 \(M_B\) 为 BIM valid mask，\(N\) 为 BIM normal，\(E_B\) 为 BIM edge。

### 7.2 三路独立金字塔

RGB、DA3 geometry 和 BIM 分别使用独立的四层卷积金字塔，避免直接混合统计分布
差异较大的颜色、对数深度与几何法向。

在 504×504 输入下，各层为：

| 层级 | 通道 | 空间尺寸 |
|---|---:|---:|
| level 0 | 16 | 504×504 |
| level 1 | 32 | 252×252 |
| level 2 | 64 | 126×126 |
| bottleneck | 128 | 63×63 |

每层先拼接 RGB 与 geometry，经 `2w → w` 融合模块处理，再加入 BIM 的
1×1 adapter：

\[
F_l=
H_l([F_l^{rgb},F_l^{geo}])
A_l(F_l^{BIM}).
\]

BIM adapter 和所有残差输出 head 都以零初始化开始，因此初始网络严格满足：

\[
\hat D=D_s.
\]

当前正式模型没有启用额外 BIM adapter gate。

### 7.3 U-Net 解码与三类残差

解码路径为：

\[
128\rightarrow64\rightarrow32\rightarrow16.
\]

网络预测三个有界对数残差：

| 分支 | 形式 | 单分支上限 | 作用 |
|---|---|---:|---|
| frame | bottleneck 全局池化后的标量 | 0.20 | 剩余帧级尺度误差 |
| low | 63×63 低频图上采样 | 0.25 | 缓慢变化的空间误差 |
| detail | 504×504 全分辨率图 | 0.15 | 局部边缘和细节 |

frame head 同时输出 frame trust；detail head 的另外两个通道输出 log variance 和
BIM reliability。

### 7.4 深度感知路由

frame 与 low residual 使用：

\[
g(D_s)=
\operatorname{sigmoid}
\left(
\frac{D_s-1.0}{0.05}
\right).
\]

总残差为：

\[
r=
\operatorname{clip}
\left(
g(D_s)r_{\mathrm{frame}}
+g(D_s)r_{\mathrm{low}}
+r_{\mathrm{detail}},
-0.45,0.45
\right).
\]

最终输出：

\[
\hat D_{\mathrm{frozen}}
=
\operatorname{clip}
\left(
D_s\exp(r),
10^{-3},
10
\right).
\]

0.2–5.0 m 正式评测不会触及 GT 范围之外，但 10 m 输出上限可以抑制异常预测。

### 7.5 参数量

- refiner 总参数：2,521,654；
- 当前数据上全部可训练；
- frozen checkpoint 约 30 MB；
- 部署仍需 DA3 权重或事先生成的 DA3 cache。

实现：

- [system.py](../src/bim_priorda3/models/system.py)
- [refiner.py](../src/bim_priorda3/models/refiner.py)

---

## 8. 模型 E：部分端到端 DA3 + refiner

### 8.1 在线 DA3

模型 E 不再读取训练时缓存的 DA3 depth，而是从当前 RGB 在线执行 DA3：

\[
D_{\mathrm{live}}=F_{\mathrm{DA3}}(I_{\mathrm{RGB}}).
\]

RGB 使用 DA3 官方归一化。ViT-L backbone 在 `torch.no_grad()` 下运行，并始终
冻结；只允许 decoder 最后阶段产生梯度。

### 8.2 可训练 DA3 模块

`trainable_scope=last_stage` 对应：

- `head.scratch.refinenet1`
- `head.scratch.output_conv1`
- `head.scratch.output_conv2`

参数量：

| 部分 | 总参数 | 可训练参数 |
|---|---:|---:|
| DA3 backbone | 304,367,616 | 0 |
| DA3 head | 29,803,778 | 2,758,081 |
| refiner | 2,521,654 | 2,521,654 |
| 完整系统 | 336,693,048 | 5,279,735 |

可训练比例为 1.568%。因此该版本应称为 **partial decoder fine-tuning**，不能称为
full DA3 fine-tuning。

### 8.3 Detached BIM scale

对 \(D_{\mathrm{live}}\) 使用与模型 B 相同的 0.45 分位数尺度：

\[
s_{\mathrm{live}}
=Q_{0.45}
\left(
\frac{D_B}{D_{\mathrm{live}}}
\right),
\qquad
D_{\mathrm{ls}}=s_{\mathrm{live}}D_{\mathrm{live}}.
\]

尺度估计在 `no_grad()` 中完成，因此：

- 不会通过 quantile 对 BIM 比值反向传播；
- \(s_{\mathrm{live}}\) 作为 detached 常量；
- 梯度仍可通过 \(D_{\mathrm{ls}}=s_{\mathrm{live}}D_{\mathrm{live}}\) 回到 DA3 head。

DA3 confidence 也作为 detached 条件，不形成第二条 decoder 梯度路径。

### 8.4 最终预测

refiner 结构与模型 D 相同，只把几何分支中的 \(D_0,D_s,C_0\) 替换成在线的
\(D_{\mathrm{live}},D_{\mathrm{ls}},C_{\mathrm{live}}\)：

\[
\hat D_{\mathrm{E2E}}
=D_{\mathrm{ls}}\exp(r_{\mathrm{live}}).
\]

E2E checkpoint 包含 DA3 和 refiner 参数，约 1.3 GB。

### 8.5 E2E 内部诊断阶段

| summary key | 含义 |
|---|---|
| `live_da3` | 微调后在线 DA3 的 RGB-only depth |
| `live_scale` | `live_da3` 加 detached BIM global scale |
| `coarse` | 与 `live_scale` 完全相同的 alias |
| `refined` | 在线 DA3、BIM scale 与 learned refiner 的最终结果 |

当前没有“完全相同在线推理路径但 DA3 head 冻结”的控制组，因此不能把
`live_da3` 与历史 cached `base` 的差异全部归因于 decoder 微调。最稳妥的增益
比较是最终模型 D 与模型 E，同时明确二者也改变了 DA3 在线路径。

---

## 9. 可靠度、鲁棒训练与损失

### 9.1 Reliability 是辅助任务

网络预测像素级 BIM reliability 和帧级 trust，但它们不直接乘到深度残差上。
这样可以学习 BIM 可靠性表征，同时避免错误 trust 形成硬门控。

可靠度目标比较 BIM 与尺度矫正 DA3 相对 GT 的误差。E2E 训练时目标基于在线
scaled depth 重新生成。

### 9.2 BIM 扰动

| 扰动 | 概率 | 幅度 |
|---|---:|---:|
| BIM 平移 | 0.20 | 横纵最多 ±4 px |
| 局部 BIM dropout | 0.15 | 约 12% 面积 |
| 整帧 BIM dropout | 0.03 | 全部 BIM 条件置空 |
| BIM 对数深度噪声 | 0.20 | 标准差 0.02 |
| BIM edge dilation | 0.15 | 3×3 |
| RGB horizontal flip | 0.50 | — |
| RGB color jitter | 每帧 | 0.10 |

当 BIM depth 或 valid 被改变时，训练代码会重新计算 scale 与 direct BIM anchor，
保持输入和监督一致。

### 9.3 损失组成

总损失包括：

- 对数深度误差；
- 对数深度梯度误差；
- 总残差 teacher；
- frame residual teacher；
- low/local residual teacher；
- low-frequency smoothness；
- detail regularization；
- pixel/frame trust；
- heteroscedastic uncertainty；
- 相对 BIM direct anchor 的 degradation hinge。

正式权重：

| 损失 | 权重 |
|---|---:|
| depth | 1.00 |
| gradient | 0.20 |
| residual teacher | 0.50 |
| frame residual teacher | 0.25 |
| local residual teacher | 0.25 |
| low smoothness | 0.02 |
| detail regularization | 0.005 |
| pixel trust | 0.10 |
| frame trust | 0.02 |
| uncertainty | 0.01 |
| degradation | 0.25 |

1 m 内像素使用 `near_range_boost=2.0`，实际权重乘数为 3。

---

## 10. 当前训练流程

### 10.1 Frozen refiner 预训练

配置：[global clean pretrain](../configs/slabim_pretrain.yaml)

- train：496 帧；
- 18 epochs；
- 416×416 随机 crop；
- batch size 16；
- 每个 epoch 完整打乱并遍历 496 帧；
- 不做区域平衡或过采样；
- 学习率 \(4.5\times10^{-4}\)；
- DA3 使用固定 cache；
- 峰值显存约 15.27 GiB。

### 10.2 Frozen refiner 全分辨率微调

配置：[global clean final](../configs/slabim.yaml)

- 从预训练最佳 checkpoint 初始化；
- 最多 12 epochs，实际 early-stop 后完成 11 epochs；
- 504×504；
- batch size 8；
- 学习率 \(4\times10^{-5}\)；
- 启用 1 m depth-aware routing；
- 加强近距离监督。

### 10.3 Partial E2E 微调

配置：[global clean E2E](../configs/slabim_e2e.yaml)

- 从 frozen final `accepted.pt` 初始化 refiner；
- DA3 使用固定官方 revision `4010e39f...`；
- 8 epochs；
- 504×504；
- micro batch 4；
- gradient accumulation 2，有效 batch 8；
- refiner learning rate \(1\times10^{-5}\)；
- DA3 last-stage learning rate \(1\times10^{-6}\)；
- warmup 1 epoch，之后 cosine decay；
- AdamW，weight decay \(10^{-4}\)；
- AMP、gradient clip 1.0；
- 峰值显存约 8.57 GiB；
- 最佳验证结果出现在第 5 个 epoch。

---

## 11. 正式 0.2–5.0 m 测试结果

### 11.1 主表

测试集为固定的 108 帧、19,422,086 个有效像素。

| 方法 | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| A. Frozen pretrained DA3 | 0.199347 | 0.421674 | 0.311094 | 0.763276 | 0.985882 | **0.995677** |
| B. DA3 + BIM global scale | 0.082008 | 0.276415 | 0.128441 | 0.927718 | 0.985259 | 0.994684 |
| C. BIM direct scale/local | 0.081447 | 0.283057 | 0.129390 | 0.923850 | 0.983165 | 0.994528 |
| D. Frozen-DA3 learned refiner | 0.062113 | **0.232572** | 0.096893 | 0.964287 | **0.990969** | 0.994560 |
| **E. Partial E2E learned refiner** | **0.061331** | 0.237706 | **0.096260** | **0.966171** | 0.990638 | 0.994484 |

主结论以 AbsRel 为预先规定的主要指标，同时完整报告 RMSE、MAE 和三个
阈值准确率，避免用单一指标掩盖权衡。

### 11.2 关键改善

以模型 C 为主非学习 baseline：

\[
\frac{0.081447-0.061331}{0.081447}
=24.70\%.
\]

即 E2E 最终模型的 AbsRel 相对 BIM direct 降低 24.70%，且 RMSE、MAE、δ1、
δ2 同时改善。δ3 从 0.994528 轻微变为 0.994484，因此不能写成“所有指标都最佳”。

模型 E 相对模型 D：

- AbsRel：改善 1.26%；
- MAE：小幅改善；
- δ1：小幅改善；
- RMSE：恶化约 2.21%。

因此：

- 若论文主指标为 AbsRel，模型 E 是当前最佳；
- 若实践更重视 RMSE、checkpoint 大小和部署简洁性，模型 D 更均衡；
- decoder 微调带来的增益较小，主要收益仍来自 BIM scale 与 learned refiner。

### 11.3 E2E 内部阶段

| 阶段 | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ1 ↑ |
|---|---:|---:|---:|---:|
| cached frozen DA3 `base` | 0.199347 | 0.421674 | 0.311094 | 0.763276 |
| live partially tuned DA3 | 0.197001 | 0.446040 | 0.321129 | 0.759430 |
| live DA3 + detached BIM scale | 0.081557 | 0.284866 | 0.129680 | 0.931760 |
| final E2E refiner | **0.061331** | **0.237706** | **0.096260** | **0.966171** |

`live_da3` 与 cached `base` 的预处理路径不同，这张表用于解释数据流，不作为严格的
单变量 decoder ablation。

### 11.4 分距离 AbsRel

| GT 距离 | 像素数 | BIM direct | E2E refined | 相对改善 |
|---|---:|---:|---:|---:|
| [0.2,1) m | 5,688,044 | 0.086182 | **0.072776** | 15.56% |
| [1,2) m | 8,732,144 | 0.079720 | **0.055554** | 30.31% |
| [2,3) m | 3,334,250 | 0.075485 | **0.051214** | 32.15% |
| [3,5) m | 1,666,828 | 0.086256 | **0.072765** | 15.64% |

E2E 在四个距离段的 AbsRel 都优于 BIM direct。近距离 `[0.2,1)` 的 E2E RMSE
为 0.120268，略差于 BIM direct 的 0.117493，因此分段结论也不能扩写成
“所有指标均优”。

分段采用左闭右开区间。GT 恰好为 5.0 m 的 820 个像素进入 overall，但不进入
`[3,5)` 分段。

---

## 12. 结果文件与复现

### 12.1 正式 checkpoint

- Frozen learned checkpoint：`outputs/slabim/accepted.pt`
- E2E learned checkpoint：`outputs/slabim_e2e/accepted.pt`

`outputs/` 不进入 Git；两个权重的完整 SHA-256、大小、发布状态和待填的下载
URL 见 [`results/manifest.json`](../results/manifest.json)。

### 12.2 正式测试结果

- [Frozen learned summary](../results/slabim/frozen_test_summary.json)
- [E2E summary](../results/slabim/e2e_test_summary.json)
- [E2E per-frame CSV](../results/slabim/e2e_test_per_frame.csv)

E2E summary 的 checkpoint/data provenance 已验证，测试时没有额外 inference override，
也没有测试后再应用 ignore filter。

### 12.3 训练命令

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/slabim_pretrain.yaml \
  --device cuda

.venv/bin/python scripts/model/train.py \
  --config configs/slabim.yaml \
  --init-checkpoint \
  outputs/slabim_pretrain/accepted.pt \
  --device cuda

.venv/bin/python scripts/model/train.py \
  --config configs/slabim_e2e.yaml \
  --init-checkpoint \
  outputs/slabim/accepted.pt \
  --device cuda
```

### 12.4 正式 E2E 测试命令

```bash
.venv/bin/python scripts/model/evaluate.py \
  --config configs/slabim_e2e.yaml \
  --checkpoint \
  outputs/slabim_e2e/accepted.pt \
  --output \
  outputs/slabim_e2e/evaluation_test \
  --device cuda \
  --split test
```

---

## 13. 当前结论与论文表述边界

当前证据支持：

1. BIM global scale 解决了最大的 DA3 尺度误差；
2. 固定 BIM local correction 是可解释且无需训练的强 baseline；
3. frozen learned refiner 融合 RGB、DA3 与原始 BIM 几何后，显著优于 BIM direct；
4. 部分微调 DA3 decoder 可继续小幅改善 AbsRel，但不是主要收益来源；
5. 最终 E2E 模型在 0.2–5.0 m overall 和四个距离段上都实现了更低的 AbsRel。

必须保留的限制：

- 数据来自同一建筑的六个区域，当前是混合后的时间段留出测试，不是未知建筑测试；
- 当前正式训练只有一个 seed；
- E2E 与 frozen 模型没有同在线路径的严格 decoder 单变量控制；
- 性能依赖 camera-to-BIM 位姿和 BIM 可见性；
- 平均指标优势不保证每一帧都优于 BIM direct；
- 全距离实验只说明范围外退化，不进入正式 0.2–5.0 m 主结果。
