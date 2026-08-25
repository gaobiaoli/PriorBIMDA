# Area 1 全景深度联合估计与 BIM 增强：正式评测

> 状态：正式协议与结果，2026-08-15。评测产物使用 schema version 3，正式协议名为
> stanford-area1-regular-and-pano-tangent-depth-v3。
> 本轮主线是 **training-free panorama joint estimation**：不在 Area 1 上训练或微调全景网络。
> 学习式 refiner 虽因兼容旧 evaluator 而存在于原始产物中，但不参与方法选择、主表、主图或
> 结论。BIM 只报告确定性的 universal scale 与 BIM-direct。
> 2026-08-20 新增 `regular→ERP→regular` validation-only candidate protocol；它不修改或
> 覆盖已经冻结的 schema 3 test artifact。

## 1. 结论先行

正式 test 包含 31 个扫描站、7 个互斥房间。主指标为 0.2–5.0 m 内、精确 ERP 球面面积
加权的 station-macro AbsRel；区间为 10,000 次 room-cluster paired bootstrap 的 95% CI。
所有 CI 均表示 candidate − reference，因此负值表示改善。

| 问题 | 结果（split） | 结论 |
|---|---|---|
| ERP 联合后回投到全部原始 regular（val） | 同融合器：0.266820 → 0.186537，下降 30.09%，CI [−0.091333, −0.076494] | 推荐的 regular benchmark 协议；7/7 房间改善 |
| pano tangent 是否改善 regular-only | 0.269850 → 0.228651，下降 15.27%，CI [−0.052933, −0.030209] | 是；30/31 站、7/7 房间改善 |
| tangent 数量 6 → 14 | 0.394129 → 0.363998，下降 7.64%，CI [−0.040528, −0.016809] | 在 tangent-only 的共同 support 上有稳定收益 |
| 球面覆盖率 | 66.92% → 99.75%（6 views）/ 99.96%（14 views） | pano tangent 基本补齐整球，但 coverage 与精度分开报告 |
| BIM 全局尺度 | 0.269850 → 0.108243，下降 59.89% | 确定性 BIM 尺度是强基线 |
| BIM 局部 direct 相对 scale-only | 0.108243 → 0.110064，变差 1.68%，CI [−0.001180, 0.004009] | 未证明局部修正有效；保留 scale-only |
| 真正单帧 + pano tangent | val：0.115857 → 0.164543（6）/0.318592（14） | 覆盖率接近整球，但单帧共同 support 上显著变差 |
| 真正单帧 vs 全部 regular 的 raw joint | 0.170524 → 0.269759，变差 58.19% | 多 regular joint 也并非总优于单帧 |
| scale 后真正单帧 vs 全部 regular joint | 0.126738 → 0.113430，下降 10.50%，CI [−0.030672, 0.001299] | 点估计转为正收益，但 test CI 跨 0，证据不足 |

主结果应表述为：

> 在相同 regular-covered 固定 support 上，validation-only 选定的 joint Huber 融合将
> Area 1 test 的 raw DA3 spherical station-macro AbsRel 从 0.269850 降至 0.228651，
> 相对下降 15.27%；room-cluster paired 95% CI 为 [−0.052933, −0.030209]。

不能表述为多视图三角测量收益：regular 与 pano 共用同一光心，没有视差。

### 1.1 面向原始 regular benchmark 的 round-trip 结论

此前 ERP 主协议和 strict-single 诊断都不能完全回答“全景联合后，数据集原有 regular 图像是否
变得更准”。为此新增 validation-only 协议
`stanford-area1-val-regular-erp-roundtrip-v1`：

1. 对同一 station 的全部原始 regular RGB 分别运行固定 DA3；
2. 利用官方每帧 K 与 W2C，把 regular z-depth 转成 ERP radial range；
3. 在 ERP 上融合全部 regular 来源，并分别测试不加 tangent、加 tangent6、加 tangent14；
4. 把联合 radial range 严格反投影为每张原始 regular 的 z-depth；
5. 在每张 regular 原有的完整 0.2–5.0 m `gt_valid` 上评测，所有方法像素完全一致。

该实验覆盖 validation 的 1,673 张 regular、30 个配对 station、7 个房间和 364,913,264 个
固定 GT 像素。pano GT 从未打开；station 球面预测冻结后才读取 regular GT。

| 方法 | Pixel-micro AbsRel | MAE (m) | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|
| raw DA3（逐帧） | 0.277104 | 0.628398 | 0.792064 | 0.328936 |
| 单帧投影→回投恒等控制 | 0.276401 | 0.626924 | 0.789432 | 0.329974 |
| regular-only joint Huber | 0.259544 | 0.582336 | 0.691244 | 0.238107 |
| regular-only weighted-log | 0.266820 | 0.601536 | 0.708413 | 0.190942 |
| regular + tangent6 weighted-log | 0.223989 | 0.513367 | 0.629726 | 0.383273 |
| **regular + tangent14 weighted-log** | **0.186537** | **0.431850** | **0.557581** | **0.576001** |

主成对比较固定同一个 weighted-log 融合器：tangent14 相对 regular-only 的 AbsRel 下降
30.09%，room-cluster paired 95% CI 为 [−0.091333,−0.076494]，7/7 房间改善；tangent14
相对 tangent6 再下降 16.72%。作为尺度参照，最终候选相对逐帧 raw DA3 下降 32.68%，相对
最佳 regular-only Huber 下降 28.13%。后两项是 validation 点估计，不是额外 confirmatory
假设。

恒等 round-trip 控制只把 0.277104 改为 0.276401（0.25%），且每张图平均原生回投覆盖率
99.997%；tangent14 覆盖率为 100%。因此约 30% 的差异不能由重采样或缺失像素造成。

这项新结果纠正了旧 strict-single 诊断的适用范围：旧实验从一张代表帧出发，再让 6/14 个
tangent 来源压过单一 anchor，且只在该帧映射到 ERP 的窄 support 上评分；它是来源失衡的
压力测试，而不是 regular 数据集主评测。新协议保留旧负结果以说明边界，但后续研究应以
round-trip 协议为主。因为 weighted-log 是在本次 validation 上选出的候选，本结果只能用于
冻结未来 blind protocol，不能事后重跑并替换已经揭盲的旧 test 结论。完整产物见
[round-trip summary](../results/stanford_area1/pano_val_regular_roundtrip/summary.json)。

### 1.2 同一 BIM-scale 基线的 regular→ERP→regular 对照

上节的 raw round-trip 回答“多 regular/pano 来源能否改善 raw DA3”，但不能单独回答
“在已经使用 BIM 尺度修正的同一基线上，ERP 联合还有多少增益”。因此增加独立的
validation-only 协议 `stanford-area1-val-bim-scale-regular-erp-roundtrip-v1`：

1. 每张原始 regular 先用当前统一的 `log_upper_cap_v1` estimator 独立估计 BIM scale；
2. reference 直接在该帧完整 `gt_valid` 上评估 scaled depth；
3. candidate 把完全相同的 scaled depth 投到 ERP，在同一 station 内以冻结的 Huber 规则联合，
   再回投到每张原始 regular；
4. 两侧使用逐帧完全相同的 0.2–5.0 m GT 像素；缺少回投值时回退到该帧 reference；
5. 不读取 pano RGB/GT，不生成 tangent，不加载 checkpoint 或 learned refiner。regular GT 只在
   station 预测冻结后打开。

结果覆盖 1,673 帧、30 个 station、7 个房间与 364,913,264 个固定 GT 像素。逐帧 reference
与原 Area_1 validation 的 `robust_global_scale` 数值在浮点精度内完全一致：

| 方法 | Pixel-micro AbsRel | MAE (m) | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|
| regular universal scale（逐帧） | 0.086442 | 0.174225 | 0.395364 | 0.915328 |
| 单来源投影→回投控制 | 0.086076 | 0.173362 | 0.386833 | 0.915428 |
| regular-scale joint weighted-log | 0.076712 | 0.154612 | 0.348067 | 0.927080 |
| **regular-scale joint Huber（主对比）** | **0.075947** | **0.152537** | 0.353872 | 0.931561 |
| regular-scale synchronized Huber（敏感性） | 0.074459 | 0.158865 | **0.335144** | **0.944022** |

预先固定的主对比 `joint_huber − per_frame` 为 −0.010495 AbsRel，即相对下降 12.14%；7/7
房间改善，10,000 次 room-cluster paired bootstrap 95% CI 为
[−0.012307,−0.006998]。投影→回投控制本身仅下降 0.42%；相对这个控制，joint Huber 仍下降
约 11.77%。joint synchronized Huber 的 AbsRel 更低，但 MAE 高于主 Huber，且它不是本次
预先指定的主方法，只作为敏感性结果，不能据此事后改写主假设。直接与普通 Huber 配对时，
sync-Huber 的 pixel-micro AbsRel 下降 1.96%、RMSE 下降 5.29%、δ1 提高 1.25 个百分点，
但 MAE 增加 4.15%；frame/station/room 胜数分别为 827/1673、14/30、4/7。room-cluster
paired AbsRel 差值 95% CI 为 [−0.006815,0.004085]，跨过 0；room-macro AbsRel 还从
0.076082 增加至 0.076792。因此不能声称残差校准稳定优于普通 Huber。

同一实验也保留 BIM-direct 次要诊断：逐帧 0.087100，经 joint Huber 后 0.078074，AbsRel
下降 10.36%，7/7 房间改善。它说明联合收益不依赖只选 scale-only，但局部 direct 本身是否
优于 scale-only 仍应由独立消融判断。

这项实验正是“同一 baseline 下 regular 与 regular→ERP→regular”的直接答案。它不涉及从
pano RGB 再切 tangent，因此不能拿来衡量新增 pano 图像内容；它衡量的是数据集已有同站多张
regular 之间的球面一致性融合。完整产物见
[BIM-scale round-trip summary](../results/stanford_area1/pano_val_bim_scale_regular_roundtrip/summary.json)。

## 2. 研究范围与名称

### 2.1 本轮实际评测的方法

1. **regular-only**：把同站已有 regular 图像的独立 DA3 预测投到 ERP，并用冻结的融合规则
   F* 聚合。它不是一张图。
2. **regular + tangent6**：在 regular 来源上增加 6 个 pano tangent 预测。
3. **regular + tangent14**：增加嵌套的 14 个 pano tangent 预测；其中包含 tangent6 的全部
   方向，因此是干净的 view-count 对比。
4. **tangent6 / tangent14 only**：仅用 pano RGB 切出的 tangent views，支持 pano-only 站点。
5. **universal scale**：仅使用 DA3 与已注册 BIM 的固定、无 GT 尺度矫正规则。
6. **BIM-direct**：在 scale-only 上增加确定性的局部 BIM 修正链。

tangent6 与 tangent14 均使用 100° FOV、504×504 输入。tangent14 由 6 个轴向视图加 8 个
角向视图组成，几何实现见
[pano_tangent.py](../src/bim_priorda3/data/pano_tangent.py)。

这里的 training-free 指 **没有项目特定的 pano 训练或微调**；DA3 是固定的公开预训练
模型。本轮推荐复现命令不传 --checkpoint，因而不会加载或执行 learned refiner。已经冻结的
历史 schema 3 raw artifact 曾由统一 evaluator 调用带入 accepted.pt；其中 learned 字段只作
archival provenance。本文件不引用其数值，也不把它放进选择、主表或主图。

### 2.2 三个容易误解的基线

| 名称 | 实际含义 | 是否是真正单图 |
|---|---|---:|
| regular-only | 多张 regular 预测的 F* 融合，不加入 pano tangent | 否 |
| single_best_view | 每个 ERP 像素只取一个来源；不同像素可来自不同图 | 否 |
| strict_single_frame | 每站固定选择一张完整 regular 图像 | 是 |

single_best_view 是逐像素单来源拼图，不能在论文中简称为 single image。strict_single_frame
使用无 GT 规则：最大化该帧 base-weight 总和，若并列则按稳定 frame ID 决胜；其评测
support 是所选帧覆盖与有效 pano GT 的交集。平均球面覆盖率只有 validation 10.87%、
test 11.47%，因此它回答的是窄 support 上的真正单帧问题，而不是完整球面问题。

## 3. 数据、几何与深度定义

### 3.1 数据划分

Area 1 共有 190 个 pano station，其中 186 个可按 camera UUID 与 regular views 配对，
4 个为 pano-only。沿用冻结的 room-disjoint 划分：

| split | 房间 | 全部 pano | regular-paired | pano-only |
|---|---:|---:|---:|---:|
| train | 30 | 127 | 125 | 2 |
| validation | 7 | 32 | 30 | 2 |
| test | 7 | 31 | 31 | 0 |

本轮不使用 train 训练 pano 模型。validation 只选择融合规则并冻结协议；test 在冻结后正式
执行一次。两个 validation pano-only 站都来自 hallway_5 的同一房间，只做描述性检查，
不能用于 CI 或跨房间泛化结论。

### 3.2 同中心，不存在视差

2D-3D-S 的 regular 图像由同一扫描位置的 equirectangular 图像采样得到，因此与对应 pano
共用光心。联合估计利用的是方向覆盖、重叠一致性和稳健聚合，不获得额外 baseline，也不能
进行三角测量。该数据生成关系见
[2D-3D-S 原始论文](https://arxiv.org/abs/1702.01105)和
[官方代码](https://github.com/alexsax/2D-3D-Semantics)。

### 3.3 z-depth 与 radial range

regular 深度是沿相机前轴的 z-depth；pano GT 是从光心到表面点的 radial range。设归一化
相机射线为 r，前向分量为 r_z，则投到 ERP 前必须执行：

~~~text
radial_range = z_depth / r_z
~~~

若直接把 z-depth 当 range，离主光轴越远误差越大。所有 raw DA3、universal scale 与
BIM-direct 分支都使用相同转换。GT 原始值按 raw / 512 m 解码，65535 无效；正式深度范围
固定为 0.2–5.0 m。

## 4. Training-free 联合流程

### 4.1 预测与球面融合

1. 固定 DA3 分别推理 regular RGB 和 pano tangent RGB。
2. 用相机内外参把 regular z-depth、tangent z-depth 转成 pano radial range。
3. 投影到相同 ERP 网格，保留每个来源的几何中心度、模型 confidence 和有效性。
4. 在 log-range 域执行 weighted、Huber、photo-Huber 或 synchronization-Huber 候选融合。
5. 在 validation 上只用 raw DA3 的 Route-R 固定 support 选择一次 F*。
6. 冻结 F*、support、深度范围、bootstrap 设置和输入 manifest 后，再执行一次 test。

光度权重只用推理时可见的 RGB；overlap synchronization 只用预测间重叠。二者都不读取
GT。正式选中的 F* 是 joint_huber。

### 4.2 BIM 的正交位置

BIM 不和 pano 收益混成一个总数字：

- raw DA3 → universal scale 测量 BIM 的全局度量尺度收益；
- universal scale → BIM-direct 测量确定性局部 BIM 修正的附加收益；
- pano 收益始终在固定深度分支、固定 support 下比较；
- BIM 收益始终在固定 F* 下比较。

BIM-direct 内含一致性门、BIM 边缘抑制和 Gaussian 局部传播。当前正式结果只比较整个
direct 链与 scale-only，尚不能把链条的效果归因到其中某一组件。

## 5. 正式评测协议

### 5.1 指标、support 与聚合

- 主指标：exact-ERP-solid-angle weighted AbsRel。
- 主聚合：每站先做球面面积加权，再对 station 等权平均。
- 辅助聚合：每房间先等权平均其 station，再对 room 等权平均，并保留逐房间差值。
- 不确定性：10,000 次 room-cluster paired bootstrap，seed 42；抽中房间时保留该房间
  全部 station 和方法配对。
- 深度范围：0.2–5.0 m。
- 尺度：模型直接输出的 metric depth；禁止任何 pano GT median、least-squares 或
  scale-and-shift 对齐。

对 ERP 第 v 行，以像素上下边界纬度 phi_top 与 phi_bottom 计算每像素精确面积权重：

~~~text
w(v) = abs(sin(phi_top) - sin(phi_bottom)) * 2*pi / width
~~~

主 pano 质量比较使用 common_regular：reference 与 candidate 都裁到相同的 GT-valid、
regular-covered support。tangent6 与 tangent14 的 view-count 比较使用共同 tangent
support。任何方法都不能按自己的有效像素另删 support。

### 5.2 质量与覆盖率必须分开

主 AbsRel 的 fixed-support 对比回答“在原有 regular 可评区域，加入 pano 是否更准”。原生
union coverage 回答“能覆盖多少球面”。coverage 从约 67% 增至近 100% 并不意味着主
AbsRel 已经在新增的整球区域与旧方法直接比较；两类数字不能合并成一个提升。

### 5.3 validation 冻结与 test 一次

F* 只由 validation 决定。选择时明确排除 strict single、single_best_view、全部 BIM/scale、
learned_refined、Route-P source-set 变体和所有 test 指标。随后写入
[selection receipt](../data/provenance/stanford_area1_pano_method_selection_v1.json)，再以
--split test --confirm-test 完成唯一一次正式 test，并写入
[execution receipt](../data/provenance/stanford_area1_pano_test_execution_v1.json)。
receipts 为审计保留了旧统一 evaluator 的 learned 元数据；execution receipt 中冻结的
post-execution reporting scope 明确将其降为 archival，不属于本轮报告。

维护者不应在同一协议名下重新调参或覆盖 test。若方法改变，应升协议版本，并使用新的盲测
数据验证。

## 6. Validation-only 方法选择

选择输入为 30 个 paired validation stations、7 个房间上的 Route-R raw DA3，support 固定为
regular-covered 区域。排序规则是最低 spherical station-macro AbsRel；仅在 AbsRel 并列时
用 MAE 决胜，本次没有并列。

| 候选融合 | AbsRel | MAE (m) | 选择 |
|---|---:|---:|---:|
| joint_weighted_log | 0.249790 | 0.556268 |  |
| **joint_huber** | **0.244128** | **0.540558** | **F*** |
| joint_photo_huber | 0.244170 | 0.540716 |  |
| joint_synchronized_huber | 0.283477 | 0.631034 |  |

photo-Huber 与 Huber 几乎相同但没有胜出；当前 synchronization 实现明显变差。它们都是
validation 消融，不得事后根据 test 改选。F* = joint_huber 随后用于 pano 主比较、view-count
对比和确定性 BIM 结果。

需要明确一个选择目标的局限：本版 receipt 按 **regular-only 自身的 validation AbsRel**
选择 F*，而不是按 `regular + tangent14` 的绝对误差或相对 regular 的增益选择。因而它是
冻结且可审计的 confirmatory 规则，但不是“pano 联合最优”的充分证据。作为 validation
敏感性，weighted-log 在 `regular + tangent14` 上为 0.162237，低于 Huber 的 0.208469；
该观察不能在 test 已揭盲后用于改选主方法。下一版协议必须先预注册与 pano 主问题一致的
选择目标，再使用新的未见区域或数据集盲测。

## 7. Validation 结果与冻结依据

### 7.1 regular + pano 主比较

质量均在 common_regular support 上计算，CI 为 candidate − regular-only：

| 输入 | AbsRel | 相对 regular-only | 95% CI | 改善房间 |
|---|---:|---:|---:|---:|
| regular-only | 0.244128 | reference | — | — |
| regular + tangent6 | 0.213329 | −12.62% | [−0.039911, −0.025981] | 7/7 |
| regular + tangent14 | 0.208469 | −14.61% | [−0.053807, −0.026930] | 7/7 |

tangent14 相对 tangent6 为 0.213329 → 0.208469，下降 2.28%，CI
[−0.014216, 0.000357] 跨 0。因此 validation 支持“加 pano”这一主效应，但单凭 validation
不能确认 14 views 必然优于 6 views。

原生 union 的球面覆盖率为：

| 输入 | validation coverage |
|---|---:|
| regular-only | 66.34% |
| regular + tangent6 | 99.75% |
| regular + tangent14 | 99.97% |

### 7.2 tangent-only、strict single 与 BIM

- tangent-only 共同 support：tangent6 0.377106 → tangent14 0.360870，下降 4.31%，CI
  [−0.039398, 0.000228]，跨 0。
- raw strict single → F* joint：0.115857 → 0.213634，变差 84.40%，CI
  [0.077980, 0.112101]，0/7 房间改善。
- universal-scale strict single → F* joint：0.084282 → 0.069769，下降 17.22%，CI
  [−0.032113, −0.003994]。
- regular-only raw → universal scale：0.244128 → 0.074442，下降 69.51%，CI
  [−0.185151, −0.161858]，7/7 房间改善。
- universal scale → BIM-direct：0.074442 → 0.076742，变差 3.09%，CI
  [−0.000384, 0.003685]，跨 0。

为直接回答“pano 联合比一张 regular 图更好吗”，另做了一个在正式 test 揭盲后新增、因此
**仅限 validation 的探索性对照**。先用同一个无 GT whole-frame selector 固定一张 regular，
再分别加入 nested tangent6/14；三种方法的质量统一裁到所选单帧的完全相同 support，原生
球面覆盖率单独报告：

| 输入 | common-support AbsRel | 相对单帧 | room-cluster 95% CI | 原生球面覆盖率 |
|---|---:|---:|---:|---:|
| strict single | 0.115857 | reference | — | 10.87% |
| strict single + tangent6 | 0.164543 | 变差 42.02% | [0.024898, 0.059466] | 99.38% |
| strict single + tangent14 | 0.318592 | 变差 174.99% | [0.098559, 0.264218] | 99.85% |

6 views 只有 1/7 房间改善，14 views 为 0/7；14 相对 6 也显著变差。可信结论不是
“pano 比单图更准”，而是 **pano tangent 极大增加覆盖，但朴素 raw Huber 融合破坏了单图
已覆盖区域的精度**。来源数失衡及 DA3 在 regular/tangent 间的尺度/上下文漂移是合理解释，
但当前只是机制假设。完整结果见
[val-only strict-single+pano summary](../results/stanford_area1/pano_val_single_plus_tangent/summary.json)。

两个 pano-only validation stations 来自同一房间。tangent14 + F* 在共同 tangent support
上的描述性 spherical station-macro AbsRel 为 0.274865；样本太少，不报告 CI，不作
泛化 claim。Route-R 主表覆盖 30/32 个 validation pano stations；Route-P tangent 分支覆盖
32/32。test 为 31/31 paired，没有 pano-only station。

## 8. 一次性 Test 结果

### 8.1 主要 pano 收益

| 输入 | AbsRel | 相对 regular-only | 95% CI | 稳定性 |
|---|---:|---:|---:|---|
| regular-only | 0.269850 | reference | — | 31 stations / 7 rooms |
| regular + tangent6 | 0.235008 | −12.91% | [−0.049554, −0.025079] | 30/31 站、7/7 房间改善 |
| **regular + tangent14** | **0.228651** | **−15.27%** | **[−0.052933, −0.030209]** | **30/31 站、7/7 房间改善** |

在同一 F* 下，tangent14 相对 tangent6 为 0.235008 → 0.228651，下降 2.70%，CI
[−0.011805, −0.000859]，27/31 站、6/7 房间改善。房间均值全部改善不等于每个 station
都改善，故主表同时保留两层计数。

| 输入 | test 原生 union coverage |
|---|---:|
| regular-only | 66.92% |
| regular + tangent6 | 99.75% |
| regular + tangent14 | 99.96% |

### 8.2 嵌套 view-count 消融

仅比较 tangent 来源，并裁到 tangent6 与 tangent14 的共同 support：

| 方法 | AbsRel | 相对 tangent6 | 95% CI | 改善 station | 改善房间 |
|---|---:|---:|---:|---:|---:|
| tangent6 + F* | 0.394129 | reference | — | — | — |
| tangent14 + F* | 0.363998 | −7.64% | [−0.040528, −0.016809] | 27/31 | 7/7 |

因为 tangent14 严格包含 tangent6，这一结果支持更多重叠方向在固定 Huber 融合下有用；
它不证明任意增加 views 都会单调改善。

### 8.3 strict single 与全部 regular joint 的负结果

这里比较一张完整 regular frame 与 Route-R **全部 regular 来源**的 F* joint，并把二者裁到同一
selected-frame strict support：

| 深度分支 | 真正单帧 | F* joint，同一 strict support | 相对变化 | 95% CI |
|---|---:|---:|---:|---:|
| raw DA3 | 0.170524 | 0.269759 | 变差 58.19% | [0.088392, 0.108800] |
| universal scale | 0.126738 | 0.113430 | 改善 10.50% | [−0.030672, 0.001299] |

raw joint 的负结果说明独立 perspective 预测的尺度/上下文差异会污染窄视野已较容易的
support。尺度校正后点估计反转，但 test CI 跨 0，不能宣称“joint 一定优于一张图”。这与
主结论不矛盾：主结论的 reference 是 regular-only 多来源融合，问题是增加 pano tangent 后
是否改善同一 regular support。

### 8.4 确定性 BIM 消融

以下三项都使用同一 regular-only support 与 F*，只改变确定性 BIM 深度分支：

| 方法 | AbsRel | 相对前一步 | 95% CI |
|---|---:|---:|---:|
| raw DA3 | 0.269850 | reference | — |
| universal scale | 0.108243 | −59.89% | [−0.195433, −0.106697] |
| BIM-direct | 0.110064 | 相对 scale-only 变差 1.68% | [−0.001180, 0.004009] |

因此确定性 BIM 的可复现结论是 **全局尺度矫正有效**。local/direct 的点估计更差且 CI
跨 0，不能声称一致性门、边缘抑制或 Gaussian 传播带来额外收益。实际部署的默认非学习
BIM 基线应保留 universal scale，并把 BIM-direct 作为负消融。

## 9. 三组选材与素材路径

所有样例仅从 validation 的 30 个 paired stations 中按数值规则选择；选样阶段不打开 RGB
或 GT 图像，且明确不读取 test。运行导出器后，总清单与预览分别位于
`docs/assets/pano_evaluation/qualitative/manifest.json` 和 `preview_sheet.png`。这些含 Stanford
RGB/GT 派生内容的素材只保存在获许可的本机工作区，不随公开仓库再分发；公开说明见
[全景素材手册](assets/pano_evaluation/README.md)。

| 备选 | 固定选择规则 | station / room | validation gain |
|---|---|---|---:|
| A：典型增益 | regular + tangent14 相对 regular-only 的改善最接近总体中位数；并列按 station ID | 2439b3… / office_6 | 0.033610；总体中位数 0.031107 |
| B：最大增益 | 同一固定指标下改善最大 | 2fb7a2… / hallway_2 | 0.074468 |
| C：最小增益/困难例 | 同一固定指标下改善最小，诚实展示边际收益 | a77fba… / office_31 | 0.006180 |

每个目录都保存无标题的 pano RGB、GT range、regular-only、regular+tangent14、误差图、
support、coverage、universal scale 和 BIM-direct，可供后续 PPT 自由拼接。深度、AbsRel
和 signed-AbsRel 的统一色标位于本机 `qualitative/colorbars/`。

素材 manifest 的导出范围只含 raw、universal scale 与 BIM-direct；**本轮主图不得引入
learned 面板或 raw evaluator 中的 learned 数值**。最终导出没有加载 checkpoint，也没有
执行 learned forward。推荐三套图稿：

1. A 组：RGB、GT、regular-only、regular+tangent14、两张误差图，作为主文典型例；
2. B 组：再加入 coverage 与 signed difference，突出覆盖和大增益；
3. C 组：同样版式展示最小正增益，作为补充材料或局限性例。

## 10. 可复现命令

### 10.1 下载与验证 pano

使用前需自行接受 Stanford 与 BIMSyn 的数据条款：

~~~bash
.venv/bin/python scripts/data/download_stanford_area1.py \
  --stanford-root ../Stanford2D3DS \
  --bimsyn-root ../BIMSyn \
  --accept-stanford-license \
  --accept-bimsyn-terms \
  --include-pano

.venv/bin/python scripts/data/verify_stanford_bimsyn_sources.py \
  --area-root ../Stanford2D3DS/no_xyz \
  --area-tar ../Stanford2D3DS/no_xyz/area_1_no_xyz.tar \
  --ifc-root ../BIMSyn/BIM_model/ifc \
  --require-pano \
  --output data/provenance/stanford_area1_pano_sources.json
~~~

### 10.2 生成冻结 tangent cache

~~~bash
.venv/bin/python scripts/data/cache_stanford_pano_da3.py \
  --config configs/stanford_area1.yaml \
  --split val \
  --preset nested14 \
  --face-resolution 504 \
  --log-every 1

# 独立复现 test cache；项目维护者不要用它覆盖正式产物
.venv/bin/python scripts/data/cache_stanford_pano_da3.py \
  --config configs/stanford_area1.yaml \
  --split test \
  --confirm-test \
  --preset nested14 \
  --face-resolution 504 \
  --log-every 1
~~~

### 10.3 正式 validation

以下推荐命令完全不加载项目 checkpoint，复现本轮 training-free 报告范围；它不试图重建
历史 raw artifact 中已排除的 learned 行，也不保证与该归档文件逐字节相同。输出写入
`outputs/`，避免覆盖仓库中冻结的正式审计产物。

~~~bash
.venv/bin/python scripts/model/evaluate_stanford_pano.py \
  --config configs/stanford_area1.yaml \
  --tangent-manifest data/processed/stanford_area1_504/pano_da3/nested14_r504_737a5fa1b07a/manifests/val_full.json \
  --split val \
  --output outputs/stanford_area1/pano_val_training_free \
  --device cuda \
  --batch-size 8 \
  --pano-height 512 \
  --bootstrap-repetitions 10000 \
  --seed 42
~~~

直接“同一单图 + pano tangent”的 validation-only 诊断使用独立脚本；它没有 test、checkpoint、
BIM 或可调 fusion 参数入口：

~~~bash
.venv/bin/python scripts/analysis/evaluate_stanford_pano_single_plus_tangent.py \
  --config configs/stanford_area1.yaml \
  --tangent-manifest data/processed/stanford_area1_504/pano_da3/nested14_r504_737a5fa1b07a/manifests/val_full.json \
  --output outputs/stanford_area1/pano_val_single_plus_tangent
~~~

面向原始 regular benchmark 的推荐 round-trip validation 命令为：

~~~bash
.venv/bin/python scripts/analysis/evaluate_stanford_pano_regular_roundtrip.py \
  --config configs/stanford_area1.yaml \
  --tangent-manifest data/processed/stanford_area1_504/pano_da3/nested14_r504_737a5fa1b07a/manifests/val_full.json \
  --output outputs/stanford_area1/pano_val_regular_roundtrip
~~~

该脚本强制 validation-only，并输出逐 regular frame/station/room 的固定支持域结果。

若要在**同一 universal BIM-scale 基线**上隔离 ERP 联合的增益，不需要 tangent cache：

~~~bash
.venv/bin/python scripts/analysis/evaluate_stanford_bim_scale_roundtrip.py \
  --config configs/stanford_area1.yaml \
  --output outputs/stanford_area1/pano_val_bim_scale_regular_roundtrip
~~~

该入口同样强制 validation-only；两侧都使用相同逐帧 BIM scale，且没有 pano RGB/GT、
tangent、checkpoint、learned 模型或 GT-based scale/fusion。

### 10.4 一次性 test 命令

以下是 training-free 的独立复现命令，不加载 checkpoint。历史单次执行及其完整输入身份以
execution receipt 为准；项目维护者不得借此在当前协议下重新调参或覆盖已有 test：

~~~bash
.venv/bin/python scripts/model/evaluate_stanford_pano.py \
  --config configs/stanford_area1.yaml \
  --tangent-manifest data/processed/stanford_area1_504/pano_da3/nested14_r504_737a5fa1b07a/manifests/test_full.json \
  --split test \
  --confirm-test \
  --output outputs/stanford_area1/pano_test_training_free \
  --device cuda \
  --batch-size 8 \
  --pano-height 512 \
  --bootstrap-repetitions 10000 \
  --seed 42
~~~

### 10.5 Schema 3 产物

validation 与 test 目录均包含：

- summary.json：聚合结果、固定对比、CI 与 schema version 3；
- provenance.json：输入、环境、参数和哈希；
- per_station.csv 与 per_room.csv：逐站/逐房间审计；
- strict_single_per_station.csv：真正单帧对比；
- tangent_per_station.csv：tangent6/14 对比；
- regular_pano_joint_per_station.csv：regular-only 与加 pano 对比。

入口：

- [validation summary](../results/stanford_area1/pano_val/summary.json)
- [validation provenance](../results/stanford_area1/pano_val/provenance.json)
- [test summary](../results/stanford_area1/pano_test/summary.json)
- [test provenance](../results/stanford_area1/pano_test/provenance.json)
- [single + pano validation-only summary](../results/stanford_area1/pano_val_single_plus_tangent/summary.json)
- [single + pano validation-only provenance](../results/stanford_area1/pano_val_single_plus_tangent/provenance.json)
- [regular round-trip validation-only summary](../results/stanford_area1/pano_val_regular_roundtrip/summary.json)
- [regular round-trip validation-only provenance](../results/stanford_area1/pano_val_regular_roundtrip/provenance.json)
- [BIM-scale regular round-trip validation-only summary](../results/stanford_area1/pano_val_bim_scale_regular_roundtrip/summary.json)
- [BIM-scale regular round-trip validation-only provenance](../results/stanford_area1/pano_val_bim_scale_regular_roundtrip/provenance.json)
- [method-selection receipt](../data/provenance/stanford_area1_pano_method_selection_v1.json)
- [test-execution receipt](../data/provenance/stanford_area1_pano_test_execution_v1.json)

关键 SHA-256：

| 产物 | SHA-256 |
|---|---|
| validation summary | 68e101b59a2bb1340f16b5ddceecfdead91ddf1395b47a79bc21994c496cd361 |
| test summary | bee9cee821e435b21a02284743d68b424f51673be76eebd8a7b38d3141a34746 |
| single + pano validation-only summary | 6f2cd6ddf2b448162ff29950549a96e11628ccc5cd4d6efb525e6dc8910ed833 |
| single + pano validation-only provenance | 95c18fd5ef82582923d9ff4680c6de32bf6cb43ce77e352999671b0cd7c4b615 |
| regular round-trip validation-only summary | 2036aa10b5c1cc009c4fe2595a0373187ea1119f072cc672c43de6def37273c0 |
| regular round-trip validation-only provenance | 92b45c77f60a28e4c29802ff4f53a4855da0b772ff2235a123d37b00fbf0377b |
| BIM-scale regular round-trip validation-only summary | 7676509318d8781c8e5f51cbce6c5d3bcfd6f57330647682921471099a4251f2 |
| BIM-scale regular round-trip validation-only provenance | 9093661f1f5b634c026bc104a2e1dee4d86818bb45f12bbe44cd9a71cd766591 |
| selection receipt | 649732c294174530e1f121324720da6d76d8a44d46cea0102f2484a9a52d2c56 |
| execution receipt | a2dde1bf08ca0a24cb3cebf7dd10e1b85c91f2535a775ee90c85aef239be1ecc |

## 11. 消融、敏感性与下一步

### 11.1 已完成

| 因素 | 正式状态 | 当前证据 |
|---|---|---|
| fusion | validation 比较 weighted log、Huber、photo-Huber、sync-Huber | regular-only 目标选中 Huber；pano 目标下 weighted 更好，仅作探索性敏感性 |
| pano view count | 嵌套 6 vs 14，FOV/分辨率固定 | test tangent-only 改善 7.64% |
| coverage vs quality | common support 质量 + native union coverage | 质量提升与近整球覆盖均成立，但分开解释 |
| true single vs joint | GT-free whole-frame selector、同 strict support | raw 为负；scale 后点估计为正但 test CI 跨 0 |
| true single + pano tangent | 同一 selected frame，+6/+14，val-only | 覆盖约 11%→99%，但共同 support 精度显著变差 |
| regular→ERP→regular round-trip | 1,673 张原始 regular、三种融合、0/6/14 tangent | weighted-log + tangent14 相对同融合 regular-only 改善 30.09%，7/7 房间 |
| BIM-scale regular→ERP→regular | 两侧同一逐帧 universal scale，无 pano/tangent/learned | Huber AbsRel 0.08644→0.07595（−12.14%），7/7 房间，95% CI 全低于 0 |
| BIM scale vs local | raw → universal scale → BIM-direct | scale 强；direct 没有附加收益 |

### 11.2 尚未完成，不能拆分归因

后续只在新 validation 协议上做，不再用现有 test 调参：

- regular view count：1、2、4、8、all，并同时报告 common-support error 与 native coverage；
- tangent sensitivity：FOV、face resolution、6/14 之外的方向数与运行时间；
- blind confirmation：冻结 round-trip validation 选出的 weighted-log + tangent14 与
  `candidate − same-fusion regular-only` contrast，在新的未见区域/holdout 上一次性验证；
- photo/consistency：光度权重、预测一致性阈值及拒绝像素比例的独立 on/off；
- BIM 因子链：scale-only → + consistency gate → + edge suppression → + Gaussian
  propagation，逐项固定 support；
- BIM reliability：家具冲突、BIM 边缘和 pose 扰动下的 calibration，而不是只报总体平均；
- 配准敏感性：小角度旋转、平移和 BIM 缺失围护面对结果的影响。

在这些因子化实验完成前，不得声称 BIM edge、consistency gate 或 Gaussian propagation
单独有效，也不得把 BIM-direct 的负结果归因给某个单一组件。

### 11.3 可写的新颖点

以下是本项目可审计的方法特点，不宣称文献首创：

1. 同中心 regular 与 pano tangent 的 training-free 球面联合，并明确排除视差叙事；
2. exact solid-angle、equal-station 主指标与 room-cluster 配对 CI；
3. common-support 精度和 native-coverage 的双轴报告；
4. pano 主效应与确定性 BIM 尺度/局部效应的正交拆分；
5. 嵌套 6/14 views、真正单帧与 single+pano 负对照、local BIM 负消融，避免只保留有利结果。

## 12. 局限

- regular 图由对应 ERP 采样，regular 与 pano 高度相关；结果不是独立相机或新增几何基线。
- 独立 tangent 预测存在尺度和上下文漂移，raw strict 结果说明稳健融合不能自动解决所有
  冲突。
- “单图 + pano”直接对照目前只有 validation，且结果为负；不能用多 regular baseline 的
  正收益替代这一结论，也不能在已揭盲 test 上补做确认。
- 主质量结论位于 common_regular support；新增约三分之一球面的质量不能由旧 reference
  直接估计，只能结合 tangent-only 与 coverage 结果理解。
- validation/test 各只有 7 个房间，cluster CI 仍可能较宽；不能把它当成跨数据集结论。
- pano-only validation 只有同一房间的 2 站，test 没有 pano-only，部署覆盖结论有限。
- 0.2–5.0 m 是项目任务范围，不能外推到远距离室外场景。
- BIM 依赖已注册围护结构；自动配准误差、家具冲突和 BIM 缺失仍需敏感性测试。
- 当前做法是固定 DA3 的独立 perspective predictions + spherical fusion，不是 DA3 原生
  pose-conditioned multi-view。
- test 已经正式访问一次；后续结构优化必须使用新协议和新的盲测数据。
- learned refiner 不属于本轮 pano claim，即使 raw artifact 中存在相关字段。

## 13. 后续研究方向：未实现的外部方法

以下方法只作为未来 baseline/替代模块，**本项目尚未集成或评测，本文不能据此声明比较优势**：

- [Depth Any Camera（DAC，CVPR 2025 官方实现）](https://github.com/yuliangguo/depth_any_camera)
  直接针对任意相机模型，并将深度定义为相机中心的欧氏距离；可用于减少 ERP/tangent 的
  相机模型失配。论文见
  [CVF 官方 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Depth_Any_Camera_Zero-Shot_Metric_Depth_Estimation_from_Any_Camera_CVPR_2025_paper.pdf)。
- [Depth Any Panoramas（DAP，官方项目页）](https://insta360-research-team.github.io/DAP_website/)
  是专门的全景 metric depth foundation model，可作为 direct-pano baseline，检验 tangent
  切分是否仍有必要。论文见
  [CVF 官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_Depth_Any_Panoramas_A_Foundation_Model_for_Panoramic_Depth_Estimation_CVPR_2026_paper.pdf)。
- [PaGeR（官方实现）](https://github.com/prs-eth/PaGeR)采用 cubemap 多方向 foundation-model
  预测和每个 panorama 的尺度头，可启发无 GT 的 per-pano scale calibration。预印本见
  [作者发布的 arXiv 页面](https://arxiv.org/abs/2605.26368)。

与本项目直接相关的已发表全景切分/融合背景包括
[360MonoDepth（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Rey-Area_360MonoDepth_High-Resolution_360deg_Monocular_Depth_Estimation_CVPR_2022_paper.html)、
[OmniFusion（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Li_OmniFusion_360_Monocular_Depth_Estimation_via_Geometry-Aware_Fusion_CVPR_2022_paper.html)
与
[Pano3D（CVPR Workshops 2021）](https://openaccess.thecvf.com/content/CVPR2021W/OmniCV/html/Albanis_Pano3D_A_Holistic_Benchmark_and_a_Solid_Baseline_for_360deg_CVPRW_2021_paper.html)。
DA3 模型定义与系列差异以
[Depth Anything 3 官方仓库](https://github.com/ByteDance-Seed/depth-anything-3)为准。

## 14. 评测演化与试错记录

本节记录方法如何从“直接估计 pano depth”收敛到“原始 regular 预测的
ERP 联合校正”。这是试错日志，不应把 validation 与已解盲 test 的结果混成一个
新的 confirmatory claim。

### 14.1 先区分不同问题和支持域

- Area 1 正式 regular test 的 `raw DA3 = 0.30123` 和 `universal scale = 0.07752`
  是 1,641 张原始 regular 图像的 blind-test pixel-micro AbsRel。
- round-trip 实验的 `0.08644 → 0.07595` 是 1,673 张 validation regular 图像、
  同一 fixed support 上的对照。它回答“将同站 regular 预测映射到 ERP 联合后，
  再回投原 regular 视野是否有益”。
- 两组数字的 split、帧集和尺度协议不同，不能直接用数值大小判断 pano
  联合是否有效。可归因的对比必须是同一行协议内的 candidate 与 reference。

### 14.2 tangent 路线为什么是必要但有代价的弯路

最初实现从 ERP pano 额外切出 cubemap6/nested14 tangent 图像，再分别运行 DA3。
这条路线验证了 ERP 坐标、视线和回投几何，也能补充原 regular 没有覆盖的方向；
但它不是用户最终关心的基线，因为数据集本已提供 regular 图像。

严格 single-regular + tangent 的 validation 负对照证明了这个风险：在同一张
regular 的 common support 上，AbsRel 从 0.11586 变为 tangent6 的 0.16454 和
tangent14 的 0.31859；同时 native spherical coverage 从 10.87% 上升到 99.38%
和 99.85%。因此额外 tangent 在“覆盖率”上有价值，但不能当成“同一
regular 区域更准”的证据。主要风险是独立 DA3 预测的尺度/投影域偏移以及
1 个 regular anchor 对 6/14 个 tangent source 的数量失衡。

### 14.3 回到原始 regular→ERP→regular

随后协议改为：对数据集提供的全部 regular 图像各自运行 DA3，使用官方
pose 反投到同一 ERP，在 ERP 上作预测间的稳健联合，再回投到每张原始
regular 的像素上评测。不重新切图，也不重新运行 DA3。

在 BIM-scale validation 固定支持域上：

| 方法 | AbsRel | MAE (m) | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|
| 每帧 regular universal scale | 0.08644 | 0.17422 | 0.39536 | 0.91533 |
| 单张投影→回投控制 | 0.08608 | 0.17336 | 0.38683 | 0.91543 |
| 同站 ERP joint Huber→回投 | **0.07595** | **0.15254** | **0.35387** | **0.93156** |

Huber 相对原每帧 regular scale 的 AbsRel 降低 12.14%，7/7 个 validation
房间均改善，room-cluster 95% CI 为 [-0.01231, -0.00700]。“仅投影→回投”
的差异很小，说明主要收益来自多 regular 的重叠一致性，而不是采样本身。

### 14.4 Huber 与残差同步的试错

joint Huber 在 log radial-depth 上融合。基础权重只由视角中心性和 DA3
confidence 组成；残差只是各预测与预测共识之间的差，不是与 GT 的差。
实现另外尝试了 overlap log-offset synchronization：在 regular 重叠区域求每对预测的
稳健 log-scale 差，再解每张图的标量 offset。

它呈现指标权衡：pixel-micro AbsRel 从 joint Huber 的 0.07595 降到
0.07446，RMSE 从 0.35387 降到 0.33514，但 MAE 从 0.15254 升到
0.15887，room-macro AbsRel 也从 0.07608 升到 0.07679。因此它没有被解释为
全面优于普通 Huber，而是保留为残差校准敏感性结果。

GT leakage 审计结论是：尺度同步、Huber 权重、一致性门和 ERP 回投全部在
GT 读取之前冻结；GT 只用于最后固定支持域的指标。但本组是 validation-only
的方法开发结果，不是新的盲测证据。

### 14.5 BIM 的两个负结果与边界

- 在当前 Area 1 test 上，单帧 robust global scale AbsRel 为 0.07752，附加局部
  BIM-direct 后为 0.07815；局部 direct 并未改善总体 AbsRel。
- 在 BIM-scale round-trip validation 中，universal-scale joint Huber 为 0.07595，
  BIM-direct joint Huber 为 0.07807；先做 local direct 仍然没有附加收益。
- Area 1 的 BIM 位姿由官方 semantic structural mesh 辅助求得每房间固定
  `T_area_from_bim`。这不是逐帧 depth-GT leakage，但属于
  scan-calibrated/oracle-style registration；无标定部署不能直接引用这些 BIM 数字。

## 15. 回到 regular：结构/家具语义增强方案

### 15.1 现有策略已经考虑了什么

现有策略只是**间接**考虑家具，并没有在推理时显式读取语义类别：

1. robust scale 假设围护 BIM 位于家具后方，因此 `BIM / DA3` 会出现单侧大比值
   长尾；用 q25/q10 log cap 限制 q45，但它不知道哪个像素是家具。
2. local direct 只在 `|log(BIM / scaled_DA3)| <= 0.10` 且远离 BIM 边缘的像素上
   采样残差。大部分家具-后墙冲突会被一致性门拒绝。
3. 被接受的残差经归一化 Gaussian 传播后，乘到 scaled DA3 上。它没有用
   BIM depth 替换家具，所以家具的相对几何主要仍来自 DA3。
4. supervised refiner 将 furniture 和 BIM-foreground-conflict 像素的 loss 加权，但
   `semantic_class`/`furniture_mask` 不是模型推理输入；BIM reliability 头也是辅助监督，
   不是直接乘到输出上的门。

这些设计能降低家具污染，但不足以完全防止 Gaussian 修正跨过物体边界。
现有 Area 1 blind-test pixel-micro 数据直接显示了这个问题：

| 子集 | robust scale AbsRel | + local direct AbsRel | 相对变化 |
|---|---:|---:|---:|
| all | 0.07752 | 0.07815 | +0.81% |
| furniture | 0.08710 | 0.08929 | +2.51% |
| BIM foreground conflict | 0.13891 | 0.14064 | +1.24% |
| BIM–GT geometric-consistent | 0.04310 | 0.04092 | **-5.06%** |

即：BIM 局部修正在 BIM 与 GT 深度几何一致区有效，但家具/前景冲突抵消了
总体收益。注意这个历史 `bim_consistent` mask 未叠加语义结构标签；它是评测用的
几何一致子集，不是 wall/floor 语义子集，也不是可用于推理的门。

### 15.2 建议的两阶段语义策略

不应直接把官方 Area 1 semantic GT 当作 test-time 输入；那只能作为 oracle
upper bound。可部署的方案应输入 RGB 并预测一个类别无关的连续结构概率
`q_struct ∈ [0,1]`，只区分：

- BIM-supported structure：wall/floor/ceiling/column/beam；
- movable/foreground：table/chair/sofa/bookcase/clutter 等；
- unknown/unsupported：door/window/board 及低置信区域。

然后将全局尺度与局部几何分开：

1. **semantic structural scale**：只在高 `q_struct`、BIM-valid、非边缘像素上估计
   robust scale，但把这个单一尺度乘到整张 DA3，因此家具也获得公制尺度而不被
   BIM 表面覆盖。
2. **semantic local gate**：局部 BIM 残差仅在结构像素上强应用；家具像素
   回退到 scaled DA3。连续形式可写成：

   \[
   \log \hat d = \log(s d_{DA3}) + \alpha\,q_{struct}\,c_{BIM},
   \]

   其中 `c_BIM` 只由高可信结构支持生成。
3. **semantic-guided propagation**：将普通 Gaussian 替换为 joint-bilateral/guided
   propagation，权重同时依赖空间距离、RGB 边界和 `q_struct` 差异，防止墙面
   残差穿过桌椅边界。
4. **asymmetric conflict rejection**：当 BIM 显著比 scaled DA3 更远时，优先解释为
   前景遮挡，快速关闭局部 BIM；对小偏差仍保留连续权重，避免 hard gate 的闪烁。

### 15.3 先做 oracle，再做可部署语义

建议两步验证：

1. **validation-only oracle semantic ablation**：允许使用官方 semantic mask，只用来
   判断这个方向的性能上限，不宣称可部署。
2. **predicted semantic protocol**：在 train rooms 上训练二值 structuralness head，或使用固定
   公开语义模型产生 mask；val 选阈值后冻结，test 只读 RGB/DA3/BIM/pose。

这两条协议必须分表报告，不得把 oracle semantic GT 的收益当成实际推理结果。

### 15.4 预注册消融与验收条件

在同一 regular split、同一 fixed support 上比较：

| ID | 方法 | 回答的问题 |
|---|---|---|
| S0 | robust scale，all-pixel | 当前最强简单基线 |
| S1 | + 现有 geometry local direct | 原 consistency/edge/Gaussian 整体效果 |
| S2 | structural-only scale | 语义能否净化尺度样本 |
| S3 | S2 + semantic application gate | 是否避免家具过校正 |
| S4 | S3 + semantic-guided propagation | 是否减少跨边界污染 |
| S5 | oracle semantic S4 | 方法上限，仅 validation/补充材料 |
| S6 | predicted semantic S4 | 最终可部署方法 |

主指标为 all pixel/frame/room AbsRel 和 MAE；必须另报 structural、furniture、
BIM-foreground-conflict 和 BIM-boundary。接受 S6 的条件是：

- all 同时优于 S0 和 S1；
- structural 显著优于 S0；
- furniture/conflict 至少不劣于 S0，而不是只胜已知在这些区域退化的 S1；
- 用 room-cluster paired bootstrap 报告差值与 95% CI；阈值、语义模型版本和
  class mapping 全部在 test 之前冻结。

### 15.5 已完成的 official-semantic oracle validation

第一版试验把语义主要用于全局尺度之后的像素级 BIM 替换。这回答的是“哪里允许
BIM 覆盖”，却没有正确回答“哪些对应关系应决定全局尺度”。该试验现保留为
`oracle_semantic_val/` 的历史试错证据，不再作为推荐语义方案。

修订后的主协议中，语义**只决定尺度样本**：仅保留
`image wall ↔ BIM wall`、`image floor ↔ BIM floor` 等同类结构对应，估计每帧唯一
标量 (s)，再将 (s d_{DA3}) 乘到整张图。家具与结构没有不同的输出分支，也没有
局部 BIM replacement、consistency gate 或 Gaussian 传播。

旧的 `log_upper_cap_v1` 是为 all-hit 比例中的家具正长尾设计的；直接把它套在净化后的
同类结构比例上，会把尺度估低：core-only 为 0.09706，category-match 仍为 0.09352，
均差于 all-hit 0.08644。因此对同类结构比例单独扫描 log-ratio quantile。分位数只在
30 个 train rooms、7,013 帧上选择；q65 同时取得 train pixel/frame/room-macro 最低
AbsRel，随后冻结到 validation。validation 的 q75 事后最优值只作为 sensitivity 上限，
不作为合法候选。

| validation 全图单尺度方法 | AbsRel | MAE (m) | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|
| all-hit universal scale | 0.08644 | 0.17422 | 0.39536 | 0.91533 |
| semantic core + old capped estimator | 0.09706 | 0.19747 | 0.41019 | 0.90060 |
| exact category match + old capped estimator | 0.09352 | 0.19187 | 0.40317 | 0.91000 |
| category-balanced scale | 0.08712 | 0.17668 | 0.37849 | 0.91456 |
| **train-selected exact-match q65** | **0.07725** | **0.15464** | **0.36982** | **0.93363** |
| val-selected q75（仅上限） | 0.07494 | 0.14772 | 0.37047 | 0.93368 |

q65 相对 all-hit scale 的 validation AbsRel 降低 10.63%，MAE 降低 11.24%，
room-macro AbsRel 从 0.08448 降到 0.07636；7/7 room 改善，room-cluster bootstrap
差值 95% CI 为 `[-0.01404, -0.00273]`。由于尺度应用于全图，家具 AbsRel 也从
0.10979 降到 0.10097，而不是逐位不变；core structure 从 0.06103 降到 0.05121。
foreground-conflict 点估计从 0.12409 降到 0.11236，但房间 CI
`[-0.01664, +0.00022]` 仍略跨 0，不能宣称该子集稳定改善。

结论是：**用户提出的“语义用于全局尺度取样”明显优于第一版像素级语义修正思路。**
但这仍是使用官方 semantic annotation 的 privileged oracle。可部署实现需要固定语义模型
预测五类结构及 other；必须先报告其 category-match precision/recall，再复用 q65，不能
在 test 上重新选择分位数。

复现命令：

    .venv/bin/python scripts/analysis/ablate_oracle_semantic_bim.py \
      --config configs/stanford_area1.yaml --split train --study semantic-scale \
      --selection-only --workers 8 \
      --bootstrap-repetitions 10000 --bootstrap-seed 42 \
      --output-dir results/stanford_area1/oracle_semantic_global_scale_train

    .venv/bin/python scripts/analysis/ablate_oracle_semantic_bim.py \
      --config configs/stanford_area1.yaml --split val --study semantic-scale \
      --workers 8 --bootstrap-repetitions 10000 --bootstrap-seed 42 \
      --output-dir results/stanford_area1/oracle_semantic_global_scale_val

冻结产物：

- `results/stanford_area1/oracle_semantic_global_scale_train/summary.json`
- `results/stanford_area1/oracle_semantic_global_scale_val/summary.json`
- 两个目录中的 `provenance.json` 与 `per_frame.csv`
