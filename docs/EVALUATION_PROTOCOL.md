# BIM-PriorDA3 科研评测协议与图件规划

## 1. 文档目的与证据等级

本文把当前项目转换为可写入论文、也可由公开代码复核的评测方案。所有数字分为两类：

- **已有证据**：已经由冻结 checkpoint、annotation、固定 support 和正式 summary 产生；
  可以据此写结果，但仍受文中披露的单次训练、数据划分和标定假设限制。
- **建议补跑**：用于完善消融、敏感性、效率或失效分析的实验设计；在结果真正落盘前，
  只能写成实验计划，不能把预期趋势写成结论。

本协议的主任务是：给定单帧 RGB、DA3 度量深度和固定 BIM 围护结构先验，预测包含家具
和临时物体的完整场景深度。正式深度范围固定为 **0.2--5.0 m**。方法思想与
[Prior Depth Anything](https://arxiv.org/abs/2505.10565) 的“先把不完整度量先验与单目
预测对齐，再条件化细化”相近，但本项目的先验来自 BIM ray casting，网络、监督和
评测子集均为本项目定义。基础预测器来自
[Depth Anything 3](https://arxiv.org/abs/2511.10647)。Area_1 和 BIM 文件分别关联
[2D-3D-S 原始论文](https://arxiv.org/abs/1702.01105)与
[BIMSyn 原始论文](https://doi.org/10.1016/j.autcon.2023.105076)。

当前可以主张的核心结果是：统一尺度协议下，学习式 refiner 在 SLABIM 正式 test 和
Area_1 room-disjoint test 的总体固定 support 上均优于直接 BIM 矫正；Area_1 家具子集
也稳定改善。冲突子集的点估计改善，但 room-bootstrap AbsRel 区间跨 0，不能主张房间
层面的显著优越。当前正式结果不包含 E2E 模型。

## 2. 研究问题与预注册式假设

### 2.1 主假设

| 编号 | 假设 | 主要比较 | 预先规定的通过条件 |
|---|---|---|---|
| H1 | BIM 可修正 DA3 的度量尺度 | universal global scale vs. raw DA3 | all pixel-micro AbsRel、MAE 均降低 |
| H2 | 学习细化优于确定性 BIM 矫正 | learned refiner vs. universal BIM-direct | all 的 AbsRel、MAE 在 pixel/frame/group 三层聚合均降低 |
| H3 | 学习细化能恢复 BIM 中不存在的家具 | 同 H2，在 furniture 子集 | AbsRel、MAE 在三层聚合均降低 |
| H4 | 改善不是少数大房间或高像素帧造成 | 配对 group 差值与 bootstrap | room/region mean difference < 0，95% CI 不跨 0 |

Area_1 的 group 是房间；SLABIM 的 group 是区域。H2/H3 的“通过”采用合取条件，不能只
挑选一个聚合或一个指标。AbsRel 是排序主指标，MAE 用于约束实际米制误差；RMSE 和
\(\delta\) 为辅助指标。

### 2.2 次要与探索性假设

- **H5，跨域可迁移性**：SLABIM 上训练的 refiner 能否不更新权重直接用于 Area_1。
  这是诊断性假设；当前已有证据明确不支持它。
- **H6，冲突区恢复**：refiner 是否优于 universal BIM-direct 处理“真实前景比 BIM 围护
  更靠近相机”的像素。盲测点估计支持，但 room-bootstrap AbsRel 区间跨 0，故仍只能
  报告为未解决问题。
- **H7，E2E 微调收益**：解冻 DA3 decoder 尾部是否在相同 validation protocol 下优于
  frozen refiner。它必须同时胜过自己的 live robust comparator 和 frozen 主模型，且
  资源增量可接受，才能晋级。Area_1 challenger 未满足第二项。

## 3. 数据协议与不可变比较条件

### 3.1 数据划分

| 数据集 | train / validation / test | 隔离原则 | 当前用途 |
|---|---:|---|---|
| SLABIM clean global | 496 / 104 / 108 帧 | `ignore.txt` 90 帧先排除，另以 13 帧防止融合 LiDAR 来源跨 split；六区域均进入各 split | 同域训练与 test |
| 2D-3D-S Area_1 + BIMSyn | 7,013 / 1,673 / 1,641 帧；30 / 7 / 7 房间 | 房间和 `camera_uuid` 均不跨 split | room-disjoint 目标域训练、验证和一次盲测 |

SLABIM 的 global split 不是区域外推测试；它只证明六区域混合总体上的表现。历史区域交叉
验证仅成功 8/18 个预定训练，因此可作为附录诊断，不能写成完整三 seed 跨区域证据。
Area_1 的固定 BIM-to-world 变换借助发布的 semantic 结构网格完成房间级标定，因此它是
**scan-calibrated / oracle-style 结构标定**，不能表述为完全无目标场景几何的零标定部署。

### 3.2 固定 support

给定有效 GT 集合 \(S\)，所有可比方法必须在完全相同的 \(S\) 上计算指标：

1. GT 在样本制备时就受 0.2--5.0 m 约束；评测时不为某个方法单独改范围。
2. `raw_da3`、尺度矫正、BIM-direct、`coarse` 和 `refined` 使用同一子集 mask。
3. support 内任何非有限或非正预测均视为评测错误，不能通过删像素改善结果。
4. E2E 必须与基于**同一在线 DA3 输出**的 `live_robust_bim_direct` 比较；缓存与在线路径
   只能作为已验证一致的辅助对照。
5. 纯 `bim_envelope` 只有 ray hit 时才有输出，必须单列 `GT valid & BIM hit` 指标和
   coverage，不能与全 support 方法直接排名。
6. `coarse` 若与某一尺度输出是严格 alias，只保留一列，避免重复方法制造表格优势。

这一规则比只报告“各方法自身有效像素”更严格，也能避免 BIM 覆盖率不同导致的虚假提升。

## 4. 指标、三层聚合与统计检验

### 4.1 深度指标

对 \(N=|S|\)、预测 \(\hat d_i\) 和 GT \(d_i\)：

\[
\operatorname{AbsRel}=\frac{1}{N}\sum_i\frac{|\hat d_i-d_i|}{d_i},\qquad
\operatorname{MAE}=\frac{1}{N}\sum_i|\hat d_i-d_i|,
\]

\[
\operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_i(\hat d_i-d_i)^2},\qquad
\delta_k=\frac{1}{N}\sum_i
\mathbf 1\!\left[\max\left(\frac{\hat d_i}{d_i},\frac{d_i}{\hat d_i}\right)<1.25^k\right].
\]

论文主表报告 AbsRel、MAE、RMSE、\(\delta_1\)，补充材料报告 \(\delta_2,\delta_3\)。这些
是单目深度论文常用指标；其经典来源可追溯至
[Eigen et al., NeurIPS 2014](https://proceedings.neurips.cc/paper_files/paper/2014/hash/91c56ce4a249fae5419b90cba831e303-Abstract.html)，
后续深度工作也沿用同类协议，例如
[Monodepth2, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.html)。

相对改善统一写为

\[
100\times\frac{M_{\mathrm{baseline}}-M_{\mathrm{candidate}}}
{M_{\mathrm{baseline}}}\%,
\]

其中误差指标越低越好。不要把百分点变化与相对百分比混写。

### 4.2 三层聚合

- **pixel-micro**：合并所有 support 像素后计算，回答“随机有效像素的期望误差”。
- **frame-macro**：每帧先计算指标，再对有 support 的帧等权平均，避免高有效像素帧
  完全主导结果。
- **group-macro**：每个房间或区域先汇总其全部 support 像素，再对 group 等权平均，
  回答“随机场景单元的期望误差”。Area_1 主张泛化时以 room-macro 为关键佐证。

三者含义不同，不应先在帧内平均再把它误称为 pixel-micro。每个结果同时记录像素数、
有效帧数和有效 group 数。

### 4.3 子集

Area_1 已冻结六个子集：

| 子集 | 定义 | 研究问题 |
|---|---|---|
| `all` | 全部有效官方 z-depth | 总体精度 |
| `furniture` | table/chair/sofa/bookcase semantic 像素 | BIM 缺失家具的恢复 |
| `non_structural` | 所有已知非围护结构语义像素 | 更广义前景泛化 |
| `bim_foreground_conflict` | GT 与 BIM 均有效，且 \(d_{GT}<d_{BIM}-\max(0.10\text{ m},0.05d_{BIM})\) | 前景遮挡围护结构 |
| `bim_consistent` | \(|d_{GT}-d_{BIM}|\leq\max(0.10\text{ m},0.05d_{BIM})\) | 是否保持可信 BIM |
| `bim_no_hit` | GT 有效但固定 BIM 无 ray hit | 无先验时的退化行为 |

主文至少报告 `all`、`furniture`、`bim_foreground_conflict`，其余放补充材料。semantic 和
conflict mask 只用于训练权重或评测分层，不进入正式推理输入。

### 4.4 配对 bootstrap

Area_1 使用房间作为重采样单位，对每个房间的
`candidate - robust_bim_direct` 差值做 10,000 次有放回配对重采样，固定 seed 42，报告
均值差、95% percentile interval 和 candidate 胜出房间比例。负差表示 candidate 更好。
同一重采样索引必须同时作用于两种方法，不能独立 bootstrap。bootstrap 的统计思想来源于
[Efron, 1979](https://doi.org/10.1214/aos/1176344552)。

当前 test 只有 7 个房间，因此区间应与逐房间结果一起解释，不能只用一个 `p<0.05` 标签。
更重要的是，房间 bootstrap 只反映评测场景采样不确定性，**不反映训练 seed 不确定性**。
当前正式模型都只有一次 `seed=42` 训练；本项目不应写“跨 seed 稳定”或给出训练方差。

## 5. 已有证据

### 5.1 两数据集正式主结果

以下均为 0.2--5.0 m、pixel-micro、同一固定 support；来源为
[`results/metrics.json`](../results/metrics.json)。

| 数据集 / split | raw DA3 | universal scale | universal BIM-direct | learned refiner |
|---|---:|---:|---:|---:|
| SLABIM clean global test，108 帧 | 0.199347 | 0.063615 | 0.062625 | **0.056013** |
| Area_1 room-disjoint test，1,641 帧 / 7 房间 | 0.301228 | 0.077521 | 0.078146 | **0.066889** |

表内为 AbsRel。SLABIM learned 相对 direct 降低 10.56%；Area_1 降低 14.41%。两个数据集
运行同一公式和参数，差异只来自数据、BIM 几何和训练权重。

### 5.2 Area_1 子集、聚合与 bootstrap

Area_1 test 的主模型相对 universal BIM-direct（正式 JSON 中兼容键仍为
`robust_bim_direct`）：

| 子集 | pixel AbsRel / MAE | frame AbsRel / MAE | room AbsRel / MAE | 结论 |
|---|---:|---:|---:|---|
| all，direct | 0.078146 / 0.138907 | 0.078809 / 0.140639 | 0.091003 / 0.166845 | reference |
| all，refined | **0.066889 / 0.117610** | **0.067672 / 0.119405** | **0.077769 / 0.141963** | 三层均改善 |
| furniture，direct | 0.089286 / 0.151608 | 0.093287 / 0.188137 | 0.086848 / 0.142063 | reference |
| furniture，refined | **0.085734 / 0.145756** | **0.090684 / 0.183606** | **0.083486 / 0.136726** | 三层均改善 |
| conflict，direct | 0.140639 / 0.179008 | 0.117081 / 0.165091 | 0.171254 / 0.207006 | reference |
| conflict，refined | **0.137720 / 0.175033** | **0.115114 / 0.161867** | **0.168528 / 0.203342** | 点估计改善，CI 跨 0 |

conflict 的 pixel AbsRel 和 room AbsRel/MAE 退化，只有 frame 层及 pixel MAE 略改善，
所以按预定合取条件必须写为“未改善”，不能用局部有利指标替代总体结论。

房间配对 bootstrap 的 `refined - direct` 结果进一步支持这一边界：

| 子集 | AbsRel room mean difference [95% CI] | MAE room mean difference [95% CI] | 胜出房间 |
|---|---:|---:|---:|
| all | -0.012055 [-0.017295, -0.006285] | -0.025625 m [-0.034186, -0.015103] | 6/7，6/7 |
| furniture | -0.004031 [-0.006578, -0.001581] | -0.006279 m [-0.008893, -0.003920] | 5/6，6/6 |
| conflict | +0.005891 [-0.006884, +0.022509] | +0.003827 m [-0.008644, +0.022054] | 3/7，4/7 |

conflict 两个区间都跨 0，现有证据不能支持未见房间上的稳定改善。已知主要退化来自
`hallway_8`；`conferenceRoom_1` 的 all 指标也有轻微退化。这两个案例应保留在 failure
analysis，而不是在看过 test 后重新调模型。

### 5.3 跨域诊断与 E2E 晋级结果

| 训练权重 → 评测域 | split | 公平 reference AbsRel / MAE | refined AbsRel / MAE | 状态 |
|---|---|---:|---:|---|
| SLABIM frozen → Area_1 | validation | robust direct 0.087100 / 0.171213 | 0.166899 / 0.375176 | 7/7 房间失败，仅作域偏移诊断 |
| SLABIM E2E → Area_1 | validation | live robust direct 0.087064 / 0.170111 | 0.164560 / 0.366513 | 7/7 房间失败，仅作域偏移诊断 |
| Area_1 frozen → Area_1 | validation | robust direct 0.087100 / 0.171213 | **0.070050 / 0.139691** | 晋级主模型 |
| Area_1 E2E → Area_1 | validation | live robust direct 0.087218 / 0.171611 | 0.070185 / 0.140186 | 胜 direct，但未胜 frozen |

因此，现有网络不是无需适配即可跨建筑直接部署的通用 refiner。这里的“通用”指尺度
估计和网络结构跨数据集一致，不表示同一组 residual 权重无需目标域训练即可跨建筑部署。

### 5.4 已有敏感性证据的边界

旧区域协议消融已从公开精简结果移除，不能用于当前统一协议的定量 claim。当前可直接
作图的尺度敏感性证据来自 Area_1 **train only**：\(c_{10}\) 有
8 个候选、\(c_{25}\) 有 6 个候选，共 48 格；按等房间 scale-only AbsRel 选出
\(c_{10}=\infty,c_{25}=0.05\)，全 train room-macro AbsRel 为 0.092673。选择过程中打开
validation/test 样本数均为 0。热图应直接读取
[`stanford_area1_robust_scale_selection_v1.json`](../data/provenance/stanford_area1_robust_scale_selection_v1.json)，
不能用 test 重画或重选最优点。

## 6. 建议补跑：最小可发表消融

非学习 BIM-direct 的逐因素 validation 消融已经完成，见
[DETERMINISTIC_BASELINE_ABLATION.md](DETERMINISTIC_BASELINE_ABLATION.md)。本节以下项目主要
针对学习网络，仍属于建议补跑。

所有新增结构比较只在冻结 validation 上完成；看过现有 test 后产生的新结构不再具有原来
“一次盲测”的身份。若未来需要最终确认，应使用新的外部建筑或预先冻结的新 test。

| ID | 变体 | 唯一改动 | 回答的问题 | 优先级 |
|---|---|---|---|---|
| A0 | universal BIM-direct | 无学习 | 最强确定性基线 | 必须 |
| A1 | full frozen refiner | 当前完整模型 | 主模型 | 必须 |
| A2 | no BIM features | BIM 分支置零，但尺度 anchor 保持相同 | 网络是否真正读取局部 BIM | 必须 |
| A3 | no RGB | RGB 分支置零 | 家具恢复是否来自视觉信息 | 必须 |
| A4 | single residual | 用单一残差替代 frame/low/detail heads | 多尺度残差是否必要 | 必须；历史结果有反例 |
| A5 | no depth routing | 固定合并残差 heads | 深度感知路由贡献 | 必须 |
| A6 | no reliability/trust | 删除可靠度辅助头和相关 loss | 可信度建模贡献 | 建议 |
| A7 | legacy Q45 anchor | robust scale 改为旧 Q45，其余不变 | 稳健尺度对学习模型的贡献 | 必须 |
| A8 | no foreground weighting | 家具/conflict 权重恢复 1.0 | 目标域前景监督贡献 | Area_1 必须 |
| A9 | E2E last-stage | 只解冻同一 decoder 尾部 | 部分 DA3 微调是否值得 | 已有负结果，可补效率 |

公平控制如下：

1. 同一 annotation、输入缓存、初始化、epoch 上限、early-stop、优化器步数和 augmentation；
2. 一次只改变一项，报告参数量和实际完成 epoch；
3. 用 validation pixel AbsRel 选 checkpoint，但最终消融表同时给出 all/furniture/conflict
   的 pixel 与 room 结果；
4. 当前若维持单 seed 策略，表题必须标注 `seed=42, single run`，差异很小的变体只写
   “观察到差异”，不写“稳定优于”；
5. 若审稿阶段需要最低限度的优化稳定性，优先只为 A1、最强 ablation 和 A0 增加三个
   训练 seed；不要把房间 bootstrap 冒充 seed 统计。

## 7. 建议补跑：敏感性与鲁棒性

### 7.1 BIM 配准误差

实际部署最重要的敏感性是固定 BIM 与相机坐标链误差。只在 validation 上对 BIM 施加
确定性扰动，DA3/RGB/GT 不变：

- translation magnitude：0、2、5、10、20 cm；每档对 \(\pm x,\pm y,\pm z\) 分别运行，
  先按方向平均再报告范围；
- yaw：0、\(\pm0.5^\circ\)、\(\pm1^\circ\)、\(\pm2^\circ\)、\(\pm5^\circ\)；
- 可选 pitch/roll：0、\(\pm0.5^\circ\)、\(\pm1^\circ\)、\(\pm2^\circ\)。

绘制 robust direct 与 refined 的 AbsRel--扰动曲线，并报告相对无扰动性能退化 5%、10%
时的容忍阈值。每个扰动必须重新 ray-cast BIM，而不能只平移已有深度图。

### 7.2 BIM 覆盖与内容缺失

- 在 BIM valid mask 上采用固定 seed 的**连通块遮挡** 0%、10%、25%、50%、75%，而非只
  做独立像素 dropout；这更接近缺失房间或未建模构件。
- 分别移除 ceiling、floor、wall、column/beam 类别，考察哪类围护先验最关键。
- 按原始 BIM hit coverage 分五分位报告 frame-macro 误差和 learned-direct gain。
- 单列 `bim_no_hit`，检查模型是否平滑退回 RGB/DA3，而不是产生异常深度。

### 7.3 输入、数据量与微调强度

| 因素 | 建议取值 | 控制原则 |
|---|---|---|
| 输入分辨率 | 336、420、504 | 同一 resize/crop 规则；报告精度、延迟和显存 |
| Area_1 训练数据比例 | 10%、25%、50%、100% | 以**房间**构造固定嵌套子集，不能逐帧随机抽样 |
| 深度段 | [0.2,1)、[1,2)、[2,3)、[3,5] m | 只分层统计，不改变训练或总 support |
| decoder learning rate | 0、1e-7、3e-7、1e-6、3e-6 | 只用 validation 选择；0 即 frozen control |
| 解冻范围 | none、output conv、last stage | 与 trainable params、显存、延迟一起报告 |

曲线应显示全部网格点，不只显示最优点。训练数据比例和 E2E 网格若仍为单 seed，应避免
对小差异作显著性解释。

## 8. 效率评测

### 8.1 已有结构规模

- frozen refiner：2,521,654 个可训练参数，checkpoint 约 30 MB；
- DA3 metric-large：约 334,171,394 个参数；
- partial E2E 完整系统：336,693,048 个参数，其中 5,279,735 个可训练，约 1.568%；
- E2E checkpoint 约 1.3 GB。

这些是模型规模，不是速度证据。现有 summary 记录 batch size 和 worker 数，但没有统一
计时，因此目前不能声称某条路线具有确定 FPS 优势。

### 8.2 建议统一 benchmark

在同一 GPU、CUDA、PyTorch、504×504 输入下分别测试：

1. DA3 online；
2. DA3 + robust scale；
3. DA3 + universal BIM-direct；
4. DA3 + frozen refiner；
5. partial E2E；
6. BIM ray casting 预处理，单独计时。

每种方法先 warm-up 50 次，再运行至少 200 帧；对 batch size 1/4/8 报告 median、P90
latency、FPS、峰值 GPU 显存、CPU 内存、总参数/可训练参数和 checkpoint 大小。计时前后
调用 CUDA synchronize。公开 GPU 型号、驱动、精度模式、编译选项和是否包含磁盘 I/O。
“cached DA3”只代表离线训练加速，不能用其时间代表端到端部署延迟。

## 9. Failure analysis

### 9.1 定量诊断

对每帧计算 \(\Delta=\operatorname{AbsRel}_{refined}-
\operatorname{AbsRel}_{direct}\)，并与下列因素关联：

- BIM hit coverage、conflict 像素比例、furniture 像素比例；
- 稳健尺度值、Q10/Q25 cap 是否触发、有效 ratio 数；
- RGB/GT/BIM 深度边缘密度；
- GT 距离段、房间/区域、相机 UUID；
- BIM 与 GT 在 consistent/conflict 区的初始误差。

连续变量报告 Spearman \(\rho\) 和 bootstrap CI，同时画散点/分箱趋势；类别变量报告
group-macro 误差与 support。相关性只用于定位机制，不能解释为因果。房间内多帧相关时，
CI 仍以房间为重采样单位。

### 9.2 定性样本选择

主文示例应先在 validation 上按固定规则选取，避免 test cherry-picking：

- **典型成功**：在足够 support 的帧中，选择 learned-direct gain 最接近中位数者；
- **家具/冲突成功**：要求 furniture/conflict support 超过预设阈值，再选 gain 较高但非
  单一最大值的代表帧；
- **困难或失败**：选择 \(\Delta>0\) 且 support 足够的代表帧，原样展示退化。

可另放 `hallway_8` 的最大 conflict 退化帧作为**事后 test failure diagnosis**，但图注
必须写清它是看过 test 后选择的失效案例，不参与性能证明。

## 10. 图资产的三套备选方案

所有图件应使用真实 RGB、GT、缓存预测、BIM render 和冻结模型输出；不使用生成式图像
代替实验结果。建议根目录为 `docs/assets/paper_evaluation/`，PNG 用于 PPT，曲线/框图同时
保存 SVG。每个素材独立存储，不把标题、箭头、图例永久烧进原始深度图。

已生成图件、三套备选及其适用场景见
[`assets/paper_evaluation/README.md`](assets/paper_evaluation/README.md)。

### 方案 A：经典论文横向对比（建议主文）

每个示例组提供独立 panel：

```text
rgb.png
gt_depth.png
raw_da3.png
bim_prior.png
robust_direct.png
refined.png
raw_error.png
direct_error.png
refined_error.png
depth_colorbar.png
error_colorbar.png
```

PPT 中按 `RGB | GT | Raw DA3 | BIM-direct | Ours` 排第一行，第二行只排三种 error map。
优点是审稿人能迅速比较输入、基线与输出；缺点是网络为何改善冲突区不够直观。

### 方案 B：前景冲突机制（建议方法图或补充材料）

在方案 A 的核心 panel 外，增加：

```text
furniture_mask.png
conflict_mask.png
bim_valid_mask.png
improvement_map.png
rgb_conflict_overlay.png
```

PPT 中以 RGB 为大图，家具与 conflict mask 作小插图，右侧放 direct/refined error 和
`direct error - refined error` 改善图。正值和负值使用以 0 为中心的发散色图。该方案
最能说明“固定围护 BIM 与真实家具深度”的矛盾，也会诚实暴露退化区域。

### 方案 C：成功—边界—失败三联案例（建议答辩或局限性页）

准备三组相互独立的素材目录：

```text
qualitative/set_a_typical_success/
qualitative/set_b_furniture_conflict/
qualitative/set_c_hard_failure/
```

每组均保存方案 A 的 panel 和一个 `preview_sheet.png`。PPT 可选其中两组作主文，也可把
三组拼成三行，分别讲总体收益、家具冲突恢复和方法边界。失败组不得隐藏；它与 Area_1
conflict test 的负结果相互印证。

### 10.4 过程图的三种可选布局

1. **线性 coarse-to-fine**：`RGB → DA3 → robust metric scale → BIM-conditioned refiner → depth`，
   BIM 以旁路箭头进入 scale 和 refiner。适合论文 overview。
2. **双流网络结构**：RGB/DA3 几何流与 BIM 几何流分别编码，在多尺度 residual heads 和
   depth routing 汇合。适合 architecture 页。
3. **科研协议流程**：公开源数据 → 质量排除/固定 split → train-only BIM 尺度选择 →
   train → validation 晋级 → test once。适合 reproducibility 或答辩页。

过程图各模块同时保存独立 SVG/透明 PNG，便于 PPT 重排；箭头、文字标签和数据卡片也
分别保存，不要只留一张不可编辑的大合成图。

### 10.5 统一视觉与 provenance

- 所有 depth panel 固定 0.2--5.0 m 色域；所有相对误差 panel 固定同一截断范围，越界
  用端点色显示，不逐图自动拉伸。
- invalid 用统一灰色或透明 alpha；mask 只使用 0/1，不做抗锯齿后再用于计算。
- 原图保持 504×504 像素；不通过 AI 放大制造细节。图表另导出 300 dpi PNG 和 SVG。
- `preview_sheet.png` 只供选择；论文/PPT 应从独立 panel 拼接。
- 每套目录写 `manifest.json`，至少记录数据集、split、sample ID、选择规则、checkpoint
  SHA-256、各方法指标、色域、源文件和 panel SHA-256。
- 主结果柱图、房间 paired-slope/forest 图、48 格 train-only sensitivity heatmap、历史
  单 seed ablation 图分别存放，图题明确写 `test`、`validation` 或 `train-only`。

## 11. 建议的论文结果组织

主文采用最小、可防止误读的结构：

1. Table 1：数据规模、split、BIM 来源和固定标定假设；
2. Table 2：SLABIM 与 Area_1 的固定 support 主结果；
3. Table 3：Area_1 all/furniture/conflict × pixel/frame/room；
4. Figure 1：方案 A 的典型成功和方案 B 的家具冲突示例；
5. Figure 2：Area_1 逐房间 direct→refined paired slope 与 95% bootstrap CI；
6. Figure 3：train-only 48 格 robust-scale 热图；
7. Table 4：当前协议重跑后的最小消融；
8. Supplement：跨域失败、E2E 未晋级、效率、配准/coverage 敏感性以及方案 C 失效案例。

摘要和结论可以写“总体与家具区域优于直接 BIM 矫正”，不能写“所有 BIM 冲突区域均
改善”“E2E 必然优于冻结模型”“无需目标域训练即可跨建筑部署”或“多 seed 稳定”。

## 12. 可审计来源

- 两数据集紧凑主表：[`results/metrics.json`](../results/metrics.json)
- SLABIM 正式结果：
  [`test_summary.json`](../results/slabim/test_summary.json)
- Area_1 正式 validation/test：
  [`val_summary.json`](../results/stanford_area1/val_summary.json)、
  [`test_summary.json`](../results/stanford_area1/test_summary.json)
- train-only 稳健尺度选择：
  [`stanford_area1_robust_scale_selection_v1.json`](../data/provenance/stanford_area1_robust_scale_selection_v1.json)
- 跨数据集冻结尺度协议：
  [`universal_scale_estimator_v1.json`](../data/provenance/universal_scale_estimator_v1.json)
- 完整 Area_1 标定和 test-once 边界：
  [`STANFORD_BIMSYNC_EVALUATION.md`](STANFORD_BIMSYNC_EVALUATION.md)
