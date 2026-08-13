# Stanford 2D-3D-S Area_1 + BIMSyn 评测协议

## 1. 目标与当前状态

本协议评测一个明确受限的问题：给定单帧 RGB、相机内外参以及一个覆盖整个 Area_1 的
**单一、固定 BIM 核心围护先验**，估计包含桌椅、沙发、书柜及杂物等前景物体的完整
场景 z-depth。这个全局先验由 44 个房间 IFC 经固定房间变换合并而成；所有帧 ray-cast
同一个 Area_1 坐标系场景，而不是按当前房间切换 BIM。BIM 不提供家具真值；模型需要
保留围护结构提供的尺度和空间约束，同时从 RGB/DA3 中恢复比 BIM 更靠近相机的前景。

截至本文写入时，已经完成并可审计的事实包括：

- Stanford 2D-3D-S `Area_1/no_xyz` 压缩包已按官方大小和 MD5 校验；
- RGB、depth、pose、semantic 四种 regular-view 模态各有 10,327 帧，配对键完全一致；
- 数据覆盖 44 个房间和 186 个 `camera_uuid`；
- BIMSyn 中取得 44 对同名 IFC/RVT 文件，与 Area_1 的 44 个房间一一对应；
- 固定核心围护过滤器、schema 2 房间级配准和 Area_1 全局场景合并已实现；正式配准
  回执中的 44 个房间全部通过预设质量门槛，未出现失败房间；
- 完整 manifest 已冻结为 10,327 帧，正式 room-disjoint annotation 已冻结为
  7,013/1,673/1,641 个 train/validation/test 样本；房间和 `camera_uuid` 均不跨 split；
- 只使用 30 个 train 房间完成了稳健尺度上限选择，并把选择协议、访问 ID 和代码身份
  写入不可变回执；该过程打开 validation/test 样本数均为 0；
- source frozen refiner 的 Area_1 validation-only zero-shot 评测已落盘。结果显示原域模型
  直接迁移明显劣于无需学习的稳健 BIM 矫正，因此它只能作为域差异诊断，不能作为已
  达成的最终模型结论；
- source E2E checkpoint 的 validation-only zero-shot 评测也已完整落盘；在线 DA3、在线
  稳健比较器及模型输出均按相同 support 评测，结论与 frozen source 一致：学习输出在
  7/7 validation 房间上均劣于相同在线 DA3 的稳健 BIM-direct；
- target frozen-DA3 refiner 已完成 12 个 epoch，并在正式 validation 的
  all/furniture/conflict 三个核心子集、pixel/frame/room 三种聚合上都同时降低了 AbsRel
  和 MAE；
- target E2E last-stage challenger 从 target frozen checkpoint 初始化，训练 5 个 epoch
  后 early-stop。其最佳 validation pixel AbsRel/MAE 略差于 frozen 主模型，因此未晋级；
- 只对锁定后的 target frozen 主模型执行了一次盲测。test all pixel AbsRel/MAE 从
  robust BIM-direct 的 0.078146/0.138907 降至 0.067924/0.117484；家具子集的三种聚合
  全部改善。冲突子集只有 frame macro 同时改善，pixel/room 结果混合，本文原样披露；
- test 输出落盘后未改模型、尺度参数、support、split 或房间标定，也未运行第二个 test
  challenger。不存在依据 test 结果进行的重新训练、调参或重跑。

因此本文的最终主模型是 **target frozen-DA3 robust refiner**；E2E 是未晋级的 validation
challenger。第 11 节给出完整 validation/test 数字、bootstrap 区间和不可变产物哈希。

## 2. 数据来源、完整性与许可边界

### 2.1 Stanford 2D-3D-S Area_1

使用的是 2D-3D-S 的 `no_xyz/area_1_no_xyz.tar`，只解出本协议需要的 regular-view
`rgb`、`depth`、`pose`、`semantic`，以及 `3d/semantic.obj` 和
`3d/semantic.mtl`。来源与本地校验值为：

| 项目 | 已核验值 |
|---|---:|
| 下载地址 | `https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/area_1_no_xyz.tar` |
| 文件大小 | 32,684,605,440 bytes |
| MD5 | `21098fbe93b561e30e79197a95fa4fd2` |
| regular RGB/depth/pose/semantic | 各 10,327 帧 |
| 房间 / camera UUID | 44 / 186 |

数据格式按官方 metadata 解析：原图为 1080×1080；depth 是 uint16 PNG，单位换算为
`raw / 512` 米，`65535` 表示无效值，regular-view depth 是相机平面的 z-depth；
semantic PNG 的 RGB 三通道编码一个 24-bit 标签表索引。项目使用的 metadata 仓库
提交为 `54a532959b20203ea4c3fcc26f5c8bf678d6fdb4`。

2D-3D-S 数据下载要求先接受其数据许可。本项目按该注册许可将数据用于学术、非商业
研究，不再分发原始图像、深度、网格或由其直接打包得到的数据。metadata 仓库中的
Apache-2.0 只覆盖仓库软件，不能替代数据许可。任何公开发布、商业使用或向第三方
传递数据前，都应重新核对并遵守[官方数据许可入口](https://docs.google.com/forms/d/e/1FAIpQLScFR0U8WEUtb7tgjOhhnl31OrkEs73-Y8bQwPeXgebqVKNMpQ/viewform)。

### 2.2 BIMSyn IFC/RVT

BIM 文件来自用户给定的[BIMSyn SharePoint 目录](https://szueducn-my.sharepoint.com/:f:/g/personal/shengjuntang_szu_edu_cn/EgpOf3leEiFOjtfQTl-k7GgB9NHyLKhaRCnaUhD-jD0-yw)，
数据关联论文页面为[Automation in Construction 156 (2023), 105076](https://www.sciencedirect.com/science/article/pii/S0926580523003369)。
本地回执已核验：

| 项目 | 已核验值 |
|---|---:|
| IFC | 44 个，合计 125,546,311 bytes |
| RVT | 44 个，合计 603,906,048 bytes |
| IFC/RVT 文件 stem | 44 对一一相同 |
| 文件签名 | IFC STEP header 与 RVT compound-file header 均通过检查 |

当前代码只读取 IFC；RVT 用于来源留档和人工 BIM 检查，不是训练或推理输入。来源目录
中未发现可由当前回执独立证明的再分发许可，因此本项目不再分发 IFC/RVT，公开模型和
派生数据前须按论文/数据发布者条款复核授权。

固定的 88 文件 SHA-256/字节清单在
[`bimsyn_models_manifest.json`](../src/bim_priorda3/data/bimsyn_models_manifest.json)；
`scripts/data/verify_stanford_bimsyn_sources.py` 会为每个本机下载另写带本机路径的
来源 receipt。

## 3. Area_1 与 BIMSyn 的对应关系

两套数据通过房间名严格对应，而不是通过图像检索或测试帧拟合。Area_1 pose 中的
`room` 带 Area 后缀，例如 `office_1_1`；移除最后的 Area 编号后得到 BIM stem
`office_1`。44 个房间由 1 个 WC、2 个 conference room、1 个 copy room、8 个
hallway、31 个 office 和 1 个 pantry 组成；每个 stem 都同时存在一个 IFC 和一个
RVT。

发现数据时，代码要求 RGB、depth、semantic、pose 的文件配对键完全相同，并要求每个
pose 解析出的房间都存在同 stem IFC。任何缺帧、重复键、未知房间或 IFC/RVT stem
不一致都会直接报错，不做模糊匹配。

## 4. 单一固定 BIM 核心围护先验

### 4.1 IFC 解析 allow-list 与实际渲染 core-list

IfcOpenShell 在产品 world placement 下三角化 IFC，并输出米制、房间局部 BIM 坐标。
解析与配准阶段采用严格的结构类别 allow-list：

- wall / curtain wall；
- floor slab、floor covering；
- ceiling covering、roof/roof slab；
- door、window、column、beam。

`IfcFurnishingElement`、`IfcBuildingElementProxy`、`IfcFlowTerminal`、
`IfcOpeningElement` 以及所有其他非 allow-list 产品都不会作为可见先验三角形加入渲染。
这一限制很重要：BIMSyn 的完整 IFC 中确实含有家具、板件、代理构件和 MEP 对象；如果
直接渲染完整 IFC，会把需要模型预测的前景泄露进输入。

正式深度输入进一步只保留核心类别 `wall/floor/ceiling/column/beam`。`door/window`
虽然可用于结构配准的类别审计与候选重排，但不进入全局 ray-cast 场景，因为 IFC 中门窗
的开闭状态与采集图像不同步，闭合门扇尤其可能错误遮挡穿过门洞的视线。门窗洞口仍由
宿主墙体的 IFC 布尔几何体现；`IfcOpeningElement` 本身、家具、proxy、MEP 和其他对象
始终不作为可见先验表面。

制备阶段先用 44 个冻结的 `T_area_from_bim[room]` 把各房间 core-list 三角形变换到
Stanford Area_1 坐标，再合并为**一个**不可变的全局 ray-casting scene。此后所有样本
只使用 `T_area_from_camera` 在同一 scene 中求交。这一点避免了只渲染“当前房间 IFC”
时在房门、走廊和相邻空间方向上产生的错误截断。

每个样本保存 `bim_depth`、`bim_valid`、相机坐标法线、深度边缘和围护类别。BIM ray
超出 5.0 m 或无命中时标为无效。相机内参从 1080×1080 等比例变换到协议分辨率
504×504。

已冻结的全局 core prior 审计值如下；这些是几何规模，不是模型指标：

| 项目 | 已核验值 |
|---|---:|
| 合并房间 | 44 |
| 全局顶点 / 三角形 | 71,506 / 9,460 |
| 保留三角形 | wall 5,236；floor 1,300；ceiling 1,196；column 1,044；beam 684 |
| 排除三角形 | door 117,081；window 5,724 |
| Area_1 bounds min | `[-22.5769, -2.1201, -0.1386] m` |
| Area_1 bounds max | `[1.9751, 46.0543, 4.5691] m` |
| 全局 BIM fingerprint | `ed4faec789125d77e749bc423732bc9abb43c8fc5bd3c1a9567939aacb4eb87c` |

逐房间来源、变换、类别计数和全局汇总保存在运行时生成的
`data/processed/stanford_area1_504/metadata.json`；`data/processed/` 是本机缓存，不随
公开仓库分发。

### 4.2 固定性约束

每个房间只允许一个 `T_area_from_bim`，它只在离线构建 Area_1 全局先验时使用，并对
该房间所有相机位置和方向保持不变；合并完成后，每一帧都读取同一份全局几何。禁止
以下操作：

- 用某一评测帧的 GT depth 单独求尺度、ICP 或位姿；
- 按帧平移/旋转 BIM 来提高指标；
- 用 test 指标选择房间变换；
- 将 BIMSyn 家具或 Stanford 家具网格加入先验渲染。

因此这里的“固定 BIM”表示由固定房间级标定合成、随后冻结的单一 Area_1 先验，不表示
BIM 和相机天然处于同一坐标系，也不表示完全不需要目标场景标定。

## 5. 坐标约定与房间级配准

### 5.1 坐标链

regular pose JSON 实际存储的 `camera_rt_matrix` 是 3×4 的 area-to-camera
`[R | t]`，尽管旧 README 文本曾写成 4×3。加载器通过
`R @ camera_location + t ≈ 0`、旋转正交性和行列式检查确认该约定，然后求出
`T_area_from_camera`。相机投影采用常规 CV 坐标：x 向右、y 向下、z 向前。

发布的 `semantic.obj` 使用 Blender Y-up 世界轴，而 regular pose/depth 使用 Area
Z-up 轴。进入配准前执行固定、非拟合的正确旋转：

```text
(x_area, y_area, z_area) = (x_obj, -z_obj, y_obj)
```

IFC 最初保留 room-local BIM 坐标。离线构建全局先验时的坐标链为：

```text
room-local BIM --T_area_from_bim[room]--> Stanford Area_1 global scene
```

合并并冻结后，每帧 ray casting 的坐标链为：

```text
camera --T_area_from_camera--> Stanford Area_1 global scene
```

样本中的 `camera_to_bim`/`area_from_bim` 保留作配准审计与兼容字段；正式全局 BIM 深度
由 `camera_to_area` 和合并后的全局 scene 生成，不再按样本选择 room-local scene。

### 5.2 配准输入、约束与质量门槛

配准只比较两类结构几何：Stanford `semantic.obj` 中的 ceiling、floor、wall、beam、
column、window、door，以及 IFC allow-list 围护三角形。配准模块不加载 regular RGB、
regular depth 或逐帧 semantic PNG。

优化是对称、鲁棒的几何 ICP，变换被约束为：

- 固定单位尺度；
- 只允许绕共享 Z-up 轴的 yaw；
- 允许 XYZ 平移；
- 每房间固定一次。

正式回执为 schema 2，方法标识是
`structural-mesh-symmetric-constrained-yaw-icp-class-rerank-v2`。结构类别**不参与**粗配准
或 ICP 拟合，也不会把几何上不合格的解救回来；它们只在几何质量门槛已通过、且目标值
与纯几何最优解近等价的候选之间做第二次排序。重排比较同类结构点，并用
door/window/beam/column 等具有方向判别力的类别避免对称房间出现约 180° 歧义。因此
schema 2 的类别信息是受几何容差约束的候选消歧信号，而不是新的逐像素监督或评测帧
拟合。正式回执中只有 `copyRoom_1` 和 `office_15` 改选了纯几何最优候选之外的近等价
候选，其余房间保留纯几何选择；每个候选的几何目标、类别分数和选择原因均写入回执。

正式回执使用 seed `20260810`、8,000 个细化采样点、1,500 个粗采样点、36 个 yaw
起点、4 个细化候选、最多 25 次迭代，以及从 1.5 m 递减到 0.18 m 的对应阈值。验收
要求 0.20 m 阈值下 fitness 不低于 0.55，RMSE 不高于 0.15 m。正式回执中 44/44
房间通过，0 个失败；这只是配准质量控制，不是深度模型结果。

正式回执为
[`data/provenance/stanford_area1_bimsyn_alignment.json`](../data/provenance/stanford_area1_bimsyn_alignment.json)。

### 5.3 必须披露的校准假设

房间变换使用了 Area_1 发布的、带结构类别标签的 3D semantic mesh。因此本协议属于
**scan-calibrated、目标场景结构网格辅助的已标定 BIM 先验评测**。就部署假设而言，
这是 oracle-style 的目标结构 mesh 辅助标定，不能表述为“不接触目标场景几何的零标定
部署”。它没有使用逐帧 GT depth，也没有把家具网格放入 BIM 输入，但 train/val/test
房间的扫描结构网格都参与了各自的固定标定；因此“zero-shot transfer”只表示模型权重
不读取 Area_1 梯度，不表示 BIM-to-world 标定也是 zero-shot。面向真实部署时，应以
测量控制点、SLAM/测绘配准或已有 BIM-to-world 标定替代这一步，并继续冻结所得变换和
合并后的全局先验。

## 6. 样本制备与深度协议

### 6.1 样本字段

所有主实验统一使用 504×504 和 0.2–5.0 m。这个范围同时约束 GT 有效像素和 BIM
ray hit，不是评测后临时裁剪。每帧 NPZ 包含：

```text
base_depth, base_confidence
bim_depth, bim_valid, bim_normals, bim_edge, bim_category
gt_depth, gt_valid, gt_weight
intrinsic, camera_to_bim, camera_to_area, area_from_bim
semantic_class, structural_mask, furniture_mask,
non_structural_mask, semantic_valid
```

`base_*` 是 `depth-anything/da3metric-large` 在该 RGB 上的输出，可先缓存以避免不同
方法重复运行 DA3。GT 是官方完整场景 z-depth，因而包含家具及其他可见物体。逐像素
semantic mask 不作为 refiner 或 E2E 模型的推理输入；它在 validation/test 用于分层
报告，在 target-domain train 中还用于构造监督损失权重。GT 只用于训练损失和评测。
每个样本记录 RGB、depth、semantic、pose、IFC、全局 BIM、配准回执、分辨率、深度
范围和 DA3 配置的 fingerprint，任一受保护输入变化时旧样本不能静默复用。

### 6.2 DA3 缓存身份

正式配置把 DA3 身份固定为：

| 字段 | 固定值 |
|---|---|
| model | `depth-anything/da3metric-large` |
| resolved revision | `4010e39f3634a45bc60553321fb49fb760bd594e` |
| process resolution / target shape | `504 / 504×504` |
| load policy | 首次制备允许联网下载；E2E 从本地 pinned snapshot 加载 |

schema 2 cache 除 `depth/confidence` 外，还必须保存原 RGB SHA-256、模型名、resolved
revision、处理分辨率、目标 shape、加载策略和 provenance status；读取时逐项严格校验，
不一致即失败。新推理标为 `direct_inference`。

公开复现应直接生成 schema 2 cache。历史实验曾把早期两字段 cache 迁移成带用户声明的
schema 2；该 10,327 项过程回执包含机器绝对路径，已从发布版清理。正式 summary 仍披露
历史 cache 的 `legacy_user_attested` 状态，不能把它表述为原推理过程的密码学证明。fresh
run 则会记录 `direct_inference`，并产生新的 preparation fingerprint；两者不能静默混用。

## 7. Room-disjoint 30/7/7 划分

所有 44 个房间一次性划分为 30 train、7 validation、7 test。划分单位是完整房间，
不按帧随机抽样；同一房间的所有相机方向和 `camera_uuid` 必须属于同一 split。生成器还
显式检查 `camera_uuid` 不跨 split。

划分算法首先按房间类型求满足精确 30/7/7 总数的整数配额，再用 seed `42` 进行
20,000 次确定性候选搜索，目标为：

```text
frame-ratio squared error + 0.25 * camera-ratio squared error
```

这样既保持房间类型分层，也尽量使帧数和相机位置数接近 30:7:7 的目标比例。最终
annotation 只保存 `id` 和 `split`，引用 manifest，不复制 RGB、depth、pose、semantic
或 IFC。正式结果为：

| split | 房间 | 帧 | camera UUID |
|---|---:|---:|---:|
| train | 30 | 7,013 | 125 |
| validation | 7 | 1,673 | 30 |
| test | 7 | 1,641 | 31 |

房间归属已经冻结：

- train：`WC_1`、`conferenceRoom_2`、`copyRoom_1`、`hallway_1`、`hallway_3`、
  `hallway_4`、`hallway_6`、`hallway_7`、`office_1`、`office_10`、`office_11`、
  `office_12`、`office_13`、`office_15`、`office_17`、`office_18`、`office_19`、
  `office_2`、`office_21`、`office_25`、`office_26`、`office_27`、`office_29`、
  `office_3`、`office_4`、`office_5`、`office_7`、`office_8`、`office_9`、`pantry_1`；
- validation：`hallway_2`、`hallway_5`、`office_14`、`office_24`、`office_28`、
  `office_31`、`office_6`；
- test：`conferenceRoom_1`、`hallway_8`、`office_16`、`office_20`、`office_22`、
  `office_23`、`office_30`。

回执确认 `room_disjoint=true`、`camera_uuid_disjoint=true`。以下文件已经冻结：

```text
data/processed/stanford_area1_504/manifest.jsonl
data/annotations/stanford_area1_room_v1.jsonl
data/annotations/stanford_area1_room_v1_receipt.json
```

关键身份为：

```text
manifest SHA-256          6bdf397abe30984c3e920068a03c09d5982d4b72e28a142be5c51f25794ba4df
annotation SHA-256        18f4e68838f24ee10feba23f66d4baddd005e5eac5ed5288a459224152c0ed59
split fingerprint         baa668313880c29bb22bb0b66eb4b8c98b76198d84f5d99d51598d9fd78f0556
preparation fingerprint   e844ff92071b60170d174dd3536e34b379438463408e08015da375a36611fb2f
```

在 annotation 和 fingerprint 冻结后，不得因 zero-shot 或 fine-tune 的 test 结果重新
抽取房间。

训练 DataLoader 使用 `region_balanced_sampling: true` 和
`region_balance_exponent: 0.5`。设房间有 `n` 帧，则其中每帧的采样权重为
`n^-0.5`：房间总权重与 `sqrt(n)` 成正比，处在完全按帧采样和每房间严格等权之间。
这既抑制大房间主导梯度，也避免把极小房间重复采样到与最大房间完全相同而导致过拟合。
validation/test 不使用该重采样器。

## 8. 对比方法与评测输出

### 8.1 方法定义

同一评测脚本一次报告以下方法：

| 名称 | 定义 | 目标域训练 |
|---|---|---|
| `raw_da3` | 缓存的冻结 DA3 深度，不使用 BIM | 无 |
| `legacy_global_scale_q45` | 历史方法：`Q45(log(BIM/DA3))` 全局尺度 | 无 |
| `legacy_bim_direct_q45` | 历史 Q45 尺度加既有固定 BIM 局部矫正 | 无 |
| `robust_global_scale` | train-only 选定并冻结的稳健 log-cap 尺度 | 无 |
| `robust_bim_direct` | 稳健尺度加完全相同的固定 BIM 局部矫正 | 无 |
| `bim_envelope` | 原始固定围护 BIM ray-cast，仅在命中处报告 | 无 |
| `coarse` | 学习模型局部细化前的度量尺度输入 | 取决于 checkpoint |
| `refined` | BIM/RGB/DA3 条件下的学习残差输出 | 取决于 checkpoint |
| `live_da3` / `live_*_global_scale` | E2E 在线 DA3 及 legacy/robust BIM 尺度诊断 | E2E 时有 |
| `live_*_bim_direct` | 对同一在线 DA3 应用相应尺度与同一局部 BIM 矫正 | E2E 时有 |

`robust_bim_direct` 是目标域实验中学习方法必须超过的主要确定性 BIM baseline；保留
legacy 两列是为了与此前 Q45 实验连续对比，不能用较弱的 legacy baseline 代替主要验收
基准。它们都不是“直接把 BIM 深度当完整场景 GT”。`bim_envelope` 单独显示纯围护先验
在 hit support 上的误差和覆盖率。E2E 的 live direct 列使 `refined` 与基于同一在线 DA3
的非学习矫正公平比较，避免把 DA3 decoder 的变化误归因于 refiner。

#### Train-only 稳健尺度选择

对每帧有效 BIM/DA3 ratio 的对数分布，最终运行时估计器为：

```text
s_robust = exp(min(Q45(log(BIM/base)), Q25(log(BIM/base)) + 0.05))
```

ratio 只在 `[0.2, 5.0]` 内统计，少于 100 个样本时回退尺度 1.0。运行时输入只有
`base_depth` 和 `bim_depth`；GT 和 semantic 只用于在 **train split 内**选择固定 cap，
不会成为部署输入。预注册网格共有 48 个 `(c10,c25)` 候选，按 leave-one-train-room-out
选择：30 个 fold 中 27 个选择 `(inf,0.05)`，3 个选择 `(inf,0.075)`；全 train refit
唯一选定 `(inf,0.05)`。最终规则在 7,013 个 train 帧中由 Q25 cap 触发 2,018 帧，
Q10 cap 触发 0 帧，fallback 0 帧。

选择回执为
[`data/provenance/stanford_area1_robust_scale_selection_v1.json`](../data/provenance/stanford_area1_robust_scale_selection_v1.json)，
文件 SHA-256 为 `c78bcea75c5387f3fabef466505624f1ea815595511576a96c4af9282d661d2d`，
协议 SHA-256 为 `2471765aad8508934f1e658de954a26c18996d746a5e8191da4e2fb17e2894c9`。
回执记录 `validation_samples_opened=0`、`test_samples_opened=0`，并把 annotation、split、
manifest、准备 fingerprint、train ID 顺序和 selector 代码哈希全部绑定。

选择完成后的 train-only direct audit 如下。这里“local”是相同的既有局部矫正；所有值
都是审计/选择集结果，不是 validation 或 test 泛化结论。

| 方法 | all pixel AbsRel | all pixel MAE | all room AbsRel | furniture pixel/room AbsRel | non-structural pixel/room AbsRel |
|---|---:|---:|---:|---:|---:|
| legacy Q45 scale | 0.103819 | 0.173222 | 0.100257 | 0.119561 / 0.136976 | 0.121430 / 0.101891 |
| legacy Q45 + local | 0.103585 | 0.169008 | 0.101358 | 0.120946 / 0.137624 | 0.121596 / 0.101804 |
| robust scale | 0.101539 | 0.175554 | 0.092673 | 0.113146 / 0.129156 | 0.111104 / 0.092152 |
| robust scale + local | 0.100697 | 0.170421 | 0.093188 | 0.112697 / 0.128935 | 0.109948 / 0.091101 |

稳健 direct 相对 legacy direct 降低了 all pixel AbsRel 和所有列出的分层 AbsRel，但 all
pixel MAE 从 0.169008 小幅升至 0.170421；因此不能声称它在每个训练指标上都占优。

### 8.2 Zero-shot transfer

Zero-shot 使用在原 SLABIM 数据上训练并冻结的 checkpoint，不读取 Area_1 train/val
梯度，也不进行 Area_1 特定的模型参数调整。分别评测：

1. frozen-DA3 refiner transfer；
2. partially trainable DA3 模型的既有 E2E checkpoint transfer（评测时仍为 eval mode）。

这里的 zero-shot 只修饰 source **模型权重**。主要 `robust_bim_direct` 比较器按第 8.1 节
使用 Area_1 train GT/semantic 选择过固定 cap，因此应准确称为“target-train-selected
non-learning baseline”，不能称为完全不接触目标域的 zero-shot baseline；它仍未读取
validation/test，且运行时不使用 GT/semantic。

因为 checkpoint 的训练数据与 Area_1 不同，命令必须显式给出 `--cross-dataset`；输出
provenance 会记录这一 opt-in。该参数只放宽数据集身份匹配，不放宽模型结构匹配。
开发阶段先在 validation 上运行两条 transfer 作为域差异诊断，不查看 test 输出；最终
配置和目标域 checkpoint 锁定后，再把 transfer test 与 fine-tune test 一起执行一次。

配置职责刻意分离：

- `configs/stanford_area1_transfer.yaml` 与 `configs/stanford_area1_transfer_e2e.yaml` 是
  **legacy source-transfer 兼容配置**。其模型尺度仍继承 source checkpoint 的 Q45 定义，避免
  通过改模型配置伪装为 checkpoint 内已有的新能力；评测端仍加载已冻结的 robust receipt，
  因而同一次输出会明确报告 legacy 与 robust 非学习比较器。
- `configs/stanford_area1.yaml` 与
  `configs/stanford_area1_e2e.yaml` 是 **target-domain robust wrapper**：将模型
  尺度固定为 train-only 选定的 `(q10=inf,q25=0.05)`，并锁定 robust residual anchor、
  frame-only routing、初始化/阶段训练策略和目标域损失权重；两者分别写入独立输出目录。
  Area_1 训练、checkpoint 接受和最终目标域评测只能使用这两个 wrapper。

因此 source frozen validation 输出中的 `coarse` 与 `legacy_global_scale_q45` 完全相同是
预期行为；`robust_global_scale`/`robust_bim_direct` 是独立、冻结且预注册的主要比较器，
不是对 source checkpoint 的事后改写。

### 8.3 Target-domain fine-tune

目标域训练只读取 30 个 train 房间，checkpoint 选择和早停只读取 7 个 validation
房间，7 个 test 房间不参与训练、超参数选择或 early stopping。评测两条训练路线：

- frozen DA3：复用缓存 DA3，只训练 BIM/RGB 深度 refiner；配置为最多 12 epoch、
  batch size 8、gradient accumulation 2、学习率 `4e-5`；
- E2E last-stage：在线执行 DA3，冻结 ViT backbone，只训练
  `head.scratch.refinenet1`、`output_conv1`、`output_conv2` 和 refiner；配置为最多
  8 epoch、batch size 4、gradient accumulation 2、refiner 学习率 `1e-5`、DA3
  学习率 `1e-6`。

目标模型的结构和初始化策略已锁定为：

- `residual_anchor_mode: robust_bim_direct`：学习残差乘在已冻结的 robust BIM-direct
  深度上，而不是较弱的 legacy Q45 coarse depth 上；
- `residual_routing_scope: frame_only`：深度感知 gate 只衰减全帧 residual，低分辨率和
  detail 局部 residual 保留近场修正容量；
- frozen target 从 source frozen checkpoint 初始化后，只把会直接产生乘性深度 residual
  的 `low/detail/frame` 输出切片清零，保留共享特征、方差和 BIM reliability/trust 辅助头；
- frozen target 的第 1 个 epoch 只训练 refiner output heads 与 BIM adapters，之后恢复全部原定
  非 DA3 refiner 参数；优化器参数集合在 warmup 前一次性建立，不在阶段切换时重建；
- E2E target 必须从 **target frozen `accepted.pt`** 开始，使用 `preserve` 保留第一阶段
  学到的 target residual heads，warmup 为 0。source E2E checkpoint 只作迁移诊断，明确
  禁止用作 target frozen 或 target E2E 初始化。

两条路线都将家具像素和 `bim_foreground_conflict` 像素的监督权重分别设为 2.0；若同一
像素同时满足两项，额外权重按加法组合，避免乘法放大。mask 由 train GT/semantic 构造，
只影响训练损失，不作为 RGB/BIM/DA3 推理输入，部署时不需要 semantic 或 GT。

第一阶段使用 `configs/stanford_area1.yaml`，从 SLABIM source frozen refiner 初始化，
因此必须使用 `--init-checkpoint` 和
`--allow-cross-dataset-initialization`。第二阶段 E2E 从已经接受的 Area_1 frozen 模型
初始化 refiner，使用 `configs/stanford_area1_e2e.yaml`，同时从 pinned HF
snapshot 加载 DA3 并只解冻上述 decoder 尾部；因为两阶段使用同一 Area_1 provenance，
不需要 cross-dataset opt-in。`--resume` 始终只允许恢复同一数据 provenance，不能借该
参数跨数据集恢复优化器。

正式模型使用 `accepted.pt`：frozen target 的 validation reference 是其
`robust_bim_direct` anchor；E2E target 的 reference 是基于同一在线 DA3 的
`live_robust_bim_direct`。refined 的 AbsRel 和 MAE 都优于相应 reference，且 near-range
AbsRel 不超过 reference 的 1.02 倍时才有资格写出。若训练没有生成 `accepted.pt`，应报告
“未超过稳健 BIM-direct 基线”，不能用 `best.pt` 冒充通过验收。

### 8.4 指标、聚合与语义子集

报告 AbsRel、RMSE、MAE、δ1、δ2、δ3，并同时给出：

- `pixel_micro`：所有有效像素合并；
- `frame_macro`：先按帧算指标再等权平均；
- `room_macro`：先按房间合并像素算指标再对房间等权平均。

除了 `all`，必须报告以下子集：

| 子集 | 精确定义 |
|---|---|
| `furniture` | semantic 为 table/chair/sofa/bookcase 且 GT 有效 |
| `non_structural` | 已知 semantic 且不属于 ceiling/floor/wall/beam/column/window/door |
| `bim_foreground_conflict` | GT 与 BIM 均有效，且 `GT < BIM - max(0.10 m, 0.05*BIM)` |
| `bim_consistent` | GT 与 BIM 均有效，且 `abs(GT-BIM) <= max(0.10 m, 0.05*BIM)` |
| `bim_no_hit` | GT 有效但固定围护 BIM 无命中 |

`furniture` 检查已标注家具，`bim_foreground_conflict` 则不依赖具体前景类别，更直接衡量
模型能否推翻位于家具后方的围护表面。评测脚本还以房间为配对单位，对 frozen 的
`refined - robust_bim_direct`、E2E 的 `refined - live_robust_bim_direct` 的 AbsRel/MAE 做
10,000 次 bootstrap，报告 95% 区间；差值为负表示学习模型更好。legacy reference 可以
附加报告，但主要判断只绑定对应的 robust reference。
`all`、`furniture` 和 `bim_foreground_conflict` 都必须给出该检验。

除独立的 `bim_envelope` 外，`raw_da3`、两个 legacy 方法、两个 robust 方法、`coarse`、
`refined` 以及 E2E live 方法必须在每个声明的 GT 子集上使用**完全相同的固定 support**。
任何方法在该 support 上产生非有限值或非正深度都直接报错，评测器还逐帧、逐房间和
汇总检查像素计数相同。`bim_envelope` 因为天然存在 no-hit，只单独报告
`GT valid & BIM hit` 上的指标以及 pixel/frame/room coverage，不能与全 support 方法的
误差数字直接横向比较。

### 8.5 Validation-first 与 test-once 生命周期

正式执行顺序固定为：

1. 生成并冻结完整 manifest 和 room-disjoint annotation；
2. 只在 validation 上运行两个 source transfer checkpoint；
3. 只用 train 更新参数，只用 validation 早停、选 checkpoint 和优化结构；
4. 锁定代码、配置、annotation/split fingerprint 及所有 checkpoint SHA-256；
5. validation 锁定最终主模型后，只对该主模型运行一次盲测；source-transfer 诊断和未
   晋级 challenger 不再读取 test，避免把 test 扩展成第二个模型选择集；
6. 不因 test 结果改结构、超参数、房间变换、support 或 checkpoint；若后续确需修改，
   必须声明为新实验并重新建立未触碰的测试协议。

这条规则既适用于学习方法，也适用于 zero-shot transfer 表。本次盲测完成前没有用
validation、配准质量或少量可视化帧冒充 test 精度；完成后也不再改动或重跑。

## 9. 公平性与论文报告规则

为使结论可复核，正式表格遵守以下约束：

1. 所有方法读取同一 annotation、同一 0.2–5.0 m GT mask、同一固定 support 和同一尺寸；
2. 所有需要 BIM 的方法读取同一个 Area_1 全局 core-envelope render；
3. semantic/conflict mask 不进入模型推理输入；它们只用于结果分层，并在 target-domain
   train 中按第 8.3 节改变监督权重；
4. zero-shot 不使用 Area_1 梯度；fine-tune 的 test 不参与模型选择，全部 test 只在锁定
   实验后执行一次；
5. 不按方法删除困难帧，不用 test GT 求尺度或位姿，不做逐帧 post-hoc ICP；
6. 同时报告 pixel、frame、room 三种聚合，避免大房间或高覆盖帧主导单一结论；
7. 主结论至少要求 frozen `refined` 相对 `robust_bim_direct`、E2E `refined` 相对
   `live_robust_bim_direct` 在各自 test `all` 上同时降低 AbsRel 和 MAE，并单独披露
   `furniture` 与 `bim_foreground_conflict`，不能只选择有利子集；
8. 公开 checkpoint SHA-256、resolved config、annotation hash、split fingerprint、
   训练来源 hash 和 cross-dataset opt-in 记录；
9. 明确披露第 5.3 节的 scan-calibrated/oracle-style“结构网格辅助标定”假设，不把本
   协议包装成无标定泛化；
10. RVT 不作为隐藏输入，完整 IFC 中被过滤的家具也不得以其他形式进入模型。

## 10. 复现命令

以下命令均从仓库根目录执行。

### 10.1 来源校验

```bash
.venv/bin/python scripts/data/verify_stanford_bimsyn_sources.py \
  --area-root ../Stanford2D3DS/no_xyz \
  --area-tar ../Stanford2D3DS/no_xyz/area_1_no_xyz.tar \
  --ifc-root ../BIMSyn/BIM_model/ifc \
  --rvt-root ../BIMSyn/BIM_model/rvt \
  --output data/provenance/stanford_area1_sources.local.json
```

### 10.2 固定房间配准

```bash
.venv/bin/python scripts/data/register_stanford_bimsyn.py \
  --semantic-obj ../Stanford2D3DS/no_xyz/area_1/3d/semantic.obj \
  --ifc-dir ../BIMSyn/BIM_model/ifc \
  --output data/provenance/stanford_area1_alignment.local.json \
  --seed 20260810 \
  --sample-points 8000 \
  --coarse-points 1500 \
  --yaw-starts 36 \
  --refine-candidates 4 \
  --max-iterations 25
```

正式配置已经固定仓库内审计通过的 alignment。重建时使用新的 `.local.json`，再在
manifest 生成前锁定 preparation-only child config；不要覆盖正式回执。

```bash
.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1_transfer.yaml \
  --alignment-receipt data/provenance/stanford_area1_alignment.local.json \
  --preparation-only \
  --output configs/local/stanford_area1_prepare.yaml
```

### 10.3 DA3 缓存、样本与 annotation

全新生成 schema 2 cache 时运行：

```bash
.venv/bin/python scripts/data/cache_stanford_da3.py \
  --config configs/local/stanford_area1_prepare.yaml
```

cache 全部通过 schema 2 校验后再制备样本：

```bash
.venv/bin/python scripts/data/prepare_stanford_area1.py \
  --config configs/local/stanford_area1_prepare.yaml \
  --overwrite

.venv/bin/python scripts/data/build_stanford_room_split.py \
  --manifest data/processed/stanford_area1_504/manifest.jsonl \
  --output data/annotations/stanford_area1_room.local.jsonl \
  --receipt data/annotations/stanford_area1_room.local.receipt.json \
  --train-rooms 30 --val-rooms 7 --test-rooms 7 \
  --seed 42 --search-trials 20000

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --output configs/local/stanford_area1_selector.yaml
```

使用仓库内冻结 alignment 时，可跳过 preparation-only 命令，并把 cache/prepare
的 `--config` 改为 `configs/stanford_area1_transfer.yaml`。

annotation 生成后，把回执中的 `annotation_raw_sha256` 和
`split_fingerprint_sha256` 固定到正式配置和实验记录。随后只在完整 train split 上选择
稳健尺度 cap：

```bash
.venv/bin/python scripts/data/select_stanford_scale_caps.py \
  --config configs/local/stanford_area1_selector.yaml \
  --output data/provenance/stanford_area1_scale.local.json \
  --workers 8 --log-every 100
```

选择器默认拒绝覆盖已有回执，并对 annotation 做穷举完整性检查；不得传入 validation/test
选择或抽样参数。随后再次运行 `materialize_runtime_config.py --scale-receipt ...`，为 fresh
run 生成 transfer/target child config。仓库中的正式 config/receipt 只绑定已发布历史结果，
不得用新 fingerprint 覆盖。此时仍不读取 test 结果。

```bash
.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1_transfer.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --scale-receipt data/provenance/stanford_area1_scale.local.json \
  --output configs/local/stanford_area1_transfer.yaml

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1_transfer_e2e.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --scale-receipt data/provenance/stanford_area1_scale.local.json \
  --output configs/local/stanford_area1_transfer_e2e.yaml

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --scale-receipt data/provenance/stanford_area1_scale.local.json \
  --experiment-output-dir outputs/stanford_area1_local \
  --output configs/local/stanford_area1.yaml

.venv/bin/python scripts/data/materialize_runtime_config.py \
  --base-config configs/stanford_area1_e2e.yaml \
  --annotation data/annotations/stanford_area1_room.local.jsonl \
  --scale-receipt data/provenance/stanford_area1_scale.local.json \
  --experiment-output-dir outputs/stanford_area1_local_e2e \
  --output configs/local/stanford_area1_e2e.yaml
```

以下命令使用这四个 fresh-data child config。若只是核对仓库内已发布结果，
则改用同名的 `configs/stanford_area1*.yaml` 和 `outputs/stanford_area1*` 历史路径。

### 10.4 Validation-only transfer 诊断

```bash
.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/local/stanford_area1_transfer.yaml \
  --checkpoint outputs/slabim/accepted.pt \
  --split val --cross-dataset \
  --batch-size 8 \
  --output outputs/stanford_area1_local_transfer/frozen_val

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/local/stanford_area1_transfer_e2e.yaml \
  --checkpoint outputs/slabim_e2e/accepted.pt \
  --split val --cross-dataset \
  --batch-size 4 \
  --output outputs/stanford_area1_local_transfer/e2e_val
```

两条 validation-only transfer 均已完整落盘，正式结果见第 11.2–11.3 节；两者都未读取
test。source E2E 结果只作域差异诊断，不改变第 8.3 节的 target 初始化链。

### 10.5 目标域训练与 validation 选择

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/local/stanford_area1.yaml \
  --init-checkpoint outputs/slabim/accepted.pt \
  --allow-cross-dataset-initialization

.venv/bin/python scripts/model/train.py \
  --config configs/local/stanford_area1_e2e.yaml \
  --init-checkpoint outputs/stanford_area1_local/accepted.pt

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/local/stanford_area1.yaml \
  --checkpoint outputs/stanford_area1_local/accepted.pt \
  --split val --output outputs/stanford_area1_local/formal_val \
  --device cuda --batch-size 8 --log-every 100 \
  --inference-seed 42 --bootstrap-repetitions 10000 --bootstrap-seed 42

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/local/stanford_area1_e2e.yaml \
  --checkpoint outputs/stanford_area1_local_e2e/accepted.pt \
  --split val --output outputs/stanford_area1_local_e2e/formal_val \
  --device cuda --batch-size 4 --log-every 100 \
  --inference-seed 42 --bootstrap-repetitions 10000 --bootstrap-seed 42
```

只有 validation 协议通过后，才锁定代码、两份 legacy transfer 配置、两份 robust target
wrapper、annotation/split fingerprint 与四个待比较 checkpoint 的 SHA-256。若任一训练
路线没有 `accepted.pt`，该路线不进入伪装成“通过”的最终模型列，而应如实记录失败。

### 10.6 锁定后的单次最终测试

validation 选择结果将 frozen refiner 锁定为主模型，E2E/source transfer 均未晋级，故只
执行以下一次 test 命令：

```bash
.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/local/stanford_area1.yaml \
  --checkpoint outputs/stanford_area1_local/accepted.pt \
  --split test --output outputs/stanford_area1_local/formal_test \
  --device cuda --batch-size 8 --log-every 100 \
  --inference-seed 42 --bootstrap-repetitions 10000 --bootstrap-seed 42
```

该命令输出 `summary.json` 和 `per_frame.csv`；结果表只从这两个文件及其内置 provenance
生成。命令只执行一次，输出后不再调参或评测其他 checkpoint。

## 11. 当前已验证结果

### 11.1 数据、配准与 split 审计

| 项目 | 结果 |
|---|---:|
| regular-view 配对帧 | 10,327 |
| 房间 / camera UUID | 44 / 186 |
| IFC / RVT | 44 / 44，stem 一一对应 |
| 固定房间配准验收 | 44/44 accepted，0 failed |
| train | 7,013 帧 / 30 房间 / 125 cameras |
| validation | 1,673 帧 / 7 房间 / 30 cameras |
| test | 1,641 帧 / 7 房间 / 31 cameras；一次盲测已完成 |

### 11.2 Source frozen zero-shot validation（已落盘）

唯一读取的是 validation split。结果来源为
[`zero_shot_frozen_val_summary.json`](../results/stanford_area1/zero_shot_frozen_val_summary.json)，
文件 SHA-256 为 `67c47e0f2fbe4986bd0ef435fc562dd63bed2b0877c4d14a9eecdcb5624aeab8`；
评测器 SHA-256 为 `a1ad7cb66d5e206f0bf03172db24063f620a5d139cf3df9a54313ca5b4e81259`。
source frozen checkpoint SHA-256 为
`a0e339fe96652e30d9a685bce2ae5c9197005b2508379176cef3d47d76d9b719`。
共评测 1,673 帧、7 个 validation 房间，所有可比方法使用同一固定 GT support。

下表的前五个误差列为 `pixel_micro`，最后一列为 all `room_macro` AbsRel；MAE 单位为米。

| 方法 | all AbsRel | all MAE | furniture AbsRel | non-structural AbsRel | conflict AbsRel | all room AbsRel |
|---|---:|---:|---:|---:|---:|---:|
| raw DA3 | 0.277104 | 0.628398 | 0.293484 | 0.293218 | 0.282429 | 0.284187 |
| legacy Q45 scale | 0.090152 | 0.176043 | **0.109113** | 0.118490 | 0.143220 | 0.088924 |
| legacy Q45 + local | 0.091201 | 0.173029 | 0.111962 | 0.119929 | 0.149590 | 0.090138 |
| robust scale | **0.086442** | 0.174225 | 0.109789 | **0.111819** | **0.124093** | **0.084485** |
| robust scale + local（主要 baseline） | 0.087100 | **0.171213** | 0.111830 | 0.112569 | 0.126972 | 0.085536 |
| source checkpoint `coarse` | 0.090152 | 0.176043 | **0.109113** | 0.118490 | 0.143220 | 0.088924 |
| source checkpoint `refined` | 0.166899 | 0.375176 | 0.149817 | 0.164612 | 0.167525 | 0.161186 |

粗体只标识本表同列的最小值，不构成 test 结论。纯 `bim_envelope` 不在固定 support 表内：
其 validation pixel coverage 为 0.917843，在 `GT valid & BIM hit` 上 pixel AbsRel/MAE 为
0.196220/0.299523；该数字不能与上表直接排序。

以房间为配对单位、10,000 次 bootstrap 的 `refined - robust_bim_direct` 结果为：

| 子集 / 指标 | room mean difference | 95% CI | refined 更好房间比例 |
|---|---:|---:|---:|
| all AbsRel | +0.075650 | [0.068665, 0.083292] | 0/7 |
| all MAE | +0.177055 m | [0.152867, 0.204725] m | 0/7 |
| furniture AbsRel | +0.052072 | [0.036453, 0.070871] | 0/7 |
| furniture MAE | +0.157432 m | [0.086829, 0.265350] m | 0/7 |
| conflict AbsRel | +0.046185 | [0.034478, 0.059672] | 0/7 |
| conflict MAE | +0.081687 m | [0.064013, 0.106131] m | 0/7 |

差值为正表示 source `refined` 更差，且这些区间都不跨 0。机器判断在 all/furniture/conflict
的 pixel/frame/room 三种聚合上均为 `false`。这说明 source frozen refiner 存在明显域偏移；
它没有超过 robust BIM-direct，不能被表述为学习方法已经优于直接矫正。另一方面，train-only
稳健尺度在 validation 的 all/conflict 等关键 AbsRel 上优于 legacy Q45，支持把它升级为
目标域训练必须超过的主要 baseline。

### 11.3 Source E2E zero-shot validation（已落盘）

结果来源为
[`zero_shot_e2e_val_summary.json`](../results/stanford_area1/zero_shot_e2e_val_summary.json)，
summary SHA-256 为 `8d3a12f110ecbb5630c42a068887cd4a69b18e3d01616f0891c29356069a31b7`，
对应 `per_frame.csv` SHA-256 为
`6b5f7d279f1c5c759c7b4c29818cf4847a52e9de744f86a3aec264e6de7501ed`，评测器 SHA-256
为 `14de577cd650586adf5a77c40e592fd0b4deedff9f7a9ddbdce48ba479b098a1`。source E2E
checkpoint SHA-256 为 `5ec9d25f58d16ff9e790c9a5763fd25a816392d3025c3d5c381baf6e9501e583`。

评测覆盖与第 11.2 节相同的 1,673 帧、7 个 validation 房间；runtime 为 batch size 4、
8 workers、inference seed 42、deterministic algorithms `true`，checkpoint 训练 provenance
也记录 deterministic algorithms `true`。checkpoint 数据集与 Area_1 不同，验证状态为
`accepted_cross_dataset`，并记录显式 `--cross-dataset` opt-in；这表示有意接受跨数据集
诊断，不表示两个数据 provenance 相同。train-only robust selection receipt 验证状态为
`verified` 且具备正式协议资格，validation/test opened count 均为 0。

E2E 的主要公平参考是 `live_robust_bim_direct`：先在该 checkpoint **在线 DA3** 输出上应用
冻结 robust cap，再使用同一固定局部 BIM 矫正。下表的 AbsRel/MAE 为 pixel micro，最后
一列为 room-macro AbsRel；MAE 单位为米。

| 子集 | 方法 | pixel AbsRel | pixel MAE | room AbsRel |
|---|---|---:|---:|---:|
| all | live DA3 | 0.282418 | 0.628314 | 0.293188 |
| all | live robust BIM-direct | **0.087064** | **0.170111** | **0.085401** |
| all | source E2E refined | 0.164560 | 0.366513 | 0.159015 |
| furniture | live robust BIM-direct | **0.113065** | **0.206516** | **0.100787** |
| furniture | source E2E refined | 0.151541 | 0.295815 | 0.147898 |
| conflict | live robust BIM-direct | **0.127785** | **0.177956** | **0.116398** |
| conflict | source E2E refined | 0.169481 | 0.253629 | 0.163112 |

以房间为配对单位、10,000 次 bootstrap 的 `refined - live_robust_bim_direct` 为：

| 子集 / 指标 | room mean difference | 95% CI | refined 更好房间比例 |
|---|---:|---:|---:|
| all AbsRel | +0.073614 | [0.066148, 0.081830] | 0/7 |
| all MAE | +0.170164 m | [0.145952, 0.196896] m | 0/7 |
| furniture AbsRel | +0.047111 | [0.033087, 0.062296] | 0/7 |
| furniture MAE | +0.132625 m | [0.079355, 0.216261] m | 0/7 |
| conflict AbsRel | +0.046714 | [0.035341, 0.059247] | 0/7 |
| conflict MAE | +0.080165 m | [0.062852, 0.102373] m | 0/7 |

所有差值均为正、区间均不跨 0；all/furniture/conflict 的 pixel/frame/room 机器判断全部为
`false`。source E2E refined 虽比 source frozen refined 的 all pixel AbsRel/MAE 略低，
仍远差于相同 live DA3 上的 robust BIM-direct，且 7/7 房间都失败。因此 source E2E
checkpoint 不进入 target 初始化链：target E2E 只能从 target frozen validation 验收后的
`accepted.pt` 开始。

### 11.4 Target frozen-DA3 训练与正式 validation

第一阶段完成全部 12 个 epoch。按 validation pixel AbsRel 选出的最佳 checkpoint 是人类
计数第 11 轮（history `epoch=10`），训练器记录的指标为 AbsRel 0.070051、MAE
0.139692、RMSE 0.363003、δ1 0.929543、near AbsRel 0.117687。全程只有一次由
GradScaler 检测并自动回退的 AMP optimizer skip，后续三轮恢复为完整 step，所有 loss、
预测指标和固定 support count 均为有限值。

不可变训练产物为：

- `outputs/stanford_area1/accepted.pt`，SHA-256
  `b651c1908fd3ebc9415471c3e1d96bada0b9e22b69602fdde29b138345fc00f4`；公开资产角色与
  下载状态见 [`results/manifest.json`](../results/manifest.json)；
- [`frozen_history.json`](../results/stanford_area1/frozen_history.json) SHA-256
  `e76aec318264e9f81daa17ef845f8e1f745939ac540f3bafa81d0d0cd87af3fd`；
- [`frozen_run_state.json`](../results/stanford_area1/frozen_run_state.json) SHA-256
  `b47401dc7a76854666209e799d068836632db3bf56645e3fe5028563b5cf9036`。

正式 validation 结果来自
[`frozen_val_summary.json`](../results/stanford_area1/frozen_val_summary.json)，
SHA-256 `1e79bc938a6828fd763735ec051b7c10762c3350af74035ccdffda0e43ceb872`；
逐帧 CSV SHA-256 为
`1713dd8348bad82b632c19fed20d20512e3a955d7b50a8feb5326cfc1994148a`。
下表为 pixel micro；MAE/RMSE 单位为米。

| 子集 | 方法 | AbsRel | MAE | RMSE |
|---|---|---:|---:|---:|
| all | robust BIM-direct | 0.087100 | 0.171213 | 0.394554 |
| all | **target frozen refined** | **0.070050** | **0.139691** | **0.363002** |
| furniture | robust BIM-direct | 0.111830 | 0.204509 | 0.397328 |
| furniture | **target frozen refined** | **0.099507** | **0.185959** | **0.368923** |
| conflict | robust BIM-direct | 0.126972 | 0.176345 | 0.324595 |
| conflict | **target frozen refined** | **0.115958** | **0.163215** | **0.312474** |

评测器生成的 all/furniture/conflict × pixel/frame/room 九个
`learned_beats_primary_bim_direct_absrel_and_mae` 判断全部为 `true`。10,000 次按房间配对
bootstrap 的 AbsRel/MAE 95% 区间分别为：all
`[-0.031064,-0.015825]`/`[-0.051196,-0.025232] m`，furniture
`[-0.015981,-0.007274]`/`[-0.047971,-0.015166] m`，conflict
`[-0.018732,-0.005235]`/`[-0.022784,-0.008202] m`，均完全小于 0。因此 frozen
checkpoint 满足“学习方法优于直接 BIM 矫正”的 validation 晋级门槛。

### 11.5 Target E2E challenger：完成但未晋级

第二阶段从上面的 target frozen `accepted.pt` 初始化并保留 residual heads；在线 DA3 只
解冻 decoder 尾部。训练在 5 个 epoch 后触发 patience=3 的 early stopping，最佳是人类
计数第 2 轮（history `epoch=1`）：AbsRel 0.070186、MAE 0.140187、RMSE 0.364560、near
AbsRel 0.116911。它仍明显优于 cached/live robust BIM-direct，但比 frozen validation
主模型的 AbsRel/MAE 分别差约 0.19%/0.35%，所以不替换主模型。

E2E 审计产物为：

- 未晋级的历史 `accepted.pt` SHA-256 为
  `c60f4375df354066fa82e7c7e67cee821dbfb73341d883184228c2cad429d5a6`；权重已从清理后的
  `outputs/` 删除，不作为公开生产模型，角色记录于 `results/manifest.json`；
- [`e2e_challenger_history.json`](../results/stanford_area1/e2e_challenger_history.json)
  SHA-256
  `dd64ad786dfeae3224777a6d2ce8d4402865d91a75083152bb950cd3b9dc9847`；
- [`e2e_challenger_run_state.json`](../results/stanford_area1/e2e_challenger_run_state.json)
  SHA-256
  `679ba65f6553ed8d0430b6b90225b160979d452d227e5954d4d8673ef69419f2`；
- [`formal_val/summary.json`](../results/stanford_area1/e2e_challenger_val_summary.json)
  SHA-256 `8760626f8cb4db7b47f3936e37c6b1deb97faa72f7ef405d20a6e0ad130c0f60`；
  逐帧 CSV 属过程产物，公开清理时未保留。

正式 validation pixel-micro 对比如下：

| 子集 | cached robust direct | live robust direct | E2E refined |
|---|---:|---:|---:|
| all AbsRel / MAE | 0.087100 / 0.171213 | 0.087218 / 0.171611 | 0.070185 / 0.140186 |
| furniture AbsRel / MAE | 0.111830 / 0.204509 | 0.112238 / 0.205946 | 0.099688 / 0.187284 |
| conflict AbsRel / MAE | 0.126972 / 0.176345 | 0.127167 / 0.177215 | 0.116477 / 0.164755 |

E2E 对两种 direct anchor 的九格机器判断全部通过，但“胜 direct”不是取代更强 frozen
checkpoint 的充分条件。模型选择在查看 test 前已经固定为 frozen；因此 E2E 不读取 test。

### 11.6 最终一次盲测：target frozen 主模型

最终 test 只运行第 10.6 节的一条命令。结果来自
[`frozen_test_summary.json`](../results/stanford_area1/frozen_test_summary.json)，
SHA-256 `a9a69ecf3b6d032164e7c869399e7fedfefc488c84a3aa18b9a95d09abb963ab`；
[`frozen_test_per_frame.csv`](../results/stanford_area1/frozen_test_per_frame.csv) SHA-256 为
`c42a69af69fe5eec6d4a2923fdbb1df77a16770a6500435c735fbd69d9b4e6e8`。评测器 SHA-256
为 `14de577cd650586adf5a77c40e592fd0b4deedff9f7a9ddbdce48ba479b098a1`。

审计确认：split=`test`，1,641 个唯一且有序的 annotation ID，7 个房间、31 个
`camera_uuid`；checkpoint dataset match=`true`、cross-dataset=`false`、模型配置 override
为空；稳健尺度回执状态为 `verified`、`formal_protocol_eligible=true`，并记录 cap 选择时
validation/test 打开样本数均为 0。all/furniture/conflict 的有效像素数分别为
404,367,334 / 59,841,152 / 115,362,205；六个声明子集、七个可比方法逐帧 support mismatch
均为 0。

all pixel-micro 主表为：

| 方法 | AbsRel | MAE | RMSE | δ1 |
|---|---:|---:|---:|---:|
| raw DA3 | 0.301228 | 0.673234 | 0.834852 | 0.264374 |
| legacy Q45 global scale | 0.094718 | 0.163512 | 0.353932 | 0.913999 |
| legacy Q45 BIM-direct | 0.095672 | 0.163103 | 0.353522 | 0.913417 |
| robust global scale | 0.077521 | 0.139378 | 0.315410 | 0.937095 |
| robust BIM-direct（主要 baseline） | 0.078146 | 0.138907 | 0.313498 | 0.937405 |
| **target frozen refined** | **0.067924** | **0.117484** | **0.304189** | **0.940424** |

相对主要 robust BIM-direct，学习模型的 all pixel AbsRel/MAE/RMSE 分别降低
13.08%/15.42%/2.97%，δ1 提高 0.302 个百分点；相对 legacy BIM-direct，AbsRel 降低
29.00%。即使 test 上 robust global scale 略优于其 local-direct 版本，refined 仍优于
两者。纯 BIM envelope 的 pixel coverage 为 0.948792；它只在 hit support 上评测，AbsRel
0.260729，不能与固定全 support 的上表直接排序。

下表完整列出核心子集和三种聚合。`通过` 要求 refined 的 AbsRel、MAE 同时低于 robust
BIM-direct；MAE 单位为米。

| 子集 | 聚合 | robust direct AbsRel / MAE | refined AbsRel / MAE | 机器判断 |
|---|---|---:|---:|---:|
| all | pixel micro | 0.078146 / 0.138907 | **0.067924 / 0.117484** | `true` |
| all | frame macro | 0.078809 / 0.140639 | **0.068773 / 0.119316** | `true` |
| all | room macro | 0.091003 / 0.166845 | **0.078948 / 0.141219** | `true` |
| furniture | pixel micro | 0.089286 / 0.151608 | **0.085884 / 0.146141** | `true` |
| furniture | frame macro | 0.093287 / 0.188137 | **0.090250 / 0.183242** | `true` |
| furniture | room macro | 0.086848 / 0.142063 | **0.082817 / 0.135784** | `true` |
| conflict | pixel micro | **0.140639** / 0.179008 | 0.141725 / **0.177721** | `false` |
| conflict | frame macro | 0.117081 / 0.165091 | **0.116221 / 0.161897** | `true` |
| conflict | room macro | **0.171254 / 0.207006** | 0.177144 / 0.210833 | `false` |

10,000 次按房间配对 bootstrap 的结果为：

| 子集 / 指标 | room mean difference | 95% CI | refined 更好房间比例 |
|---|---:|---:|---:|
| all AbsRel | -0.012055 | [-0.017295, -0.006285] | 6/7 |
| all MAE | -0.025625 m | [-0.034186, -0.015103] m | 6/7 |
| furniture AbsRel | -0.004031 | [-0.006578, -0.001581] | 5/6 |
| furniture MAE | -0.006279 m | [-0.008893, -0.003920] m | 6/6 |
| conflict AbsRel | +0.005891 | [-0.006884, +0.022509] | 3/7 |
| conflict MAE | +0.003827 m | [-0.008644, +0.022054] m | 4/7 |

all 和 furniture 的区间完全小于 0，支持稳定改善；conflict 两个区间都跨 0，不能声称
在未见房间上稳定优于 direct。room audit 显示 `hallway_8` 是主要 conflict 退化来源，
`conferenceRoom_1` 的 all 指标也有轻微退化；这两个事实保留为局限，不在 test 后修补。

### 11.7 最终结论与实验边界

| 路线 | validation | test | 最终角色 |
|---|---|---|---|
| source frozen transfer | 完成，明显失败 | 未运行 | 域差异诊断 |
| source E2E transfer | 完成，明显失败 | 未运行 | 域差异诊断 |
| target frozen robust refiner | 通过全部九格门槛 | 一次盲测完成 | **最终主模型** |
| target E2E last-stage | 胜 direct、略差于 frozen | 未运行 | 未晋级 challenger |

最终结果满足最初的核心要求：在完整 test 固定 support 上，学习模型显著优于 robust 和
legacy 两种 BIM-direct 的总体 AbsRel/MAE，并在包含家具的子集上跨 pixel/frame/room
三种聚合保持优势。更窄的 `bim_foreground_conflict` 子集没有获得同等稳定性，因此论文
只能主张“总体和家具深度改善”，不能主张“所有遮挡冲突区域均改善”。

本结论还受第 5.3 节 scan-calibrated/oracle-style 房间标定假设约束：当前 BIM-to-Area
变换借助了 Area_1 semantic 结构网格。实际部署必须由测量控制点、SLAM/测绘配准或既有
BIM 坐标链提供同等固定变换；未解决该外部标定问题前，不能把本实验描述为完全无目标域
几何校准的跨建筑泛化。
