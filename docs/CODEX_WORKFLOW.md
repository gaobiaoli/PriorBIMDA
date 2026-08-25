# Codex 对话要点与项目演化

> 给接手的新 chatbot / 研究者：先读本页、`README.md`、`docs/USER_GUIDE.md`、
> `docs/UNIVERSAL_SCALE_PROTOCOL.md` 与 `results/metrics.json`。不要恢复历史 V1–V6、region-CV
> 或 BIM-direct 网络锚点。

## 1. 当前唯一活动方法

项目用固定 BIM 先验细化 DA3 Metric Large 的单帧公制深度，支持 SLABIM 与
2D-3D-S Area_1 + BIMSyn。两个数据集运行同一个尺度估计器：

```text
log(s) = min(Q45(log(BIM/DA3)), Q25(log(BIM/DA3)) + 0.05)
D_scaled = s * D_DA3
D_pred = D_scaled * exp(clamp(r_frame + r_low + r_detail))
```

网络同时读取 RGB、DA3 geometry/confidence 与 BIM depth/valid/normal/edge；学习 frame、low、
detail 三个有界 log-residual。乘法锚点始终是 `D_scaled`，不是 BIM-direct。BIM-direct 是所有
数据集共享的确定性强比较器，也是训练 acceptance 的门槛。推理不读取 GT、语义或家具 mask。

尺度参数最初只用 Area_1 train rooms 选择，随后冻结并原样应用于 SLABIM；SLABIM validation
没有用于重新调参。机器 receipt SHA 为
`361fc2c97ffca9dde1a3a1b6b97fcd1d894003809a01935205f16cb4043c3ba1`。

## 2. 最终结果（0.2–5.0 m，pixel-micro）

| 数据 / 方法 | AbsRel | MAE | RMSE | δ1 |
|---|---:|---:|---:|---:|
| SLABIM raw DA3 | 0.19935 | 0.31109 | 0.42167 | 0.76328 |
| SLABIM universal BIM-direct | 0.06263 | 0.10334 | 0.24761 | 0.96320 |
| SLABIM learned | **0.05601** | **0.09210** | **0.22725** | **0.97759** |
| Area_1 raw DA3 | 0.30123 | 0.67323 | 0.83485 | 0.26437 |
| Area_1 universal BIM-direct | 0.07815 | 0.13891 | 0.31350 | 0.93740 |
| Area_1 learned | **0.06689** | **0.11761** | **0.30823** | **0.94295** |

学习模型相对 direct 的 AbsRel 改善为 10.56% 与 14.41%。Area_1 test 的 all/furniture/conflict
在 pixel/frame/room 三种点估计聚合上都同时改善 AbsRel 与 MAE；但 conflict AbsRel 的
room-bootstrap 95% CI 为 `[-0.00692, 0.00064]`，跨 0，不能写成显著优势。训练均为单 seed。

SLABIM 108 帧三维融合评测同样支持 learned：Chamfer-L1 `0.09109 m`（direct
`0.10515 m`），F-score@10 cm `0.79622`（direct `0.74003`）。

最近完成但未晋升为公开主模型的 Area_1 hybrid 候选，把冻结 DA3 encoder 第 11/23 层 feature
只送入 low/detail refiner，尺度头使用更好的无 token 版本；随后并联一个受限米制加法头，
从头按 3/9/3/3 四阶段训练。validation-selected checkpoint 的 val/test AbsRel 为
`0.06131/0.06235`，上一版同时向尺度/refiner 融合 feature 的候选为 `0.06209/0.06244`。
加法头相对同检查点比例分支只改善 `0.007%/0.119%`，不构成实质收益；新候选相对旧 feature
模型的 MAE/RMSE 也退化。因此当前研究候选已回退到无加法的 DA3-feature checkpoint；hybrid
只保留为诊断。test 早已揭盲，结果不能改写上方公开主表。

正式文件：

- `results/slabim/test_summary.json`
- `results/stanford_area1/val_summary.json`
- `results/stanford_area1/test_summary.json`
- `results/manifest.json`

## 3. 对话驱动的关键演化

1. **复核原项目和位姿**：检查 sensor_data、相机/LiDAR/BIM 变换、时间戳、camera-z 深度，
   排除由坐标方向错误造成的虚假精度。
2. **否定 V1/BIM 加权融合**：旧方案只在几个候选间分配权重，不能保证纠正系统误差；V1–V3
   候选、旧信任网络和版本配置随后退出活动代码。
3. **建立 coarse-to-fine 残差网络**：先恢复尺度，再学习 frame/low/detail 残差；可靠性和
   uncertainty 是辅助监督，不作为硬输出门控。
4. **修复 SLABIM 数据协议**：`ignore.txt` 的 90 帧在引入时排除，另隔离 13 个共享
   fused-LiDAR 帧；811 条 exhaustive annotation 得到活动 `496/104/108`。
5. **Area_1 + BIMSyn 适配**：10,327 帧、44 rooms；IFC 过滤家具/proxy/MEP，只保留固定
   envelope/core prior；按 room/camera UUID 做 `7013/1673/1641` 划分。
6. **公平评测修复**：所有方法共享固定 GT support；无效预测 fail-fast；E2E 的比较器必须基于
   live DA3；validation RNG 独立重置；checkpoint 严格绑定数据与配置 provenance。
7. **统一尺度方法**：发现 SLABIM q45 与 Area_1 robust-cap 不一致后，删除活动 q45 模型路径与
   Area_1 BIM-direct 网络锚点。两个数据集均重新训练和完整评测。
8. **公开项目精简**：脚本拆分为 data/model/pipelines/analysis；下载器固定 revision/hash；
   删除 region-CV、旧消融、未晋级 E2E 权重和过程 checkpoint，只保留两份统一主模型。
9. **pano 路线收敛**：额外 ERP tangent 能补覆盖，但 strict single+tangent 在同一
   regular support 上显著退化。因此主协议回到数据集原始 regular DA3，按
   pose 投到 ERP 做同站稳健联合，再回投原 regular 评测。BIM-scale validation
   AbsRel `0.08644→0.07595`；细节和负结果见 `docs/PANO_DEPTH_EVALUATION.md`
   第 14 节。
10. **纠正 regular 语义作用位置**：第一版 oracle 在 all-hit scale 后按语义类别局部替换 BIM，
    validation `0.08644→0.08328`，但家具逐位不变、仅 5/7 room 改善且 CI 跨 0。用户指出
    语义应决定全局尺度样本，而不是只作为输出像素门。修订版仅用 image/BIM 同类结构比例
    估计每帧一个尺度，再乘到全图；q65 只由 7,013 个 train 样本/30 rooms 选择。冻结到
    validation 后 AbsRel `0.08644→0.07725`（-10.63%），MAE `0.17422→0.15464`，7/7
    room 改善且 room-bootstrap CI `[-0.01404,-0.00273]`。家具也随同一尺度改善到
    `0.10097`。它仍使用官方 semantic GT，只是 privileged 上限，不是 deployable/test claim。
11. **引入冻结 DA3 encoder feature**：固定 revision 和 504 preprocessing，一次性缓存 10,327
    帧的 layer-11/layer-23 FP16 token；中层特征进入尺度 attention 与 refiner 1/4 层，深层特征
    进入尺度全局条件和 refiner bottleneck。DA3 不进 optimizer，任务网络从头训练。最终候选
    test AbsRel `0.06244`，但 scale-only 指标退化；完整协议与失败边界记录在
    `docs/ATTENTIVE_SCALE_EXPERIMENT.md`。
12. **检查尺度后 residual 的真实分布**：在 Area_1 validation 的 1,673 帧中每帧确定性采样
    4,096 个固定-support 像素。无 DA3-feature attention scale 的误差—距离幂指数为 `0.404`，
    房间均值 bootstrap CI `[0.142,0.719]`；`mean |error| ≈ 0.0566 m + 0.0402 depth`
    比纯加法和纯比例都更贴合。BIM-consistent 指数 `0.328`，furniture/no-hit 为
    `0.762/1.061`。因此后续应采用带米制误差底噪的异方差/双专家 residual，不能仅按
    low/detail 名称硬编码残差类型；test 未访问。
13. **训练比例+加法 hybrid residual**：按验证诊断实现
    `D_final=D_scale*exp(r_prop)+Delta_D_add`；尺度头不使用 DA3 token，refiner 使用冻结
    layer-11/23 token，加法头零初始化并限幅 `+/-0.20 m`。任务网络从头按 3/9/3/3 阶段训练，
    human epoch 17 仅由 validation 选中。正式相同输出边界评测显示，加法分支的 val/test
    AbsRel 增益只有 `0.0000045/0.0000745`，整体 final 为 `0.06131/0.06235`。因此保留为
    可复现的负/混合诊断：整体 recipe 略进步，不能把改进归因于 additive head。
14. **回退无加法版本并逐级归因**：以此前冻结 DA3-feature checkpoint 为活动研究候选，
    同一前向顺序评测 `scale`、`scale+r_low`、`scale+r_low+r_detail`。validation AbsRel 为
    `0.06960→0.06217→0.06209`，test 为 `0.07144→0.06285→0.06244`；low 分别贡献总改善的
    `98.98%/95.44%`，detail 只贡献 `1.02%/4.56%`。因此保留 detail 作为小补充，不再使用
    additive head。
15. **纠正全图诊断与全深度训练的混淆**：旧 `--depth-support all-valid` 只改变 evaluator，
    checkpoint 仍由 0.2--5.0 m loss 训练。随后新增 `official_all_valid` dataset 模式，从官方
    regular uint16/512 深度动态重载 GT，仅排除 `0`/`65535`，并把 128 m 输出上限与 GT
    support 解耦。从头按同一 3/9/3 schedule 训练后，全深度 validation/test AbsRel 为
    `0.06861/0.06741`；相对旧 hit-only checkpoint 在相同 support 上的 `0.07214/0.06744`，
    validation 明显改善而 test AbsRel 基本持平，不能夸大为跨房间泛化突破。

## 4. 不可破坏的实验约束

1. 同一表内所有可比预测必须使用完全相同的 `gt_valid`；既有主表为 `0.2–5.0 m`，全深度
   诊断必须明确标记 `official_all_valid`，两种 support 不能混表或按方法删无效像素。
2. estimator 不允许数据集覆盖；修改 `0.05` 等参数必须升级协议并重跑两域。
3. test 只在参数/checkpoint 由 train/val 冻结后执行；已有 test 结果不能反向用于调参。
4. SLABIM 保证 fused-LiDAR 跨 split 隔离；Area_1 保证 room/camera UUID 隔离。
5. furniture/conflict/semantic 只用于训练权重和评测，不进入推理输入。
6. Area_1 BIM→Area 变换固定，不做逐帧 ICP；当前 semantic-mesh 辅助配准必须披露为
   scan-calibrated/oracle-style。
7. source→target 初始化必须显式 opt-in；resume 永远要求同数据、同模型配置。
8. DA3 revision 固定为 `4010e39f3634a45bc60553321fb49fb760bd594e`。

## 5. 当前关键文件

- 配置：`configs/slabim_base.yaml`、`slabim_pretrain.yaml`、`slabim.yaml`、
  `stanford_area1_transfer.yaml`、`stanford_area1.yaml`
- 数据协议：`ignore.txt`、`data/annotations/*.jsonl`、`data/provenance/*.json`
- 主 checkpoint：`outputs/slabim/accepted.pt`、`outputs/stanford_area1/accepted.pt`
- 数据说明：`docs/DATA_PREPARATION.md`
- 脚本与命令：`docs/USER_GUIDE.md`
- 评测设计：`docs/EVALUATION_PROTOCOL.md`
- pano 试错与 regular 语义增强路线：`docs/PANO_DEPTH_EVALUATION.md`
- attention scale、scratch curriculum、冻结 DA3 feature 与 hybrid residual：
  `docs/ATTENTIVE_SCALE_EXPERIMENT.md`

## 6. 新 chatbot 启动清单

1. 运行 `git status --short`，保留现有用户改动。
2. 读取上述文档和 `results/metrics.json`，不要从终端日志猜结果。
3. 运行 `pytest -q`、`ruff check src scripts tests` 和两个 checkpoint 的 SHA 校验。
4. fresh 数据 fingerprint 改变时使用 `materialize_runtime_config.py`，不把 hash 设为 null。
5. 解释/诊断任务默认只读；未经用户授权不重训、发布或访问新的 test 数据。

## 7. 已知限制

- 单 seed 结果不能表达训练随机性；bootstrap 只量化房间/帧采样不确定性。
- SLABIM 与 Area_1 都不是跨多个建筑的大规模外部验证。
- Area_1 使用目标域 semantic structural mesh 辅助 BIM 配准；部署需测量/定位系统提供变换。
- 冻结 DA3-feature/hybrid 候选只有单 seed，且 Area_1 test 已揭盲；须在新 blind
  area/dataset 上确认。hybrid additive head 的现有增益小到不支持增加部署复杂度。
- checkpoint 尚未上传公共 Release，`results/manifest.json` 的 URL 待所有者填写。
- 仓库仍需要所有者选择软件 LICENSE；第三方数据许可不能由本项目代授。
