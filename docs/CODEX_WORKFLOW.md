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

## 4. 不可破坏的实验约束

1. 所有可比预测使用相同 `gt_valid` 和 `0.2–5.0 m`；不能按方法删无效像素。
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
- checkpoint 尚未上传公共 Release，`results/manifest.json` 的 URL 待所有者填写。
- 仓库仍需要所有者选择软件 LICENSE；第三方数据许可不能由本项目代授。
