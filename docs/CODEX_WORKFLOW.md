# Codex 对话要点与项目演化

> 给接手的新 chatbot / 研究者：先读本页，再读 `README.md`、`docs/USER_GUIDE.md` 和两套
> 正式结果 summary。不要从旧版网络或旧区域划分重新开始。

## 1. 当前结论

项目已经从“BIM 与旧 V1 预测加权融合”演化为“以非学习 BIM 矫正为锚点、学习有界残差”
的单帧深度系统。当前支持：

- SLABIM：公开下载、位姿恢复、坏帧排除、pooled split、训练、E2E、2D/3D 评测。
- 2D-3D-S Area_1 + BIMSyn：公开下载、IFC 围护过滤、固定全局 BIM 配准、DA3 缓存、
  room split、robust scale 选择、迁移/target 训练、家具与冲突子集评测。
- 正式主模型：SLABIM frozen/E2E 与 Area_1 target frozen。Area_1 E2E 只是未晋级 challenger。

统一指标范围 `0.2–5.0 m`。最关键结果：

| 数据 / 方法 | AbsRel | MAE | 备注 |
|---|---:|---:|---|
| SLABIM raw DA3 | 0.19935 | 0.31109 | pooled test，108 帧 |
| SLABIM direct BIM | 0.08145 | 0.12939 | 非学习基线 |
| SLABIM frozen | **0.06211** | **0.09689** | 比 direct AbsRel 好 23.74% |
| Area_1 raw DA3 | 0.30123 | 0.67323 | blind test，1641 帧/7 rooms |
| Area_1 robust BIM-direct | 0.07815 | 0.13891 | train-only cap |
| Area_1 target frozen | **0.06792** | **0.11748** | 比 robust direct AbsRel 好 13.08% |

完整数值在 `results/metrics.json`。

## 2. 对话驱动的演化时间线

### 阶段 A：读取原项目、数据和位姿

1. 拉取并理解 `gaobiaoli/PriorBIMDA`。
2. 准备 SLABIM 数据；用户随后放入 `sensor_data`，用于核对原作者生成的位姿。
3. 复核相机/LiDAR/BIM 坐标方向、时间戳配对与深度定义，确认不能靠错误的 pose 解释性能。
4. 重现旧 BIM 矫正约 `AbsRel≈0.07` 的来源，区分 raw DA3、global scale 和 direct local BIM，
   避免把不同区域/支持集的数值混为同一个 baseline。

### 阶段 B：否定 V1/BIM 加权融合

旧设计把 BIM 与 V1 输出做权重融合，存在三个根本问题：

- 融合对象都可能有系统性偏差，权重不能保证安全改进；
- 旧候选缓存、区域阈值和门控使训练/推理协议复杂且不可迁移；
- 学习模型可能只学会“选择哪个旧结果”，而没有学习几何残差空间。

因此删除 V1/V2/V3 的候选融合、信任网络和局部仿射分支。历史代码/config 不再是活动路径。

### 阶段 C：V5 尺度锚定残差网络

参考 prior-domain-adaptation 的思路，先用 BIM 矫正 DA3 尺度，再由网络微调。输入同时包含：

- RGB；
- DA3 base depth、置信度与 log-depth 几何；
- BIM depth、valid、normal、edge 及 base/BIM 关系。

输出拆成 frame、low-frequency、detail 三个 log-residual：

```text
D_pred = D_anchor * exp(clamp(r_frame + r_low + r_detail))
```

三个分量分别承担全帧偏差、平滑空间偏差和局部边缘；总 residual 有界。可靠性和不确定性是
辅助监督，不做会导致硬切换伪影的推理门控。训练 acceptance 要求 learned depth 在同一固定
GT support 上优于 direct BIM 的 AbsRel/MAE，并保护 near range。

### 阶段 D：资源利用、区域 CV 与全局 pooled split

先做了六区域交叉验证、batch/image-size 调整和单 seed 对比，确认不同区域误差主要来自：

- BIM/pose 局部偏移；
- DA3 尺度尾部；
- BIM coverage、近距遮挡与几何冲突；
- 场景像素分布差异，而非 RGB 外观本身。

随后用户确认 `ignore.txt` 是源数据错误清单，应在数据引入时排除。最终 SLABIM 不再按区域
训练/验证/测试，而是用 exhaustive annotation 做时序分段 pooled split：811 总记录，90 个
source-data-error，13 个 fused-LiDAR embargo，活动 `496/104/108`。不复制源文件。

### 阶段 E：DA3 部分 E2E

加入 pinned DA3 Metric Large 的 last-stage 微调，refiner LR 与 DA3 LR 分离。关键公平性修复：

- E2E 的 anchor 必须由当前 live DA3 计算，不能用 frozen cache 的 direct BIM；
- loss、validation 和 acceptance 都与 live robust/direct anchor 比较；
- validation 前重置独立 inference RNG，避免 DA3 内部随机采样使 accepted 与正式评测不一致。

SLABIM E2E 的 AbsRel/MAE 略优 frozen，但 RMSE 稍差，因此两个模型均保留。

### 阶段 F：Area_1 + BIMSyn 适配

下载并核验：

- 2D-3D-S Area_1 noXYZ：10,327 个规则视图，44 rooms，186 camera UUID；regular depth 是
  `uint16/512 m` 的 camera-z，不是 radial range；pose JSON 的 3×4 `[R|t]` 是 Area→camera。
- BIMSyn：44 个同名 IFC/RVT；IFC 是 mm、IFC2X3，计算只需 IFC。

IFC 原始内容包含家具。固定 prior 只保留 wall/slab-floor/covering-ceiling/column/beam；排除
door/window、furnishing、proxy、MEP 和 openings。44 个 room BIM 用同一组固定
`T_area_from_bim` 合入全局 Area BIM，不能逐帧用 test depth ICP。

当前 alignment 用 Area semantic structural mesh 辅助 4-DoF 配准，且仅在几何近等价候选间
用 door/window/beam/column 类别重排。这是公开披露的 scan-calibrated/oracle-style 协议，
不是完全无目标域标定。

### 阶段 G：robust scale 与 target-domain 模型

Area_1 家具会让 `BIM/base` 产生单侧大比值尾，旧 q45 scale 会出现过尺度。只用 30 个 train
rooms 注册选择：

```text
log s = min(Q45(log(BIM/base)), Q25 + c25, Q10 + c10)
```

固定 48 个候选、leave-one-train-room-out；最终 `c10=∞, c25=0.05`。selector 明确记录
validation/test opened=0。这个 robust estimator 同时用于模型 anchor、非学习 comparator、loss
和 acceptance，保证 learned 改进不是仅来自换 quantile。

SLABIM source refiner 在 Area_1 val 上零样本迁移很差：frozen AbsRel `0.16690`，而 robust
BIM-direct 为 `0.08710`。诊断表明 source frame residual 平均方向与 target 理想方向相反。
因此 target frozen 初始化严格归零 6 个 multiplicative residual 输出 slice（275 参数），保留
encoder/fusion，并做 1 epoch heads/adapters warmup；robust anchor 零 residual 时精确等于
BIM-direct。

12 epochs 后最佳为第 11 个 human epoch：val AbsRel `0.07005`。正式 val 中 all/furniture/
conflict 的 pixel/frame/room 九格均优于 robust BIM-direct，room bootstrap 结论通过。之后才读取
blind test，得到 `0.06792`。

Area_1 E2E 从 target frozen 保留 residual heads 初始化，validation AbsRel `0.07019`，比 frozen
差约 0.193%，因此未晋级、未做 test，权重也从清理后的 `outputs/` 删除。

### 阶段 H：公开复现整理

最后一次整理完成：

- 配置从带 `v4/v5/v6/global_clean/resource` 的过程名收敛为 12 个语义配置；
- `outputs/` 从约 13 GB 裁到约 1.4 GB，只留 3 个生产 checkpoint、正式结果和 challenger
  审计摘要；小结果迁入 tracked `results/`；
- 增加 Area_1/BIMSyn 下载器、固定 88 文件 BIMSyn hash manifest、SLABIM 固定 revision/hash
  manifest、annotation-aware audit、portable runtime config 生成器；
- 冻结 alignment receipt 移入 `data/provenance/`；
- 最终仓库只保留 12 个语义配置和 30 个有手册的 CLI；两个数据 manifest 均进入
  wheel，全部 273 个测试通过；
- 30 个 CLI 进一步按责任拆成 `scripts/data/` 15 个、`scripts/model/` 5 个、
  `scripts/pipelines/` 4 个和 `scripts/analysis/` 6 个；训练前流程独立记录在
  `docs/DATA_PREPARATION.md`，拆分后全部 275 个测试通过；
- 训练 history/run-state、正式 summary 和逐帧结果已移入可提交的 `results/`，
  `outputs/` 仅保留 3 个生产 checkpoint 及必要本机副本；
- 删除旧 chatlog/archive，以本页代替冗长逐轮对话。

## 3. 不可破坏的实验约束

1. **固定支持集**：所有可比方法用完全相同的 `gt_valid`；预测无效应报错，不能静默删像素。
2. **测试只读一次**：参数、模型、checkpoint 和 claim 在 val 锁定后才运行 test。
3. **split 隔离**：SLABIM 保证 fused-LiDAR 不跨 split；Area_1 按 room/camera UUID 隔离。
4. **训练标签不进推理**：furniture/conflict mask 只用于 loss weighting 和评测，不是模型输入。
5. **固定 BIM**：Area_1 不做逐帧配准、尺度或 pose 微调；SLABIM 新区域仍需外部位姿。
6. **严格 provenance**：annotation raw SHA、manifest preparation fingerprint、alignment、scale
   receipt、checkpoint model config 都必须核验。不要为了“跑起来”关闭这些校验。
7. **跨数据集边界**：source→target 初始化必须显式允许；resume 永远必须同数据集、同配置。
8. **DA3 pinned**：不得把 mutable `main` 或无 revision cache 混入正式实验。

## 4. 当前关键文件

### 配置

- SLABIM：`configs/slabim_pretrain.yaml` → `configs/slabim.yaml` →
  `configs/slabim_e2e.yaml`
- Area_1 transfer：`configs/stanford_area1_transfer.yaml` / `_e2e.yaml`
- Area_1 target：`configs/stanford_area1.yaml` / `_e2e.yaml`
- fresh data：用 `scripts/data/materialize_runtime_config.py` 创建 `configs/local/*.yaml`

### 数据协议

- `ignore.txt`
- `data/annotations/slabim_clean_global_v1.jsonl`
- `data/annotations/stanford_area1_room_v1.jsonl`
- `data/provenance/stanford_area1_bimsyn_alignment.json`
- `data/provenance/stanford_area1_robust_scale_selection_v1.json`

robust receipt 与历史 checkpoint 密码学绑定，内部三个绝对路径字段只是生成时审计字符串，
不能在不更新 checkpoint/config 的情况下“美化”文件内容。
alignment receipt 中的源 IFC/semantic 绝对路径同样只是审计字符串；运行时
使用配置中的当前数据根目录和回执里的哈希/变换数值。

### checkpoint

- `outputs/slabim/accepted.pt`：SHA `a0e339fe...b719`
- `outputs/slabim_e2e/accepted.pt`：SHA `5ec9d25f...e583`
- `outputs/stanford_area1/accepted.pt`：SHA `b651c190...00f4`

完整值、字节数和发布角色见 `results/manifest.json`。

## 5. 新 chatbot 的启动清单

1. `git status --short`，确认现有 dirty worktree；不要覆盖用户修改。
2. 阅读本页和 `docs/USER_GUIDE.md`。
3. 读取 `results/metrics.json` 与相关正式 summary，不从训练日志猜指标。
4. `pytest -q`、`ruff check src scripts tests`；先修复回归，再做新实验。
5. 若任务是解释/诊断，只读；除非用户明确要求，不自动重训、删除、发布或访问 test。
6. 若 fresh 数据 fingerprint 改变，生成 local child config；不要把 config hash 设置为 null，
   不要使用 `--allow-unverified-robust-comparator` 伪装正式结果。
7. target E2E 只有 val 同时胜 frozen、cached/live robust direct 且 near 不退化时才可晋级。

## 6. 已知限制与下一步

- 当前训练仍是单 seed；不能把小差异解释为多 seed 均值。
- SLABIM 是同一建筑内 pooled split；Area_1 也只有一个 Area。跨建筑外部效度仍有限。
- Area_1 alignment 使用发布的目标域结构 mesh；实践版需独立测量变换。
- 大型 E2E checkpoint 尚未上传公共 release；`results/manifest.json` 需要发布 URL。
- 仓库所有者尚未选择软件 LICENSE，这是公开发布前最后的治理阻塞项。
- 上述两项是需要仓库所有者/外部托管权限的发布动作；不应由新 chatbot
  自行选择许可证或上传权重。
