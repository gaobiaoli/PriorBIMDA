# 实验与数据流水线

本文说明两套正式 pipeline 的数据边界、状态转换和复现实验顺序。逐个 CLI 参数见
[USER_GUIDE.md](USER_GUIDE.md)；仅关心训练前数据步骤时读
[DATA_PREPARATION.md](DATA_PREPARATION.md)。

## 共同原则

```text
public raw data
  -> verified source inventory
  -> pinned DA3 + fixed BIM rendering
  -> immutable manifest
  -> exhaustive split annotation
  -> one frozen universal scale + non-learning BIM baseline
  -> train/val model selection
  -> one-time blind test
```

- `0.2–5.0 m` 是正式深度协议。
- split 在训练前固定，坏帧在 annotation 中标记 `excluded`。
- manifest 的逐帧 preparation fingerprint 会进入 split fingerprint 和 checkpoint provenance。
- test 不参与 cap、loss、epoch、checkpoint 或阈值选择。
- 所有方法共享同一 GT support；无效预测直接失败。

## SLABIM

入口：

```bash
python scripts/pipelines/run_slabim_experiments.py --slabim-root ../SLABIM --stages all
```

### 数据

- 原始来源由 `src/bim_priorda3/data/slabim_download_manifest.json` 固定 revision、size 和
  SHA-256。
- 位姿由 rosbag 轨迹与官方 SLAM PCD 做离线 ICP 恢复，并写回各 region。
- `scripts/data/prepare_dataset.py` 生成 RGB 引用和包含 DA3/BIM/GT 的 NPZ。
- `slabim_clean_global_v1.jsonl` 对 manifest exhaustive：811 总记录；103 excluded，其中
  90 个 `ignore.txt` 数据错误，13 个 fused-LiDAR embargo；活动 496/104/108。

### 训练

```text
slabim_pretrain.yaml --accepted.pt--> slabim.yaml --accepted.pt--> slabim_e2e.yaml (optional)
```

`slabim_pretrain` 先学习无 near-routing 的多尺度 residual；`slabim` 启用 depth-aware routing
并用 504²、BS8 微调。活动主模型始终以 universal scaled DA3 为 residual anchor；
`robust/universal BIM-direct` 仅参与 loss/acceptance 公平比较。

### 产物

- `outputs/slabim/accepted.pt`：唯一 SLABIM 生产 checkpoint。
- `results/slabim/`：正式 summary、逐帧 CSV 和训练审计。

## 2D-3D-S Area_1 + BIMSyn

### 数据

- Area_1 noXYZ TAR：32,684,605,440 bytes，MD5
  `21098fbe93b561e30e79197a95fa4fd2`。
- 10,327 个规则视图；regular depth 为 camera-z `uint16/512 m`，65535 无效。
- pose 的 `camera_rt_matrix` 实际是 3×4 Area→camera `[R|t]`。
- BIMSyn 44 个 IFC 是 room-local、mm、IFC2X3；RVT 可选。
- downloader/verifier 按 88 文件 canonical manifest 验证 IFC/RVT。

### BIM prior

1. 每个 IFC 仅保留 wall/floor/ceiling/column/beam。
2. 44 个固定 `T_area_from_bim` 将 room BIM 合入一个全局 scene。
3. 相机用 JSON Area→camera 位姿直接从全局 scene 渲染 camera-z。
4. 不允许逐帧 ICP、尺度或位姿微调。

冻结 alignment receipt：

```text
data/provenance/stanford_area1_bimsyn_alignment.json
SHA256 079ff394fbfa9317953e0358d71e0548cd39171278dd16121d6c300c5a23e6d6
```

### split 与统一尺度

room-disjoint split 为 30/7/7 rooms，7013/1673/1641 frames。robust scale selector 只打开
train NPZ，固定 48 候选和 leave-one-room-out，最终规则被冻结，并原样用于 SLABIM：

```yaml
name: log_upper_cap_v1
q10_log_cap: inf
q25_log_cap: 0.05
ratio_min: 0.2
ratio_max: 5.0
min_samples: 100
```

fresh cache 会生成新的 preparation fingerprint。此时必须运行
`scripts/data/materialize_runtime_config.py` 绑定当前
annotation/split/alignment/scale receipt；不能把 config
中的 SHA 置空。

### 模型选择

```text
SLABIM universal --cross-dataset init + zero residual heads--> Area_1 universal
```

当前统一 target 模型 val AbsRel 为 0.07115，优于 universal direct 0.08710；锁定 checkpoint
后运行 test，得到 0.06689 vs direct 0.07815。旧 E2E challenger 与 BIM-direct 网络锚点不属于
当前协议，权重和过程结果均已清理。

### 产物

- `outputs/stanford_area1/accepted.pt`：Area_1 universal `0.2–5.0 m` 兼容 checkpoint。
- `outputs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth/accepted.pt`：当前推荐的
  Area_1 官方全深度发布 checkpoint。
- `results/stanford_area1/`：两种支持域各自的 val/test、逐帧 CSV 和训练审计；禁止混表。

可靠性门控后继版本没有超过全深度发布 checkpoint，输出只作为负实验审计，不属于生产
产物。发布状态以 `results/manifest.json` 的 `publish` 字段为准。

## 恢复与防错

- `scripts/model/train.py --resume` 只接受同配置、同数据 fingerprint 和同训练源码；
  不要给 resume 使用
  cross-dataset 开关。
- fresh run 使用全新空 output 目录；已有 history/run_state/checkpoint 会 fail-fast，避免旧
  accepted.pt 被误认作新结果。
- smoke subset checkpoint 的 provenance 包含实际 sample IDs/fingerprints，不能冒充 full run。
- Area_1 evaluator 默认 `split=val`。只有明确传 `--split test` 才会读取 blind test。
- 运行失败时先读 `run_state.json`；不要删除 receipt 或绕过 hash 检查。
