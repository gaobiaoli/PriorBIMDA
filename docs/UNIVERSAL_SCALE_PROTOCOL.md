# 统一尺度估计与模型锚点协议

## 1. 为什么需要统一

旧实验在 SLABIM 使用 `Q45(BIM/DA3)`，在 Area_1 使用带上界截断的稳健尺度；此外，旧
Area_1 网络把 BIM-direct 深度作为学习残差锚点，而 SLABIM 网络以尺度矫正后的 DA3 为锚点。
这会把“数据集适配”混入方法定义，使跨数据集比较难以解释。

当前公开协议删除了这两处差异。SLABIM、Area_1 和新场景都运行同一条规则：

```text
r = log(BIM / DA3),  0.2 < BIM/DA3 < 5.0
log(s) = min(Q45(r), Q25(r) + 0.05)
D_scaled = s * D_DA3
D_pred = D_scaled * exp(clamp(r_frame + r_low + r_detail))
```

有效比例少于 100 个像素时尺度回退为 1。公式只读取固定 BIM、BIM valid mask 与 DA3
深度；不读取 GT、语义、家具 mask、区域 ID，也不在测试时拟合参数。

## 2. BIM 在统一框架中的角色

BIM 有两个彼此独立的角色：

1. 参与上式的全局尺度估计，并作为网络的几何条件输入；
2. 构造确定性的 `universal_bim_direct` 非学习比较器。

学习输出始终以 `D_scaled` 为乘法锚点。`universal_bim_direct` 不再作为某个数据集专属的
网络锚点，因此零残差的语义、输入通道和深度路由在所有数据集上完全相同。模型验收仍要求
学习结果在相同 GT support 上同时优于 `universal_bim_direct` 的 AbsRel 与 MAE，并保护近距
误差；这能防止网络仅通过换尺度规则获得表面提升。

## 3. 参数来源与科研边界

常数 `0.05` 最初由 Area_1 的 30 个 train rooms、固定 48 候选网格和 leave-one-room-out
协议选出。validation/test 在选择时均未打开。为了形成单一公共方法，该参数随后被冻结，
没有在 SLABIM 上重新选择；SLABIM validation 只用于确认固定规则不会失效。

因此应把当前结果表述为“一个冻结超参数在两个不同数据集上的复用”，不能表述为“完全没有
任何目标域开发数据的先验设计”。更强的未来验证是在第三个建筑数据集上原样应用 receipt，
不修改 `0.05`、ratio 范围或最小支持数。

冻结的机器可读协议为
[`data/provenance/universal_scale_estimator_v1.json`](../data/provenance/universal_scale_estimator_v1.json)。
训练和评测会校验该文件的 SHA-256；配置若覆盖 estimator、恢复 BIM-direct 网络锚点或移除
receipt 会直接失败。

## 4. 正式评测要求

- 两个数据集都必须报告 raw DA3、`universal_global_scale`、
  `universal_bim_direct` 和 learned refined；历史 q45 只能标作 retrospective baseline。
- 所有可比方法使用同一 `gt_valid` 和 `0.2–5.0 m` 范围，无效预测不得从分母静默删除。
- 分别报告 pixel-micro、frame-macro 和 region/room-macro；Area_1 另报 furniture、
  BIM-conflict、BIM-consistent 与 BIM-no-hit 子集。
- 迁移实验区分同一 SLABIM checkpoint 的 Area_1 zero-shot 与 Area_1 supervised fine-tune，
  不把二者混写为同一种泛化。
- 新数据集不得修改 estimator；如确需改动，必须提升 protocol schema/name 并作为新方法重跑
  所有基线和学习模型。

