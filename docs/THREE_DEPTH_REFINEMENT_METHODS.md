# 三尺度深度细化网络

本文说明当前公开版本的学习模块。项目只保留一条正式方法链：SLABIM 与
2D-3D-S Area 1 使用相同的尺度估计、相同的确定性 BIM 基线和相同的三尺度残差网络。

## 1. 统一输入协议

令 DA3 输出为 `D`，渲染的固定围护 BIM 深度为 `B`。先在两者都有效的位置计算
`r = B / D`，只保留 `0.2 < r < 5.0`。有效像素不少于 100 时：

```text
s = exp(min(Q45(log r), Q25(log r) + 0.05))
```

否则回退到 `s=1`。模型的乘法锚点始终是：

```text
A = s * D
```

确定性 `universal_bim_direct` 在 `A` 上应用固定局部 BIM 矫正。它是强基线、训练约束
和验收对照，不是学习模型的输出锚点。完整冻结协议与参数来源见
[UNIVERSAL_SCALE_PROTOCOL.md](UNIVERSAL_SCALE_PROTOCOL.md)。

## 2. 三种改进方法

### 2.1 帧级残差

帧级分支预测全图共享的对数深度偏移，负责剩余的全局尺度/偏置。它按深度路由，近距
区域会减弱该分量，避免一个全局数覆盖家具和边界的局部变化。

### 2.2 低频空间残差

低频分支预测平滑的空间变化，纠正 BIM 对齐、DA3 几何与真实场景之间缓慢变化的误差。
它保留空间结构，但不会承担细小物体边缘。

### 2.3 高频细节残差

细节分支恢复家具轮廓、遮挡边界与局部几何变化。RGB、DA3 几何条件和 BIM 条件在多
尺度融合后共同驱动该分支。

三个分量求和并限幅：

```text
R = clip(R_frame + R_low + R_detail, -0.45, 0.45)
D_refined = A * exp(R)
```

因此网络学习的是对数空间中的有界修正，不是 BIM 与 DA3 的加权平均。

## 3. 条件输入

- RGB 分支：外观、遮挡和物体边界；
- DA3 几何分支：原始/尺度化深度、置信与几何变化；
- BIM 分支：深度、有效掩码、法向、边缘和 BIM–DA3 分歧；
- BIM reliability：辅助监督信号，不作为输出的乘法门。

推理不使用 GT、家具 mask 或语义标签。Area 1 的家具/conflict mask 只用于训练权重和
分层评测。

## 4. 两个正式模型

| 数据域 | 配置 | checkpoint | 作用 |
|---|---|---|---|
| SLABIM | `configs/slabim.yaml` | `outputs/slabim/accepted.pt` | SLABIM 正式模型与 Area 1 初始化源 |
| Area 1 | `configs/stanford_area1.yaml` | `outputs/stanford_area1/accepted.pt` | 房间隔离训练的正式模型 |

`configs/*_e2e.yaml` 仅保留为研究入口，用于联合微调 DA3 后段；当前公开主结果不依赖
E2E checkpoint。

## 5. 训练与验收

训练损失以 `log(GT)-log(A)` 为残差目标，并使用固定 GT support。模型必须同时满足：

- AbsRel 与 MAE 严格优于同输入的 `universal_bim_direct`；
- 近距 AbsRel 不超过基线容差；
- 所有可比方法的有效像素计数完全一致；
- 预测含 NaN、Inf 或非正值时直接失败。

Area 1 从 SLABIM checkpoint 初始化时精确清零六个乘法残差输出切片，使初始预测逐像素
等于 `A`；随后在目标 train/val 上训练。这样不会把源域残差头的方向偏差带到目标域。

## 6. 正式测试结果

固定深度协议为 0.2–5.0 m，以下为 pixel-micro：

| 数据集 | Raw DA3 | 统一尺度 | BIM direct | Learned | Learned 相对 direct |
|---|---:|---:|---:|---:|---:|
| SLABIM test（108） | 0.19935 | 0.06361 | 0.06263 | **0.05601** | **-10.56%** |
| Area 1 test（1641） | 0.30123 | 0.07752 | 0.07815 | **0.06689** | **-14.41%** |

完整 MAE/RMSE/δ1、子集和 bootstrap 见 [results/README.md](../results/README.md)。

## 7. 运行入口

```bash
python scripts/model/train.py --config configs/slabim.yaml --device cuda
python scripts/model/evaluate.py \
  --config configs/slabim.yaml \
  --checkpoint outputs/slabim/accepted.pt \
  --split test --output outputs/slabim/evaluation_test --device cuda

python scripts/model/train.py \
  --config configs/stanford_area1.yaml \
  --init-checkpoint outputs/slabim/accepted.pt \
  --allow-cross-dataset-initialization --device cuda
python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1.yaml \
  --checkpoint outputs/stanford_area1/accepted.pt \
  --split test --output outputs/stanford_area1/formal_test \
  --batch-size 8 --inference-seed 42 --device cuda
```

数据准备步骤见 [DATA_PREPARATION.md](DATA_PREPARATION.md)，完整脚本索引见
[USER_GUIDE.md](USER_GUIDE.md)。
