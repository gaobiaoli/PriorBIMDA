# Attention-based global scale experiment

## Status

This document records the completed Area_1 attention-scale development line.
The public Area_1 full-depth release is now frozen at
`stanford_area1_attentive_scale_da3_features_hit_only_full_depth`: attentive
scalar scale, frozen DA3 layer-11/layer-23 features, and the no-additive
low/detail refiner. The later reliability-gated successor regressed final
validation/test AbsRel and is archived as a negative diagnostic. Earlier
bounded-core and hybrid runs remain model-design evidence, not release
checkpoints. Area_1 test had already been revealed before these iterations, so
publishing the selected checkpoint supports reproducibility/deployment but does
not turn its test numbers into a new blind confirmation; that still requires a
new area/dataset.

## Network

DA3 is frozen and read from the pinned cache.  For every BIM-valid pixel,

\[
x_i = \log(D_{\mathrm{BIM},i}/D_{\mathrm{DA3},i}).
\]

The attention network receives RGB, DA3 depth/confidence, BIM depth/validity,
normals, edges, and BIM--DA3 disagreement.  These features only produce the
spatial weights.  The values being aggregated remain the measured \(x_i\), so
the network cannot hallucinate metric scale directly from RGB.

Four attention heads perform two differentiable Huber/IRLS updates over the
measured log-ratios.  The result is one scalar scale per image.  The frozen
universal estimator

\[
\log s_0=\min(Q_{45}, Q_{25}+0.05)
\]

is used only as a learned low-confidence fallback; support below 100 valid
ratios forces the fallback exactly.  The coarse depth is

\[
D_{\mathrm{scale}}=\hat{s}D_{\mathrm{DA3}}.
\]

The old frame/global residual is disabled because it is redundant with the
learned scalar.  The checkpoint-compatible low/detail branches predict the
remaining spatial log residual:

\[
D_{\mathrm{final}}=D_{\mathrm{scale}}
\exp(r_{\mathrm{low}}+r_{\mathrm{detail}}).
\]

The spatial residual is not hard-centered.  Its per-image valid-pixel mean is
softly regularized toward zero, allowing real residual bias when demanded by
the final metric-depth supervision.

Implementation:

- `src/bim_priorda3/models/attention_scale.py`
- `src/bim_priorda3/models/system.py`
- `src/bim_priorda3/losses.py`
- `configs/stanford_area1_attentive_scale.yaml`

## Training protocol

- Dataset: Area_1 room-disjoint split, train/val/test = 7013/1673/1641.
- Depth support: the same fixed official GT support in 0.2--5.0 m for every
  method.
- Initialization: existing Area_1 accepted refiner; only the new
  `attention_scale.*` tensors are missing and freshly initialized.
- Epoch 1: train only the attention-scale module.
- Epochs 2--12: jointly train the attention scale and low/detail refiner; DA3
  remains frozen.
- Resolution 504 x 504, physical batch 8, accumulation 2, effective batch 16.
- AdamW, weight decay 1e-4, scale-head LR 8e-5, refiner LR 4e-5, cosine decay,
  AMP, gradient clip 1.0.
- Metric GT directly supervises both coarse scaled depth and final refined
  depth.  Auxiliary terms include gradient, local residual, soft spatial-mean,
  attention entropy, and artificial DA3-scale equivariance losses.
- One controlled AMP overflow skipped an optimizer update: 5255/5256 updates
  succeeded.  All 12 validation passes were finite and accepted.
- Best checkpoint: human epoch 10 (`history` epoch 9).

Training command:

```bash
python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale.yaml \
  --init-checkpoint outputs/stanford_area1/accepted.pt \
  --device cuda
```

## Results on the same benchmark

All numbers below are pixel-micro and use exactly the same 0.2--5.0 m support.
`Learned scale` is the evaluator's `coarse` method for this configuration.

### Validation, 1673 frames / 7 rooms

| Method | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Raw DA3 | 0.27710 | 0.62840 | 0.79206 | 0.32894 |
| Universal scale | 0.08644 | 0.17422 | 0.39536 | 0.91533 |
| Direct BIM correction | 0.08710 | 0.17121 | 0.39455 | 0.91312 |
| Learned attention scale | 0.07058 | 0.14114 | 0.35653 | 0.93776 |
| Existing refiner | 0.07115 | 0.14316 | 0.35880 | 0.93000 |
| Attention-scale + low/detail | **0.06371** | **0.12759** | **0.33601** | **0.94303** |

The learned scale reduces AbsRel by 18.35% relative to the universal scale.
The complete candidate reduces AbsRel by 10.46% relative to the existing
refiner on validation.

### Test, 1641 frames / 7 rooms

| Method | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Raw DA3 | 0.30123 | 0.67323 | 0.83485 | 0.26437 |
| Universal scale | 0.07752 | 0.13938 | 0.31541 | 0.93710 |
| Direct BIM correction | 0.07815 | 0.13891 | 0.31350 | 0.93740 |
| Learned attention scale | 0.07361 | 0.12882 | 0.31639 | 0.93899 |
| Existing refiner | **0.06689** | **0.11761** | **0.30823** | **0.94295** |
| Attention-scale + low/detail | 0.06752 | 0.11870 | 0.30876 | 0.94219 |

The learned scale still improves over universal scale by 5.05% AbsRel.  The
complete candidate improves over direct BIM correction by 13.59%, but is 0.95%
worse than the existing refiner in AbsRel.  It is therefore not promoted.

Compared with the existing refiner, the candidate improves furniture AbsRel
from 0.08573 to 0.08064 and BIM-foreground-conflict AbsRel from 0.13772 to
0.13664.  It regresses on BIM-consistent pixels from 0.02924 to 0.03097.  This
suggests the attention scale is useful for movable/ambiguous content but the
coupled spatial refiner sacrifices some already-strong structural accuracy.

Against direct BIM correction, the test all-pixel improvement is stable over
rooms: learned-minus-direct room-bootstrap AbsRel difference is -0.01240 with
95% CI [-0.01938, -0.00402], and 6/7 rooms improve.  This supports the method's
value over the non-learning baseline, but not superiority over the existing
learned refiner.

## Evaluation commands and artifacts

```bash
python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale/accepted.pt \
  --split val \
  --output results/stanford_area1/attentive_scale_val \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42

python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale/accepted.pt \
  --split test \
  --output results/stanford_area1/attentive_scale_test \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42
```

Key SHA256 identities:

- accepted checkpoint: `631c7a0f3b95ccc89215693a95514758fb3ee5f788303c8ed97c8437f626eee9`
- validation summary: `96383e9d28d14d14f47b95c76da2d1d4a98179774340f4e0fd0fb1e9a3244afe`
- validation per-frame CSV: `406610798c9353382c5eaa0f058b626bb30175023336c4d4eb1de691e74004a6`
- test summary: `cc0028b818b322d29a3c97dd90a4ee1770ebddc848edd5a56c3f2bda1c2b715c`
- test per-frame CSV: `01915f940008995d7ccaf0dbe9083fe8cb0af0d36839719ff020c9195b56d250`

The test split had already been revealed by earlier project experiments before
this architecture was proposed.  The test table is therefore a post-hoc
candidate evaluation, not a newly blinded confirmatory result.  No further
model or hyperparameter change should be selected from these test numbers.

## Frozen DA3 encoder-feature candidate

The fourth candidate keeps the bounded-MLP scratch curriculum but
adds native features from the same frozen DA3 encoder that produced the input
depth. It does not load an earlier refiner checkpoint and does not fine-tune
DA3. Layer 11 and layer 23 ViT-L/14 tokens are cached as FP16
`1024 x 36 x 36` grids from the pinned 504-pixel preprocessing pass.

The scale head projects the layer-11 grid to 96 channels and fuses it with the
existing RGB/DA3/BIM key encoder before predicting attention logits. A separate
projection and global pool of layer 23 is concatenated with the masked pooled
key feature for head mixing, fallback confidence, and the bounded scale MLP.
The token values being aggregated remain measured `log(BIM/DA3)` ratios: DA3
features decide where to look and how much to trust the result, but do not
replace the metric anchor.

The spatial refiner fuses layer 11 only at its 1/4-resolution level and layer 23
only at its 1/8-resolution bottleneck. The full-resolution RGB path and detail
head are unchanged. Thus high-level DA3 appearance/geometry can guide the low
residual while the shallow RGB stream preserves local boundaries. No zero
`gamma` is placed in front of these native branches; their projections are part
of the model from the first scratch-training update. The output heads themselves
retain the existing safe zero initialization.

Configuration and preparation command:

```bash
python scripts/data/cache_stanford_da3_features.py \
  --config configs/stanford_area1_attentive_scale_da3_features.yaml \
  --device cuda --batch-size 16

python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_da3_features.yaml \
  --device cuda
```

The train/validation/test split, 0.2--5.0 m support, direct scale target, loss
weights, augmentation, optimizer rates, batch size, and 3/9/3
scale/refiner/joint schedule are kept equal to the preceding bounded-MLP
scratch run. Horizontal augmentation applies the same spatial flip to the
cached token grids. The task network has 4,267,754 trainable parameters
(810,484 in the scale head and 3,457,270 in the refiner); all DA3 parameters
stay frozen and are absent from the optimizer.

All 15 epochs completed. Three controlled AMP overflows skipped one optimizer
step each, so 6,567/6,570 attempts succeeded; every train/validation metric
remained finite. Human epoch 14 (`history` epoch 13) was selected strictly by
validation refined AbsRel. Human epoch 15 was not selected. Peak allocated CUDA
memory was 11.00 GiB in the single-branch stages; total device use in the joint
stage remained within the 16 GiB GPU.

### DA3-feature results on the unchanged benchmark

All rows use the same frozen input depth, room split, fixed 0.2--5.0 m support,
masks, and pixel-micro evaluator. `Scale output` precedes low/detail refinement.

#### Validation, 1,673 frames / 7 rooms

| Method | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Universal scale | 0.08644 | 0.17422 | 0.39536 | 0.91533 |
| BIM-direct | 0.08710 | 0.17121 | 0.39455 | 0.91312 |
| Previous bounded-MLP scratch | 0.06387 | 0.12881 | 0.32380 | 0.94435 |
| DA3-feature scale output | 0.06960 | 0.13942 | 0.35738 | 0.93567 |
| DA3-feature final output | **0.06209** | **0.12401** | **0.30842** | **0.94615** |

The final output improves over the previous scratch candidate by 2.78% AbsRel,
3.73% MAE, and 4.75% RMSE. It improves over BIM-direct by 28.71% AbsRel. The
room-paired difference against the previous candidate favors the new model in
5/7 validation rooms, but its room-bootstrap 95% CI is
[-0.00389, +0.00092], so the cross-room gain is not yet conclusive on validation.

#### Test, 1,641 frames / 7 rooms

| Method | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Raw DA3 | 0.30123 | 0.67323 | 0.83485 | 0.26437 |
| Universal scale | 0.07752 | 0.13938 | 0.31541 | 0.93710 |
| BIM-direct | 0.07815 | 0.13891 | 0.31350 | 0.93740 |
| Existing public refiner | 0.06689 | 0.11761 | 0.30823 | 0.94295 |
| Previous bounded-MLP scratch | 0.06442 | 0.11498 | 0.29802 | 0.94380 |
| DA3-feature scale output | 0.07144 | 0.12545 | 0.31189 | 0.93980 |
| DA3-feature final output | **0.06244** | **0.10780** | **0.28166** | **0.94391** |

The final output improves over the previous scratch candidate by 3.07% AbsRel,
6.25% MAE, and 5.49% RMSE; it improves over the existing public refiner by
6.66% AbsRel and over BIM-direct by 20.10%. Against the preceding scratch model,
6/7 test rooms improve and the post-hoc room-bootstrap difference is -0.00331
with 95% CI [-0.00622, -0.00020]. Against BIM-direct, all 7 rooms improve and
the pre-existing evaluator reports a room-bootstrap 95% CI
[-0.02781, -0.01099]. Test remains diagnostic because it was already revealed.

The scale output itself is worse than the previous scratch scale head
(validation 0.06960 versus 0.06877; test 0.07144 versus 0.07006). Therefore the
result does **not** show that DA3 features improve scalar estimation in
isolation. The complete network improves because the scale is recoverable by a
stronger spatial refiner; the training trajectory also places the first clear
gain after refiner feature fusion. Separating scale-head and refiner feature
fusion would require an ablation and is intentionally not inferred here.

Formal evaluator outputs:

- accepted checkpoint: `b4d3457936ba8d14921c2f72a0010249bba7d4702717797405e20368bdaa11a4`;
- feature-cache manifest: `f954559744b198a2c59906b532809185822f92721df43754c75407b42c0dd9de`;
- validation summary / CSV: `432290a297c7a8737be319928e5acc0228cb6ca3ae96579c54d869fbcee23adc` /
  `3c3ab5070e44f1ad81f6ec671b5b96a526fa9a391cd97066d61913a7fbbe1ecc`;
- test summary / CSV: `de2715e6b7a229f1c0e784d628724b1c2d7fb410be5a886ffdf9f47bb83f43e4` /
  `e727ec446c1c45fce23c1f21593b0e08e88e967db525c1871e53a206cd04ee11`.

## Follow-up: direct supervision of the learned scale

The follow-up keeps the network, data, initialization, optimization schedule,
and all other losses unchanged. Its only intervention is a direct loss on the
single learned scale. For every training image, the GT-derived scalar target
is the exact minimizer of scale-only AbsRel:

\[
s^*=\operatorname{weighted\ median}\left(
\frac{D_{\mathrm{GT},i}}{D_{\mathrm{DA3},i}};
w_i=\frac{D_{\mathrm{DA3},i}}{D_{\mathrm{GT},i}}
\right),
\qquad
\mathcal L_{\mathrm{scale}}=
\operatorname{SmoothL1}(\log\hat{s},\log s^*).
\]

The loss weight is 0.50, SmoothL1 beta is 0.02, and at least 100 valid GT
pixels are required. GT is used only to form a training/validation loss
target. It is not an input to the scale head, is not used by inference, and no
GT scale alignment is performed by either formal evaluator.

Implementation and configuration:

- `absrel_optimal_log_scale()` and `attention_scale_oracle` in
  `src/bim_priorda3/losses.py`;
- `configs/stanford_area1_attentive_scale_oracle.yaml`.

The run starts from the same pre-attention Area_1 checkpoint as the first
experiment, rather than continuing from the first attention experiment. This
makes the scale loss the only intended experimental variable.

```bash
python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_oracle.yaml \
  --init-checkpoint outputs/stanford_area1/accepted.pt \
  --device cuda
```

All 12 epochs completed. Two controlled AMP overflows skipped optimizer
updates (5254/5256 updates succeeded); every training and validation metric
remained finite. The selected checkpoint is human epoch 10 (`history` epoch
9).

### Direct scale-loss results

All rows below use the same Area_1 images, fixed 0.2--5.0 m support, masks,
aggregation, and evaluator. `Scale output` is the learned scalar applied to raw
DA3; `Final output` additionally applies the unchanged low/detail spatial
refiner.

#### Validation, 1673 frames / 7 rooms

| Variant | Output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---|---:|---:|---:|---:|
| Attention, no direct scale loss | Scale output | 0.07058 | 0.14114 | 0.35653 | 0.93776 |
| Attention + direct scale loss | Scale output | **0.06928** | **0.13876** | **0.35276** | **0.94053** |
| Attention, no direct scale loss | Final output | **0.06371** | **0.12759** | 0.33601 | 0.94303 |
| Attention + direct scale loss | Final output | 0.06373 | 0.12791 | **0.33390** | **0.94561** |

The direct target improves the scale head's validation AbsRel by 1.85%. The
final AbsRel is effectively unchanged (+0.00002), while RMSE and delta1
improve. Thus the supervision is doing what it was designed to do, but the
unchanged spatial refiner can compensate for much of the original scale-head
error.

#### Test, 1641 frames / 7 rooms

| Variant | Output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---|---:|---:|---:|---:|
| Attention, no direct scale loss | Scale output | 0.07361 | 0.12882 | 0.31639 | 0.93899 |
| Attention + direct scale loss | Scale output | **0.07029** | **0.12544** | **0.31294** | **0.94285** |
| Attention, no direct scale loss | Final output | 0.06752 | 0.11870 | 0.30876 | 0.94219 |
| Existing public refiner | Final output | 0.06689 | **0.11761** | 0.30823 | 0.94295 |
| Attention + direct scale loss | Final output | **0.06578** | 0.11763 | **0.30614** | **0.94438** |

The scale head improves by 4.51% AbsRel relative to the same attention model
without direct scale supervision. The final output improves by 2.58% over that
model and by 1.66% over the existing public refiner. Its MAE is only 0.000016 m
worse than the public refiner, whereas RMSE improves by 0.68% and delta1 by
0.142 percentage points.

The gain is not yet uniform enough to promote this checkpoint as the public
default. Against the existing refiner, room-macro AbsRel changes from 0.07777
to 0.07697, but only 3/7 rooms improve and the paired room-bootstrap 95% CI of
the difference is [-0.00379, +0.00181]. By subset, furniture AbsRel improves
from 0.08573 to 0.07792 and BIM-conflict AbsRel from 0.13772 to 0.13023, while
BIM-consistent pixels regress from 0.02924 to 0.03081. These post-hoc test
results justify evaluating the design on a new blind area/split, not further
tuning it on the revealed Area_1 test set.

Formal commands use the same evaluator invocation shown above with the new
config/checkpoint and output directories
`results/stanford_area1/attentive_scale_oracle_{val,test}`.

Key SHA256 identities:

- configuration: `e24e52c9bd3eb24556de5cf8316e09654da15ccdb729cc6e79bc5a85dcefd87f`;
- accepted checkpoint: `681bbf2b4004e41baf6704c8cb516b2a6bdf01715d8a1921da4522edb057b5e8`;
- validation summary / CSV: `426815700cfe628877bb9410f68f2335e6301b413e2ecb01df6c94b818562b5f` /
  `9f5de3b2f4e78f701fc6943e17f24d3982cdb3f950448044558793b1622e77ff`;
- test summary / CSV: `df394bba02da8f3cebae62ab88d039a139b23a56efb5e1bd46108f19a196f6cb` /
  `faaf836ba64e04928e3ef6376dfeaee1f3413fb4bb1711b0b15e3584110668b5`.

## Bounded MLP scale correction with scratch curriculum

### Change to the scale head

The final candidate retains measured BIM/DA3 log-ratios and the safety
fallback. It does not replace them with an unconstrained RGB regression.
Instead, the 96-channel valid-token pooled feature `z` is passed through a
small 96--32--1 MLP:

\[
\Delta\alpha_s=0.05\tanh(\operatorname{MLP}(z)),\qquad
\alpha_{\mathrm{att}}'=\alpha_{\mathrm{att}}+\Delta\alpha_s.
\]

The last linear layer is initialized to zero, so the new branch is an exact
no-op at the start of training. The correction is bounded to +/-0.05 in log
scale (at most about +/-5.13% multiplicative change) and is applied before the
existing fallback gate. This adds 3,137 parameters. The complete frozen-DA3
task network has 2,799,050 trainable/non-DA3 parameters.

### Training policy

Unlike both earlier attention runs, this run does **not** load the existing
Area_1 refiner or another attention checkpoint. All task-specific weights use
their deterministic fresh initialization; only DA3 stays frozen and its
pinned cache is reused. A single optimizer and cosine schedule span all 15
epochs:

1. epochs 1--3: train only the attention scale head (277,396 parameters);
2. epochs 4--12: freeze the scale head in evaluation mode and train only the
   low/detail refiner (2,521,654 parameters);
3. epochs 13--15: jointly fine-tune both at the already-low cosine learning
   rate (2,799,050 parameters).

The initial learning rates are 8e-5 for the scale head and 4e-5 for the other
task modules. The direct per-frame AbsRel-optimal scale loss from the previous
experiment remains enabled. DA3 is never updated, no test-time GT is read,
and no initialization checkpoint is allowed by this schedule. All 15 epochs
and 6,570/6,570 optimizer updates completed without AMP skips. Human epoch 14
was selected strictly by validation refined AbsRel; epoch 15 was slightly
worse and therefore was not selected.

Training command:

```bash
python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_mlp_scratch.yaml \
  --device cuda
```

### Results on the unchanged benchmark

All values use the same room-disjoint split, fixed 0.2--5.0 m GT support,
cached raw DA3 input, evaluator, masks, and aggregation as the earlier tables.
`Scale output` is the learned image-level scale before low/detail refinement.

#### Validation, 1,673 frames / 7 rooms

| Variant | Output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---|---:|---:|---:|---:|
| Universal scale | Scale | 0.08644 | 0.17422 | 0.39536 | 0.91533 |
| Direct BIM correction | Direct | 0.08710 | 0.17121 | 0.39455 | 0.91312 |
| Previous attention + direct scale loss | Final | **0.06373** | **0.12791** | 0.33390 | **0.94561** |
| Bounded MLP + scratch curriculum | Scale | 0.06877 | 0.13730 | 0.35655 | 0.93932 |
| Bounded MLP + scratch curriculum | Final | 0.06387 | 0.12881 | **0.32380** | 0.94435 |

The latest final AbsRel is essentially tied with the previous initialized run
on validation (+0.00014), while its RMSE is 3.03% lower. Relative to direct
BIM correction, its validation AbsRel is 26.67% lower; all 7 validation rooms
improve, with a paired room-bootstrap 95% CI of [-0.03644, -0.02064] for
`new - direct` room AbsRel.

#### Test, 1,641 frames / 7 rooms

| Variant | Output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---|---:|---:|---:|---:|
| Raw DA3 | Raw | 0.30123 | 0.67323 | 0.83485 | 0.26437 |
| Universal scale | Scale | 0.07752 | 0.13938 | 0.31541 | 0.93710 |
| Direct BIM correction | Direct | 0.07815 | 0.13891 | 0.31350 | 0.93740 |
| Existing public refiner | Final | 0.06689 | 0.11761 | 0.30823 | 0.94295 |
| Previous attention + direct scale loss | Final | 0.06578 | 0.11763 | 0.30614 | **0.94438** |
| Bounded MLP + scratch curriculum | Scale | 0.07006 | 0.12403 | 0.31318 | 0.94223 |
| Bounded MLP + scratch curriculum | Final | **0.06442** | **0.11498** | **0.29802** | 0.94380 |

The latest final output lowers test AbsRel by 17.57% versus direct BIM,
3.70% versus the existing public refiner, and 2.07% versus the preceding
attention + direct-scale-loss run. It improves 6/7 rooms against direct BIM;
the formal paired room-bootstrap `new - direct` 95% CI is
[-0.02218, -0.00934]. Post-hoc paired room comparisons also favor the latest
model over the existing refiner in 5/7 rooms (95% CI
[-0.00549, -0.00028]) and over the preceding attention run in 6/7 rooms
(95% CI [-0.00444, -0.00030]). These test comparisons remain diagnostic
because the split was already revealed.

On the complete test split, the MLP log-scale correction has mean 0.02403,
5th/95th percentiles 0.01900/0.02782, and maximum 0.04249; no frame reaches
the +/-0.045 near-saturation threshold. The scale head receives a mean
fallback-gate attention weight of 0.70947. After that gate, the MLP's mean
effective log-scale shift is 0.01724 (about +1.74% multiplicatively). Thus the
branch learned a small systematic correction rather than exhausting its
bound. This is an inference-only diagnostic and uses no GT.

This run changes both initialization/training curriculum and the bounded MLP,
so it establishes the performance of the combined recipe, not the isolated
causal contribution of the MLP. Per the agreed scope, no additional ablation
or seed sweep was used to select it.

Formal evaluator commands are the same as above, with
`configs/stanford_area1_attentive_scale_mlp_scratch.yaml`, the accepted
checkpoint, and output directories
`results/stanford_area1/attentive_scale_mlp_scratch_{val,test}`.

Key SHA256 identities:

- configuration: `dd479125eb72dd5f1d376ac01c3d56ba46a04e82f350806e7372e3a8fd75ef30`;
- accepted checkpoint: `a98e72e9fa4b80146bc4a464147bbc689d60a2d7c511d467d741bc13b505e6cb`;
- training history / run state: `31aee50226038755a60e41ca4a8383cc461200fc5557e4cb05e747a0e226ddcc` /
  `125dd85983e5461660444a1a059c6775a76081bb243c1f4a18a28c0169b16bf2`;
- validation summary / CSV: `ffd4605e1074a65a398bbf8b87be9d51ee449a15668fe7585a9f32bd71f0c726` /
  `2503a7237e7d2e6d96ad6214afd1e771b51f42651b7dcc202fa5d88c8b3629b7`;
- test summary / CSV: `09c8d3cf739c695a36823d13f7304c41ed53fc71f712b6d632fe3d3936fbfd18` /
  `2c5a8ac2178e0fa3893b43147d7d22df38047d2cdb4f2ac9f82f0ab4c68020a7`.

## Pixel residual distribution after scale correction

To decide whether the spatial refiner should predict an additive metric
residual or a multiplicative log residual, a validation-only diagnostic was
run on the retained no-DA3-feature bounded-MLP scale head. It samples up to
4,096 fixed-support pixels independently from each of all 1,673 validation
frames: 6,852,608 pixels over 7 rooms. The model first predicts its image-level
scale without GT. GT is read only afterward to measure
`GT - scaled` and `log(GT / scaled)`; test is prohibited by the CLI.

```bash
python scripts/analysis/analyze_scale_residual_distribution.py \
  --config configs/stanford_area1_attentive_scale_mlp_scratch.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_mlp_scratch/accepted.pt \
  --split val \
  --output results/stanford_area1/scale_residual_distribution_val \
  --device cuda --batch-size 8 --pixels-per-frame 4096 --seed 42
```

For a purely additive residual, the power law
`mean |GT-scaled| = c * depth^p` should have `p=0`; for a purely proportional
residual it should have `p=1`. The learned scale gives `p=0.404` globally.
The equal-room mean is `0.381`, with room-cluster bootstrap 95% CI
`[0.142, 0.719]`; only 1/7 rooms is closer to 1 than to 0. The frozen universal
scale independently gives `p=0.478`, so the mixed behavior is not created by
the learned scale head.

The best simple model for bin-level mean absolute metric error is

```text
mean |GT - scaled| ~= 0.0566 m + 0.0402 * GT depth
```

with `R^2=0.891` and normalized RMSE `0.130`. A pure proportional-through-zero
model has normalized RMSE `0.265`; a pure additive constant model has `0.395`.
Thus the empirical residual contains both an approximately 5.7 cm floor and a
distance-dependent component. It is neither purely additive nor purely
multiplicative.

The distribution is also depth- and content-dependent. Learned-scale mean
absolute error is 0.105 m at 0.2--0.5 m, falls to 0.087 m at 0.75--1.0 m, then
rises to 0.266 m at 4--5 m. Mean absolute log ratio falls from 0.207 in the
nearest bin to about 0.052--0.063 beyond 2 m. Signed correction is -0.075 m in
the nearest bin (prediction too deep) and +0.097 m at 4--5 m (prediction too
shallow), which a second global scale cannot remove.

| Sampled subset | Pixels | MAE (m) | Power exponent `p` |
|---|---:|---:|---:|
| BIM-consistent | 3,597,195 | 0.096 | 0.328 |
| BIM foreground conflict | 1,557,041 | 0.163 | 0.512 |
| Furniture | 693,614 | 0.179 | 0.762 |
| BIM no-hit | 596,446 | 0.311 | 1.061 |

Structural/BIM-consistent residuals are more additive-like, whereas furniture
and BIM-missing regions are substantially more proportional. The global
histogram is sharply peaked but very heavy-tailed: learned-scale absolute
metric correction has median/90th/95th/99th percentiles
`0.062/0.295/0.513/1.502 m`. A single Gaussian, fixed-meter residual, or
fixed-percentage residual is therefore an inadequate noise model.

This evidence supports a mixed heteroscedastic correction, for example

```text
delta_D = (a + b * D_scaled) * bounded_residual(x, y)
D_final = positive_clamp(D_scaled + delta_D)
```

or parallel bounded additive and log-residual experts. It does not support
hard-wiring `low=multiplicative` and `detail=additive`: furniture/no-hit pixels
show the opposite of that simple assignment. The scale/refiner redesign should
estimate `a,b` on train only and use validation only for checkpoint selection.

Artifacts are in `results/stanford_area1/scale_residual_distribution_val/`.
Summary / bin CSV / depth plot / histogram SHA256 are respectively
`dc6f1bcb502a8df5c344a43ff8eb89e4bd6fcba63b08bdbc471d0c2e6fa09c7e`,
`46c31f905a01c81c46ac1e0a110ef66916639ef42a00041053b92fff4719a2d2`,
`9b779464e5a82409200dc8fd7ae3cebf2906c249c792fad2623587a46d59169b`, and
`786ff0bd07d485c4481299a9ce55024cab66f6e2fde07d01b6217614783a68c6`.

## Hybrid proportional and additive residual

### Architecture

The fifth candidate tests the mixed residual suggested by the validation-only
distribution analysis:

\[
D_{\mathrm{prop}}=D_{\mathrm{scale}}\exp(r_{\mathrm{low}}+r_{\mathrm{detail}}),
\qquad
D_{\mathrm{final}}=D_{\mathrm{prop}}+\Delta D_{\mathrm{add}}.
\]

The attention scale head returns to the stronger no-DA3-token design. Frozen
DA3 layer-11/layer-23 features are retained only in the spatial refiner. The
additive branch receives the final decoded feature, log metric anchor, and
proportional log residual. It uses one convolutional block, one residual block,
and a zero-initialized 1x1 output:

\[
\Delta D_{\mathrm{add}}=0.20\tanh(h_{\mathrm{add}})\ \mathrm{m}.
\]

The shared decoded inputs are detached at this branch, so its teacher loss
cannot turn the proportional refiner into a hidden additive path. The ordinary
final-depth loss still supervises the complete expression end to end during
the joint stage. The model has 3,741,979 trainable task parameters; DA3 remains
frozen and is represented by the existing verified feature cache.

Training-only auxiliary supervision uses

\[
\Delta D^*=\operatorname{clamp}(D_{\mathrm{GT}}-D_{\mathrm{prop}},-0.20,0.20),
\]

with SmoothL1 beta 0.02 m and weight 0.50. Additive L1 regularization has
weight 0.005 and the squared cosine with detached proportional depth has weight
0.01. These terms use GT only during training; inference inputs are unchanged.

### Scratch training

The task network is initialized from scratch and uses one AdamW optimizer and
one cosine schedule for all 18 epochs:

1. epochs 1--3: attention scale only;
2. epochs 4--12: proportional low/detail refiner only;
3. epochs 13--15: additive head only;
4. epochs 16--18: low-learning-rate joint training.

Resolution is 504 x 504, physical batch size is 8, gradient accumulation is 2,
and the effective batch size is 16. Initial learning rates are 8e-5 for the
scale head, 4e-5 for the proportional refiner, and 1.6e-4 for the additive
parameter group. The run completed 7,879/7,884 optimizer attempts; five AMP
overflows were safely skipped. Peak allocated GPU memory was 12.92 GiB. Human
epoch 17 was selected strictly by validation final-output AbsRel; epoch 18 was
not selected.

```bash
python scripts/model/train.py \
  --config configs/stanford_area1_hybrid_additive.yaml \
  --device cuda
```

### Same-checkpoint residual attribution

`Proportional` and `proportional + additive` below come from the same frozen
checkpoint. Both apply the model's identical `1e-3--10 m` output safety bound.
The benchmark's `0.2--5.0 m` interval defines valid GT support and does not clip
predictions. This distinction is necessary: clipping only the proportional
branch to 5 m produces an invalid optimistic comparison.

#### Validation, 1,673 frames / 7 rooms

| Method | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Raw DA3 | 0.27710 | 0.62840 | 0.79206 | 0.32894 |
| Universal scale | 0.08644 | 0.17422 | 0.39536 | 0.91533 |
| BIM-direct | 0.08710 | 0.17121 | 0.39455 | 0.91312 |
| Learned attention scale | 0.06988 | 0.14056 | 0.36083 | 0.93621 |
| Hybrid proportional | 0.06131284 | 0.12391499 | 0.31443954 | 0.95080094 |
| Hybrid proportional + additive | **0.06130833** | **0.12390198** | **0.31443435** | 0.95080030 |

The additive branch changes validation AbsRel by only -0.00000451, a 0.007%
relative reduction. It is therefore practically neutral. The complete hybrid
improves AbsRel by 1.27% over the preceding DA3-feature final output (0.06209),
but its RMSE is worse (0.31443 versus 0.30842).

#### Test, 1,641 frames / 7 rooms

| Method | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Raw DA3 | 0.30123 | 0.67323 | 0.83485 | 0.26437 |
| Universal scale | 0.07752 | 0.13938 | 0.31541 | 0.93710 |
| BIM-direct | 0.07815 | 0.13891 | 0.31350 | 0.93740 |
| Existing public refiner | 0.06689 | 0.11761 | 0.30823 | 0.94295 |
| Previous frozen-feature final | 0.06244 | **0.10780** | **0.28166** | 0.94391 |
| Hybrid proportional | 0.06242490 | 0.10960736 | 0.28667718 | 0.94520012 |
| Hybrid proportional + additive | **0.06235036** | 0.10953104 | 0.28663240 | **0.94525070** |

The additive branch reduces test AbsRel by 0.00007453, or 0.119%. The complete
hybrid improves AbsRel by 20.21% over BIM-direct, 6.78% over the public
refiner, 3.21% over bounded-MLP scratch, and only 0.14% over the preceding
frozen-feature model. Its MAE and RMSE are 1.61% and 1.77% worse than that
preceding model. Consequently the experiment supports the feature-routing and
joint-training recipe as an AbsRel candidate, but does not establish a useful
effect from the additive head itself. It should remain a diagnostic until a
new blind area confirms the trade-off.

Formal evaluator outputs:

- configuration: `15d08c8afc417b6761f81d87e5ad9699a1975572d8918efe6450e1d01710c4e2`;
- accepted checkpoint: `7d8ff8748b43f4f4cebcea0f1ec1dc8b53231a8dfab1e72758b2f41220e83cb6`;
- training history / run state: `36a90a5a34c613b9fe9136aefc77fcdc8b7911f7a552fd79235bb7bc6361c1dd` /
  `04f02bd4745d626edb435def8a05d1a2bd542a2d4ac54fc4f30396f5556aa6ce`;
- validation summary / CSV: `3996851e853c807dc5aa133d84dbeb5ab63657ca675afb06499be802ee4f5cc7` /
  `049cc44a3d67147d1de825bb7187131d60a43bece7a48eaf24066f1071e4ca78`;
- test summary / CSV: `64bdbbcb0f8022b6fbcd54fbe373da0887fb47620f0ba5c94920915f3aa6cde2` /
  `531bd67283df1aa4b27806ce61c2f47e467e57c5338deb7225869c0307aaa972`.

## Rollback and sequential scale/low/detail attribution

The active research candidate is rolled back to the preceding no-additive
checkpoint `stanford_area1_attentive_scale_da3_features/accepted.pt`. This is
the last completed model before the additive experiment and has better test
MAE/RMSE than the hybrid. The additive implementation and results remain only
as a reproducible negative diagnostic.

The formal evaluator now reconstructs three predictions from one frozen
forward pass:

\[
\begin{aligned}
D_0 &= D_{\mathrm{scale}},\\
D_1 &= D_{\mathrm{scale}}\exp(r_{\mathrm{low}}),\\
D_2 &= D_{\mathrm{scale}}\exp(r_{\mathrm{low}}+r_{\mathrm{detail}}).
\end{aligned}
\]

The frame residual is disabled in this attention-scale model. Every stage uses
the same fixed GT support and the same `1e-3--10 m` model-output safety bound.
This is a sequential attribution: the detail row measures its incremental
effect after low, not a detail-only network.

### Validation, 1,673 frames / 7 rooms

| Cumulative output | AbsRel | MAE (m) | RMSE (m) | delta1 | Incremental AbsRel change |
|---|---:|---:|---:|---:|---:|
| Scale | 0.06959680 | 0.13941597 | 0.35737642 | 0.93566826 | -- |
| Scale + low | 0.06217133 | 0.12404750 | **0.30837328** | 0.94615063 | -0.00742547 (-10.67%) |
| Scale + low + detail | **0.06209457** | **0.12400717** | 0.30842432 | **0.94615475** | -0.00007676 (-0.12%) |

Low accounts for 98.98% of the total validation AbsRel reduction after scale;
detail accounts for 1.02%. Detail slightly worsens RMSE and also slightly
worsens validation furniture and BIM-no-hit AbsRel, while improving conflict
and BIM-consistent pixels.

### Test, 1,641 frames / 7 rooms

| Cumulative output | AbsRel | MAE (m) | RMSE (m) | delta1 | Incremental AbsRel change |
|---|---:|---:|---:|---:|---:|
| Scale | 0.07144184 | 0.12544790 | 0.31188626 | 0.93979523 | -- |
| Scale + low | 0.06284588 | 0.10845274 | 0.28180913 | 0.94370244 | -0.00859596 (-12.03%) |
| Scale + low + detail | **0.06243524** | **0.10779922** | **0.28165639** | **0.94391001** | -0.00041064 (-0.65%) |

Low accounts for 95.44% of the total test AbsRel reduction after scale and
detail for 4.56%. Thus `r_low` is the essential spatial correction. `r_detail`
is a small complementary term: positive on aggregate test, but much less
stable across validation subsets. Removing detail would simplify the head but
would give up a reproducible 0.65% relative test AbsRel improvement; this is
not enough evidence to redesign it without a new blind validation domain.

Stage-ablation summary / CSV SHA256:

- validation: `44599787b2ea2dd60ff440c9a3d4e6ccf6a7240dc4e175dec9c7e97dab3cb4a0` /
  `cba14929a6d9673ada3bbc8321ef0b9ebe38554d4c5112bd7d6a6cede87ba215`;
- test: `f8fb917bb0a31bd23ce1b946446b52858055b7e10f1f5ded698cbda46e5e0321` /
  `df11d30076432b2f2378e6d54c84c097a616c9d3bfcc54189364d723ba91c776`.

## Unbounded hit-only BIM prior retraining

The fixed-envelope prior was regenerated once under a deliberately broader
validity rule:

- retain wall/floor/ceiling/column/beam/door/window while still excluding
  furniture, proxy geometry, and MEP;
- mark BIM valid for every finite positive ray hit;
- do not apply the `0.2--5.0 m` interval to BIM validity;
- retain the unchanged `0.2--5.0 m` GT/loss/evaluation support.

The old prepared dataset and results remain immutable. The new artifacts use
`data/processed/stanford_area1_hit_only_504`, protocol
`global-area-hit-only-fixed-envelope-v2`, and configuration
`configs/stanford_area1_attentive_scale_da3_features_hit_only.yaml`. RGB,
raw DA3 depth, GT depth, GT mask, room assignments, and frozen DA3 layer-11/23
features are exactly unchanged for all 10,327 frames.

### Prior distribution change

| Prior | Valid pixels | Coverage | >5 m among valid | <0.2 m among valid |
|---|---:|---:|---:|---:|
| Legacy bounded core envelope | 2,329,973,343 | 88.8210% | excluded | excluded |
| Hit-only all envelope | 2,621,636,015 | 99.9395% | 4.8156% | 0.7492% |

The new rule adds 291,662,672 pixels, or 11.1185% of the full image
population. Hit depth spans 0.00452--39.0 m. This is a coverage increase, not
evidence that the added correspondences are reliable scale observations.

### Controlled training

The no-additive attention-scale + low/detail network was trained from fresh
task initialization. DA3 remained frozen. The schedule and data split were
unchanged: 3 scale-only, 9 refiner-only, and 3 low-learning-rate joint epochs;
batch size 8, accumulation 2, and 7,013/1,673/1,641 train/validation/test
frames. Peak allocated CUDA memory was 11.02 GiB. Human epoch 12 was selected
strictly by validation final-output AbsRel (`0.06306101` in the training
history); human epoch 15 was not selected.

#### Validation, 1,673 frames / 7 rooms

| Output | Legacy prior AbsRel | Hit-only AbsRel | Hit-only MAE (m) | Hit-only RMSE (m) | Hit-only delta1 |
|---|---:|---:|---:|---:|---:|
| Raw DA3 | 0.277104 | 0.277104 | 0.628398 | 0.792064 | 0.328936 |
| Robust BIM-direct | 0.087100 | 0.124111 | 0.237813 | 0.514650 | 0.846893 |
| Learned scale | 0.069597 | 0.072255 | 0.143481 | 0.364915 | 0.930964 |
| Scale + low | 0.062171 | 0.065675 | 0.133328 | 0.316151 | 0.942606 |
| Scale + low + detail | **0.062095** | 0.063062 | **0.125689** | **0.313415** | **0.943760** |

The hit-only final output is 1.56% worse than the legacy learned model, but
49.19% better than its own hit-only BIM-direct comparator. Low reduces
hit-only scale AbsRel by 9.11%; detail then reduces it by another 3.98%.

#### Test, 1,641 frames / 7 rooms

| Output | Legacy prior AbsRel | Hit-only AbsRel | Hit-only MAE (m) | Hit-only RMSE (m) | Hit-only delta1 |
|---|---:|---:|---:|---:|---:|
| Raw DA3 | 0.301228 | 0.301228 | 0.673234 | 0.834852 | 0.264374 |
| Robust BIM-direct | 0.078146 | 0.107449 | 0.196647 | 0.435984 | 0.882243 |
| Learned scale | 0.071442 | 0.075468 | 0.135775 | 0.332289 | 0.933664 |
| Scale + low | 0.062846 | 0.066961 | 0.120872 | 0.302624 | 0.939681 |
| Scale + low + detail | **0.062435** | 0.065852 | **0.117256** | **0.300934** | **0.939138** |

The hit-only final output is 5.47% worse than the legacy learned model. It is
still 38.71% better than its own BIM-direct comparator and passes all
all/furniture/conflict pixel-, frame-, and room-level direct-comparator safety
checks. Against the legacy learned model it improves only 2/7 validation and
2/7 test rooms. A post-hoc room bootstrap of `hit-only - legacy` AbsRel gives
validation CI `[-0.00152, +0.00887]` and test CI
`[-0.00417, +0.00499]`; both cross zero, so seven rooms do not establish a
room-level difference despite the worse pixel-micro point estimates.

The experiment is a negative protocol result: near-complete BIM coverage
dilutes correspondence precision. The learned scale head recovers most of the
damage, and the spatial refiner remains essential, but retraining does not
surpass the bounded core-envelope model. The broader hit-only rule is
therefore retained as a diagnostic rather than promoted as the recommended
prior. A better next design would keep hit-only geometry available as context
while learning or measuring correspondence reliability before those pixels
contribute to scale estimation.

Reproduction:

```bash
python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only.yaml \
  --device cuda
python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_da3_features_hit_only/accepted.pt \
  --split val \
  --output results/stanford_area1/attentive_scale_da3_features_hit_only_val \
  --batch-size 8 --bootstrap-repetitions 10000 \
  --bootstrap-seed 42 --inference-seed 42 --device cuda \
  --allow-unverified-robust-comparator
```

The opt-out is recorded because the previous train-only robust comparator
receipt is cryptographically bound to the legacy manifest. Comparator
parameters were not retuned, so this remains an explicitly exploratory
protocol comparison rather than a replacement formal claim.

Artifact SHA256:

- configuration: `b8baa61ae70620bf8afc3bd75e2d209ed106128a944909f140d4dbf9b2cf0e8f`;
- accepted checkpoint: `972e056187c36129bde6ff4d0baae774ca59e65285db6b5926ab8861f787410e`;
- training history / run state: `b9f9e9fca22db464f544ade35e93d981e52854fb88b365e4d251ac378940880d` /
  `a52d2e38c4f0f83d7e4b7945e2e1b14284302877f50e3cf6088a04a8c93e59be`;
- validation summary / CSV: `86258b2d1a52bc5a32a6fbc6724cea52eb55aaeb21b63c6b80d18bea63a03d1f` /
  `cf2f6ed78b457c63466236bc0888b5af03a25547902e166c133dbe8dfa2a67c0`;
- test summary / CSV: `bb6e6e3a602d38ca71d4568fef3f28c139bb2d0e834cd17d3a20120907abd317` /
  `9f913601bf0740b3e09c089e6b7214f70760c709d52a33a8380d80073f3cd46e`.

### All-valid official GT diagnostic

The evaluator also supports an explicitly non-primary full-range diagnostic.
`--depth-support all-valid` reloads the official regular-view uint16 z-depth,
keeps every positive value, rejects only `0` and the `65535` invalid sentinel,
and applies no metric depth cutoff. It does not change model inputs, scale
estimation, or the checkpoint. The network still has its trained architectural
safety bound of 10 m (`2 * data.max_depth`), so official GT beyond 10 m measures
real out-of-training-range behavior rather than an extrapolation-tuned model.

| Split | Support | Valid pixels | Observed GT range | <0.2 m | >5.0 m |
|---|---|---:|---:|---:|---:|
| Validation | configured | 364,913,264 | 0.2--5.0 m | excluded | excluded |
| Validation | all-valid | 422,186,615 | 0.299--51.605 m | 0 | 57,273,351 (13.566%) |
| Test | configured | 404,367,334 | 0.2--5.0 m | excluded | excluded |
| Test | all-valid | 413,756,559 | 0.156--20.562 m | 185,805 (0.045%) | 9,203,420 (2.224%) |

Pixel-micro all-valid results for the same hit-only checkpoint are:

| Split | Output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---|---:|---:|---:|---:|
| Validation | Raw DA3 | 0.283990 | 0.880281 | 1.324833 | 0.317410 |
| Validation | Robust BIM-direct | 0.122034 | 0.325024 | 0.864076 | 0.845214 |
| Validation | Learned scale | 0.076307 | 0.237851 | 0.691676 | 0.921152 |
| Validation | Scale + low | 0.075252 | 0.272762 | 0.780194 | 0.920450 |
| Validation | Scale + low + detail | **0.072138** | **0.260275** | **0.771848** | **0.922364** |
| Test | Raw DA3 | 0.302748 | 0.710133 | 0.928631 | 0.262282 |
| Test | Robust BIM-direct | 0.108663 | 0.216459 | 0.554772 | 0.879886 |
| Test | Learned scale | 0.076768 | 0.152233 | 0.436496 | 0.930859 |
| Test | Scale + low | 0.068568 | 0.138879 | 0.425313 | 0.936480 |
| Test | Scale + low + detail | **0.067443** | **0.135015** | **0.423193** | **0.935996** |

Relative to the configured support, final AbsRel rises by 14.39% on validation
and 2.42% on test. The split difference is explained by support composition:
validation adds 15.70% more pixels and 13.57% of all its valid pixels are over
5 m, whereas test adds only 2.32% and 2.22% are over 5 m. Even under this
diagnostic, the final model reduces all-valid test AbsRel by 77.72% from raw
DA3 and by 37.93% from robust BIM-direct. These figures must not replace the
0.2--5.0 m primary benchmark because both training and checkpoint selection
used that configured interval.

Reproduction:

```bash
.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_da3_features_hit_only/accepted.pt \
  --split test --depth-support all-valid \
  --output results/stanford_area1/attentive_scale_da3_features_hit_only_all_valid_test \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --allow-unverified-robust-comparator
```

The validation/test receipts are stored in
`results/stanford_area1/attentive_scale_da3_features_hit_only_all_valid_{val,test}`.

For a controlled prior comparison, the original bounded-core checkpoint
(`outputs/stanford_area1_attentive_scale_da3_features/accepted.pt`) was then
evaluated on exactly the same all-valid GT pixels. No retraining or full-range
checkpoint selection was performed:

| Split | Bounded-core final AbsRel | Hit-only final AbsRel | Bounded-core MAE (m) | Bounded-core RMSE (m) | Bounded-core delta1 |
|---|---:|---:|---:|---:|---:|
| Validation | 0.074197 | **0.072138** | 0.280646 | 0.797840 | 0.920554 |
| Test | **0.064335** | 0.067443 | 0.126649 | 0.405005 | 0.940857 |

The bounded-core model's all-valid test AbsRel is 3.04% above its configured
0.2--5.0 m result (`0.062435`) and 4.61% below the hit-only model on the same
all-valid test support. Validation reverses the point estimate: bounded-core is
2.85% worse than hit-only. Thus the full-range test favors retaining the
bounded prior, but the room-split reversal prevents claiming a universal
full-range advantage from these two diagnostic splits.

The bounded-prior receipts are stored in
`results/stanford_area1/attentive_scale_da3_features_bounded_prior_all_valid_{val,test}`.

### Full-depth supervised retraining

**Release decision:** this validation-selected checkpoint is the recommended
public Area_1 model for official-all-valid regular-view depth. The immediately
following reliability-gated candidate failed to improve it and was rolled
back.

The preceding all-valid experiment changed only evaluation support. To test the
actual full-depth objective, the task network was subsequently retrained from
scratch with the same room split, RGB, frozen cached DA3 predictions/features,
hit-only BIM prior, attention-scale head, and low/detail refiner. The only
protocol changes are:

- `data.ground_truth_support: official_all_valid` reloads official regular-view
  uint16 z-depth for every sample during both training and validation;
- all positive depths except raw values `0` and `65535` enter scale supervision,
  residual losses, validation checkpoint selection, and final metrics;
- `model.output_max_depth_m: 128.0` replaces the historical 10 m safety clamp,
  so predictions can cover the complete official encoding. The inherited
  `data.min_depth/max_depth` values remain legacy preparation/visualization
  metadata (and the default output-cap source when no override is present);
  they do not mask GT in this run.

This is a fresh task-network run, not a fine-tune of the 0.2--5.0 m checkpoint.
DA3 remains frozen. Training uses batch size 8, gradient accumulation 2, 504 px
inputs, and the unchanged 3 scale-only + 9 refiner-only + 3 low-LR joint epoch
schedule. Human epoch 15 is the validation-selected checkpoint; peak allocated
GPU memory was 12.822 GiB. The full test was already revealed by earlier
iterations, so the checkpoint may be released for reproducibility and use, but
test numbers remain post-hoc rather than a new blind claim.

Pixel-micro results on the identical official all-valid support are:

| Split | Output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---|---:|---:|---:|---:|
| Validation | Raw DA3 | 0.283990 | 0.880281 | 1.324833 | 0.317410 |
| Validation | Robust BIM-direct | 0.122034 | 0.325024 | 0.864076 | 0.845214 |
| Validation | Learned scale | 0.076347 | 0.233052 | 0.684443 | 0.921965 |
| Validation | Scale + low | 0.069401 | 0.213574 | 0.620449 | 0.930863 |
| Validation | Scale + low + detail | **0.068605** | **0.210650** | **0.618012** | **0.931023** |
| Test | Raw DA3 | 0.302748 | 0.710133 | 0.928631 | 0.262282 |
| Test | Robust BIM-direct | 0.108663 | 0.216459 | 0.554772 | 0.879886 |
| Test | Learned scale | 0.077944 | 0.152493 | 0.435735 | 0.930607 |
| Test | Scale + low | 0.067464 | 0.132510 | 0.413857 | **0.938357** |
| Test | Scale + low + detail | **0.067407** | **0.131672** | **0.413329** | 0.937845 |

After the release checkpoint had already been frozen, the same deterministic
evaluator was extended to accept `--split train` and run once on all 7,013
optimization images. This is a fit diagnostic only: it uses inference mode
without augmentation or parameter updates, and GT enters only metric
calculation after prediction. It was not used to select an epoch, tune a
threshold, or change the released model.

| Train diagnostic output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Raw DA3 | 0.281670 | 0.670579 | 1.113652 | 0.341935 |
| Robust BIM-direct | 0.173598 | 0.416509 | 1.033740 | 0.759010 |
| Learned scale | 0.089397 | 0.210565 | 0.708067 | 0.919732 |
| Scale + low | 0.067586 | 0.165058 | 0.629919 | 0.950122 |
| Scale + low + detail | **0.066722** | **0.162565** | **0.628261** | **0.950469** |

Final train AbsRel is lower than validation/test by `0.001883/0.000685`
(`2.82%/1.03%` relative), a small fit-to-held-out gap rather than evidence of
severe overfitting. MAE and RMSE are not ordered across the three splits
because their scene and depth distributions differ: train contains 1.772
billion valid pixels spanning 0.234--49.977 m, validation reaches 51.605 m,
and test reaches only 20.562 m. Within train, low-frequency refinement reduces
scale-only AbsRel by 24.40%, while detail contributes another 1.28%.

Against the closest controlled baseline--the same hit-only network trained on
0.2--5.0 m but evaluated on these exact all-valid pixels--full-depth training
changes final validation/test AbsRel from `0.072138/0.067443` to
`0.068605/0.067407` (relative reductions 4.90%/0.054%). Validation MAE/RMSE
fall 19.07%/19.93%; test MAE/RMSE fall 2.48%/2.33%, and test delta1 rises by
0.185 percentage points. Thus full-depth supervision clearly helps validation
and meter-error metrics, but its test AbsRel gain is practically neutral. It
also does not overturn the previously observed test advantage of the bounded
BIM-prior checkpoint (`0.064335` all-valid AbsRel); GT support and BIM-prior
quality are separate factors.

Reproduction:

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --device cuda

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth/accepted.pt \
  --split val --depth-support all-valid \
  --output results/stanford_area1/attentive_scale_da3_features_hit_only_full_depth_val \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --allow-unverified-robust-comparator
```

Repeat the evaluation with `--split test` and the `_test` output path. The
post-training fit diagnostic can be reproduced by replacing the split with
`train` and the suffix with `_train`; do not use that output for model
selection. The accepted checkpoint SHA256 is
`f330a987d638482636e225ebdf326612209fa672ea3c5c77a11049f05b655349`;
validation/test summary SHA256 values are
`dea8f01a5ede9848714a6f6283404a253a62b72dd030119f98fc878b9d2ea8a7` and
`f5d006e72a4055d5275255d902618ae4df8fbc64600148dbcac1b985b9f0995a`.
The train diagnostic summary/per-frame SHA256 values are
`fb9c88b6a00e26c2018bb4c881cbd67dad5b32d221be4cdedbf31d2836c0a730` and
`01b04be0f8d991e2ebdfab0cce07f46846040c9d05bf6b90ecc5e7edcbfdb22b`.

### Train-selected fixed BIM/DA3 quantile (negative result)

The scale-only comparator was also isolated from the local BIM-direct field.
On the 1,641-frame official-all-valid test, the existing
`log_upper_cap_v1` scale alone gives AbsRel/MAE/RMSE/delta1
`0.108401/0.217625/0.553864/0.879748`. Pure q45 gives
`0.109843/0.214767/0.530377/0.885148`: the robust cap improves AbsRel by
1.31%, but q45 is better on the other three metrics. BIM-direct's local field
changes robust scale AbsRel only from `0.108401` to `0.108663` and is therefore
not responsible for the scale-only result.

To ask whether a different fixed quantile is better, a new locked selector
searched q=0.05--0.95 at step 0.01 using all 7,013 train frames and the same
official-all-valid pixel-micro AbsRel headline objective. Runtime candidates use
only the hit-only BIM/DA3 ratios in (0.2, 5.0); GT is used only to score the 91
candidates on train. The selected q56 was then frozen in a receipt before one
test execution. No alternative quantile was tested after seeing that result.

| Split/method | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Train, current robust scale | 0.176655 | -- | -- | -- |
| Train, pure q45 | 0.143765 | -- | -- | -- |
| Train, selected q56 | **0.137321** | -- | -- | -- |
| Test, current robust scale | **0.108401** | 0.217625 | 0.553864 | 0.879748 |
| Test, pure q45 | 0.109843 | **0.214767** | **0.530377** | **0.885148** |
| Test, train-selected q56 | 0.127587 | 0.244577 | 0.577203 | 0.867391 |

Q56 reduces train AbsRel by 4.48% relative to q45, but regresses test AbsRel
by 17.70% relative to the current robust estimator and improves only 2/7 test
rooms. The paired equal-room `q56 - robust` AbsRel interval is
`[-0.01968, 0.02736]`. Train room balancing independently preferred q54, and
leave-one-train-room-out winners were q52/q53/q54/q55/q56 in
3/10/14/2/1 rooms. This disagreement already warned that the pixel-micro q56
optimum was driven by room weighting. Q56 also raises mean scale from 1.336 on
train to 1.492 on test, amplifying the split-dependent BIM/DA3 ratio shift.

The correct conclusion is not that q45 is globally optimal. It is that a
single train-optimized quantile is not sufficiently transferable on these
room splits. Keep the current robust rule for the published protocol; a q54
room-balanced hypothesis must be evaluated only on a new blind area, not by
trying another value on this already-revealed test.

Reproduction:

```bash
.venv/bin/python scripts/analysis/search_stanford_scale_quantile.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --split train \
  --output results/stanford_area1/fixed_scale_quantile_full_depth_train/selection.json

.venv/bin/python scripts/analysis/search_stanford_scale_quantile.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --split test \
  --selection-receipt results/stanford_area1/fixed_scale_quantile_full_depth_train/selection.json \
  --output results/stanford_area1/fixed_scale_quantile_full_depth_test
```

### Full-depth GT-oracle DA3 scale diagnostic

To separate DA3 relative-shape error from scale-estimation error, an explicitly
privileged diagnostic computes the exact AbsRel-optimal positive scalar for
each frame from all official all-valid GT pixels, multiplies the entire cached
raw DA3 map by that one scalar, and only then evaluates the frozen subsets.
This is intentional GT leakage and is therefore an oracle capability
diagnostic, never an inference baseline. It is stronger and less noisy than
estimating the scalar from a random GT sample.

| Split / subset | Raw DA3 | One frame-global GT scale | Current learned final |
|---|---:|---:|---:|
| Validation / all | 0.283990 | **0.061651** | 0.068605 |
| Validation / furniture | 0.300145 | **0.083774** | 0.087571 |
| Validation / BIM foreground conflict | 0.287560 | **0.096926** | 0.109202 |
| Validation / BIM consistent | 0.274981 | 0.036994 | **0.035622** |
| Validation / BIM no-hit | 0.284433 | **0.061604** | 0.091514 |
| Test / all | 0.302748 | **0.058849** | 0.067407 |
| Test / furniture | 0.289866 | **0.067407** | 0.075306 |
| Test / BIM foreground conflict | 0.294726 | **0.098664** | 0.127180 |
| Test / BIM consistent | 0.297525 | 0.029546 | **0.026939** |
| Test / BIM no-hit | 0.340841 | **0.031353** | 0.743506 |

The exact frame-global oracle scales have validation/test means
`1.40985/1.45161`, medians `1.36264/1.40334`, and ranges
`0.83015--2.37435` / `0.81941--2.65385`. Thus scale error explains most of
raw DA3's apparent failure: on test, one perfect scalar removes 80.56% of raw
DA3 AbsRel and leaves a `0.05885` relative-shape result. The current learned
pipeline is 0.00856 AbsRel above that oracle overall. Conversely, its BIM-aware
local refinement already beats scalar-only DA3 in the BIM-consistent subset.

A second, still looser oracle computes a different GT-optimal scalar for every
frame and every evaluated subset. Test AbsRel becomes:

| Subset-specific oracle | AbsRel | Supported frames |
|---|---:|---:|
| BIM consistent | 0.026837 | 1,638 |
| Furniture | 0.053114 | 1,431 |
| Non-structural | 0.061032 | 1,580 |
| BIM foreground conflict | 0.080358 | 1,537 |
| BIM no-hit | 0.021909 | 80 |

The difference between frame-global and subset-specific oracles shows that a
single scalar cannot fully reconcile structural and foreground regions. The
remaining `0.08036` conflict error after even a conflict-specific optimal
scalar is genuine DA3 relative-shape/occlusion error; however, the much larger
gap from the current learned `0.12718` also shows substantial avoidable scale
and BIM-fusion error. The no-hit failure is severe but covers only 57,407 test
pixels (0.014%) and is not the main aggregate bottleneck.

## Reliability-gated full-depth candidate (negative result)

The next candidate tested whether the full-depth model could learn where BIM
ratios are trustworthy without an explicit semantic classifier. It changes
three coupled components while retaining the exact room split, official
all-valid GT support, hit-only BIM, frozen DA3 predictions/features, 504 px
input, batch size 8, gradient accumulation 2, seed 42, and 3/9/3 staged
schedule:

1. the scale head exposes its exact post-dropout token distribution and receives
   a train-only target based on each BIM/DA3 log-ratio's distance from the
   frame's AbsRel-optimal GT scale;
2. the old attention-entropy regularizer decays linearly to zero over the three
   scale-only epochs, allowing attention to become sparse after its warm start;
3. RGB-aware gates control BIM feature injection at all four refiner scales,
   and the detached predicted BIM-reliability probability gates only
   `r_detail` (floor 0.10). `r_low` remains available everywhere.

GT is used only to construct losses on the training split. Validation and test
scale are produced entirely by the network; neither oracle scale nor GT enters
inference. The complete fresh run used 7,013/1,673/1,641 train/validation/test
frames and peaked at 13.376 GiB allocated GPU memory. Validation selected human
epoch 12, before the final low-LR joint stage; joint epochs 13--15 did not beat
its `0.069280` training-loop validation AbsRel.

The independent evaluator reloads official z-depth and uses the same all-valid
pixel support as the public full-depth release:

| Split | Output | Previous full-depth | Reliability-gated | Relative change |
|---|---|---:|---:|---:|
| Validation | Scale | 0.076347 | 0.076691 | +0.45% |
| Validation | Scale + low | 0.069401 | 0.069871 | +0.68% |
| Validation | Final | **0.068605** | 0.069279 | +0.98% |
| Test | Scale | 0.077944 | **0.076740** | -1.54% |
| Test | Scale + low | **0.067464** | 0.069145 | +2.49% |
| Test | Final | **0.067407** | 0.068837 | +2.12% |

Lower is better; a positive relative change is a regression. The scale-only
test result improves, so directly supervising attention is promising. The
RGB/reliability-gated refiner, however, loses more than the scale head gains.
On test, the low branch reduces the new scale error by 9.90%, versus 13.44% in
the previous model. Detail gating makes the final detail increment larger
(0.446% versus 0.085%), but it cannot recover the weaker low correction.

Final pixel-micro metrics are:

| Split | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Validation, previous | **0.068605** | **0.210650** | **0.618012** | **0.931023** |
| Validation, reliability-gated | 0.069279 | 0.213985 | 0.623480 | 0.928973 |
| Test, previous | **0.067407** | **0.131672** | **0.413329** | **0.937845** |
| Test, reliability-gated | 0.068837 | 0.135740 | 0.417432 | 0.937092 |

The subset result also rejects the intended foreground hypothesis. Relative to
the public full-depth model, validation/test furniture AbsRel regresses
3.08%/4.58%, and foreground-conflict AbsRel regresses 0.17%/0.88%. The new
candidate improves the tiny BIM no-hit subset by 12.72%/6.26%, and improves
validation non-structural pixels by 0.85%, but these gains do not dominate the
aggregate. The test no-hit subset contains only 57,407 pixels (0.014%).

A post-hoc paired room bootstrap of `new - previous` final AbsRel gives a 95%
interval of `[-0.00308, 0.00278]` on validation and
`[0.000385, 0.002237]` on the already-revealed test. The validation room-level
difference is inconclusive; the test interval is wholly positive and only one
of seven rooms improves. Consequently this candidate is archived as a negative
diagnostic and does **not** replace the public full-depth release.

Reproduction:

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/stanford_area1_reliability_gated_full_depth.yaml \
  --device cuda

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_reliability_gated_full_depth.yaml \
  --checkpoint outputs/stanford_area1_reliability_gated_full_depth/accepted.pt \
  --split val --depth-support all-valid \
  --output results/stanford_area1/reliability_gated_full_depth_val \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --allow-unverified-robust-comparator
```

Repeat with `--split test` and the `_test` output directory. The configuration,
accepted checkpoint, history, and run-state SHA256 values are respectively
`71972b32bf04794283bd5d2c438cf029b8e147b688ddecd56e8413098d456e48`,
`bec1cfc902c4f299fea9e465c12c3c6d0c07282e2a17a58320e7ff0d23907dba`,
`750dcdbc939f5f75aeb50bad991ad47aad60bfcc4a7574cb459dc349b09507aa`, and
`28b610103d4f06902e74643c17210692f6f3762ba1a317f9c09e46863bf9704a`.
Validation/test summary SHA256 values are
`3091faf248320e43796d08e39977be03b098cb6d0f5691948e0cb5a50ef7f8c9` and
`15078365e4af91b9b32604b04513e71f8f9a54e9e55db43ec8b23832ff5d652e`.

## DA3METRIC focal-scaling audit and baseline correction

The cached `base_depth` tensor was generated by the standalone
`depth-anything/da3metric-large` head at `process_res=504`. That head predicts
depth at a canonical focal length rather than applying camera-specific metric
conversion internally. The project historically evaluated this tensor directly
and called it `raw_da3`. The correct no-GT conversion is

\[
D_{\mathrm{metric}}=D_{\mathrm{cache}}
\frac{(f_x+f_y)/2}{300},
\]

with `fx/fy` expressed at the DA3 processing resolution. The prepared samples
already store that resized intrinsic matrix, so the audit does not fit anything
to GT. It also does not run the task network or mutate caches.

| Dataset / support / split | Cached canonical AbsRel | Focal-corrected metric AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|---:|
| SLABIM, 0.2--5.0 m, test (108) | 0.199347 | **0.073729** | 0.115593 | 0.255675 | 0.938905 |
| Area_1, 0.2--5.0 m, test (1,641) | 0.301228 | **0.084433** | 0.161070 | 0.336196 | 0.940648 |
| Area_1, official all-valid, validation (1,673) | 0.283990 | **0.090705** | 0.266045 | 0.654622 | 0.926403 |
| Area_1, official all-valid, test (1,641) | 0.302748 | **0.085453** | 0.177124 | 0.435242 | 0.938046 |

SLABIM has `fx=fy=252 px` after resize, hence a fixed factor `0.84`. Area_1 has
per-frame intrinsics: full-depth test factors range from `1.09477` to `2.02612`
(mean `1.44269`), and the correction improves 1,581/1,641 frames. This range is
also consistent with the previously observed approximately 1.45 frame-oracle
scale, explaining why the old raw baseline appeared much worse than expected.

This finding changes claims against raw DA3. On Area_1 all-valid test, the
public learned final `0.067407` improves over the correct raw metric baseline by
21.12%, not 77.74%. The learned-vs-BIM-direct comparison is unchanged. Existing
task checkpoints remain bound to the canonical cache because their scale heads
were trained on that representation; multiplying their input after training is
not a valid repair. The next section records the completed consistent conversion
and scratch retraining. Pano experiments still require a per-view focal rerun
before their raw-DA3 comparisons can be used as corrected headline results.

Reproduction:

```bash
.venv/bin/python scripts/analysis/audit_da3_focal_scaling.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml \
  --split val --split test --workers 8 \
  --output results/stanford_area1/da3_focal_scaling_full_depth_audit/summary.json
```

## Focal-corrected rollback retraining

The audit was followed by the required controlled retraining. The abandoned
BIM cross-attention candidate was removed, and the task network was restored to
the preceding `attention scale + r_low + r_detail` implementation. The new
dataset option `data.apply_da3_metric_focal_scaling` applies each view's
`mean(fx,fy)/300` factor immediately after loading the cached DA3 tensor. The
corrected tensor is therefore used consistently by the deterministic BIM ratio,
the learned scale head, all losses, and the residual refiner. Cached
canonical-input scale/anchor arrays are explicitly recomputed.

The run used the existing room-disjoint split (`7,013/1,673/1,641`), 504-pixel
processing resolution, batch size 8, seed 42, frozen DA3 encoder features, and
the scratch 3/9/3 curriculum: three scale-only epochs, nine refiner-only epochs,
then three low-learning-rate joint epochs. The task network has 4,267,754
parameters. Peak CUDA allocation was 11.03 GiB in scale-only, 11.99 GiB in
refiner-only, and 12.83 GiB in joint training. No initialization checkpoint,
GT-derived inference scale, NaN, or OOM was used/observed.

Human epoch 13 was selected solely by validation final AbsRel (`0.0644479`);
epochs 14/15 were `0.0645711/0.0645649` and did not replace it. The accepted
checkpoint was then frozen before test evaluation. All metrics below use the
same official-all-valid pixel support.

| Split / output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Validation raw DA3 metric | 0.090705 | 0.266045 | 0.654622 | 0.926403 |
| Validation robust BIM-direct | 0.118152 | 0.316482 | 0.837841 | 0.849215 |
| Validation learned scale | 0.075313 | 0.231490 | 0.678366 | 0.924178 |
| Validation + `r_low` | **0.064405** | **0.198659** | **0.605855** | **0.935902** |
| Validation + `r_detail` | 0.064447 | 0.198849 | 0.606138 | 0.935868 |
| Test raw DA3 metric | 0.085453 | 0.177124 | 0.435242 | 0.938046 |
| Test robust BIM-direct | 0.110724 | 0.217921 | 0.557609 | 0.879482 |
| Test learned scale | 0.078467 | 0.153309 | 0.439284 | 0.928162 |
| Test + `r_low` | 0.066284 | 0.128949 | **0.411195** | 0.939657 |
| Test + `r_detail` | **0.066144** | **0.128792** | 0.411298 | **0.939751** |

The final output reduces AbsRel versus focal-corrected raw DA3 by 28.95% on
validation and 22.60% on the post-hoc test. It also improves over the superseded
canonical-input full-depth model (`0.068605/0.067407`) by 6.06%/1.87%. The stage
results preserve the earlier attribution: `r_low` reduces scale-only AbsRel by
14.48%/15.53%, whereas `r_detail` changes it by -0.064%/+0.212% on
validation/test. Detail is therefore retained as part of the requested rollback
architecture but is not supported as a stable independent contribution.

Reproduction:

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth_metric_da3.yaml \
  --device cuda

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth_metric_da3.yaml \
  --checkpoint outputs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth_metric_da3/accepted.pt \
  --split val --depth-support all-valid \
  --output results/stanford_area1/attentive_scale_da3_features_hit_only_full_depth_metric_da3_val \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --allow-unverified-robust-comparator
```

Repeat the evaluator with `--split test` and the `_test` output directory only
after the validation-selected checkpoint is frozen. The config, accepted
checkpoint, training history, and run-state SHA256 values are respectively
`120b0c99f8137c1e5dcf589fbd72ab519219f3061f28de1b897239410b7221d4`,
`64e154b9deeb0f3152406f1bae5bc6e19ae0fef69d4d313ccc8c8874208c8d24`,
`72c4a197a38b45f0bcee08db407176696384f4730d7edf571aff7423338bf18b`, and
`e23645da356288d96d3642ffcb8c2ada9518114faf4cd5d7b525f02ed9750df7`.
Validation/test summary SHA256 values are
`bb46f8dd05166a5fa8f286a5e0d1355143354ee48126b116c6d390b9a7f8b1dc` and
`68413154d31f18af0e418ba8849bfb5fdb5a85a09a8832f1bb7a38c6e627a565`.

## Focal-corrected deterministic BIM quantile audit

The focal correction changes the DA3/BIM ratio population, so the deterministic
baseline was re-audited rather than assuming that q45 remained appropriate.
The scan used focal-corrected DA3, hit-only BIM, official-all-valid GT, and all
91 fixed quantiles from q05 to q95 at step 0.01. GT is used only to select/score
the fixed development-set quantile; runtime scale estimation still reads only
DA3 and BIM.

Two different development views expose a distribution shift:

- train pixel-micro scale-only selects q52 (`0.129114`), while q45 is
  `0.132284`; train room-macro selects q51, and q51 wins 24/30 leave-one-room-out
  folds;
- freezing q52 to test gives `0.125125`, worse than q45 `0.114221` and the
  existing robust log-cap scale `0.110329`. Thus q52 is rejected as room-domain
  overfitting;
- validation pixel-micro scale-only selects q45 (`0.105683`). Validation
  room-macro scale-only selects q52, so the preferred q depends on aggregation;
  the registered headline aggregation remains pixel-micro.

Because the complete BIM-direct also contains a consistency gate and Gaussian
local propagation, the same 91-point validation scan was repeated after
recomputing those operations for every candidate. Both pixel-micro and
room-macro complete BIM-direct select **q45** (`0.104263`). Therefore q45 is not
merely inherited history under the registered validation objective; it is the
measured optimum for the full deterministic chain on this split.

| Split / deterministic output | AbsRel | MAE (m) | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Validation q45 scale-only | 0.105683 | 0.299724 | 0.777043 | 0.876910 |
| Validation q45 full BIM-direct | **0.104263** | **0.282716** | **0.763848** | **0.877395** |
| Validation robust scale-only | 0.119341 | 0.333050 | 0.847420 | 0.849301 |
| Validation robust full BIM-direct | 0.118152 | 0.316482 | 0.837841 | 0.849215 |
| Test q45 scale-only | 0.114221 | 0.219490 | 0.540775 | 0.881606 |
| Test q45 full BIM-direct | 0.114367 | 0.217653 | 0.540258 | 0.881781 |
| Test robust scale-only | **0.110329** | 0.218909 | 0.556052 | 0.879256 |
| Test robust full BIM-direct | 0.110724 | 0.217921 | 0.557609 | 0.879482 |

The local correction lowers validation AbsRel by 1.34% for q45 and 1.00% for
the robust scale, but changes test AbsRel by -0.13%/-0.36% (negative means a
regression). Its MAE can improve while AbsRel remains neutral, so Gaussian
propagation is not a robust accuracy source. More importantly, corrected raw
DA3 test AbsRel is `0.085453`, better than every deterministic BIM variant in
this table. The focal correction therefore reverses the earlier narrative:
the current non-learning BIM chain is a comparator/diagnostic, not an
enhancement over standalone DA3 on held-out rooms.

Artifacts:

- `results/stanford_area1/fixed_scale_quantile_metric_da3_full_depth_train/selection.json`
- `results/stanford_area1/fixed_scale_quantile_metric_da3_full_depth_test/summary.json`
- `results/stanford_area1/fixed_bim_direct_quantile_metric_da3_full_depth_val/selection.json`

The first two implement a train-only selection/frozen-test audit. The last is
the validation-only exhaustive full-chain scan and must not be described as a
blind-test selection. The shared script SHA256 is
`a9dc44f680e1f57b6e3ce52ae467cf9de232c11854ad8dd935fd25d6b254cb38`.

## Three-round scale-conditioned attention

The static head predicts spatial attention once and then performs analytic
Huber updates with fixed learned weights. The new candidate starts from raw
focal-corrected DA3, `z^(0)=log(s)=0`, and unrolls three updates:

\[
e_i^{(t)}=\log(D_{BIM,i}/D_{DA3,i})-z^{(t)},\qquad
a_i^{(t)}=f_\theta(F_i,e_i^{(t)},|e_i^{(t)}|,\rho(e_i^{(t)})).
\]

The same 1x1 reliability MLP is reused in every round and for every head. Each
round forms a weighted Huber center and applies a bounded damped log-scale
update. Only the damping coefficients are round-specific; they converged from
`0.5/0.5/0.5` to `0.5152/0.5133/0.5117`. Frames without BIM support return
`s=1`. The final scale anchors the unchanged `r_low+r_detail` refiner. The
three outputs receive normalized train-only oracle-scale supervision weighted
`0.25/0.50/1.00`; no GT enters validation or inference.

Training keeps the static comparator's room split, focal-corrected frozen DA3
inputs/features, official-all-valid GT, batch 8, seed 42, and scratch `3 scale
/ 9 refiner / 3 joint` schedule. It has 4,270,990 task parameters; recurrence
adds only 3,236. Peak allocated memory was 12.96 GiB. Validation selected
zero-based epoch 11, before joint tuning, solely by final AbsRel.

| Split / stage | Raw DA3 | Round 1 | Round 2 | Round 3 | + low | + detail |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.090705 | 0.076070 | 0.070925 | 0.069853 | 0.064756 | **0.064213** |
| Test | 0.085453 | 0.073603 | 0.069722 | 0.068885 | 0.063515 | **0.063049** |

Relative to static attention, round-3 scale improves validation/test AbsRel by
7.25%/12.21%; final improves by 0.36%/4.68%. A 10,000-resample paired room
bootstrap for new-minus-static scale gives 95% CIs `[-0.01406,-0.00013]` and
`[-0.01371,-0.00528]`. Final-output CIs are `[-0.00491,0.00318]` and
`[-0.00914,-0.00035]`: scale gains are consistent on both splits, final
validation is inconclusive, and final test improvement is consistent.

The gain is not uniform. Test BIM-foreground-conflict AbsRel improves from
`0.12390` to `0.11449`, while furniture worsens from `0.07265` to `0.07609`
and BIM-consistent pixels from `0.02709` to `0.02770`. Validation furniture and
non-structural pixels also regress. The recurrence is a stronger global scale
estimator and aggregate checkpoint, but foreground residual refinement remains
open. Area_1 test was already revealed, so new blind-area confirmation is still
required for a paper claim.

Reproduction:

```bash
.venv/bin/python scripts/model/train.py \
  --config configs/stanford_area1_iterative_scale_3round_full_depth_metric_da3.yaml \
  --device cuda

.venv/bin/python scripts/model/evaluate_stanford_area1.py \
  --config configs/stanford_area1_iterative_scale_3round_full_depth_metric_da3.yaml \
  --checkpoint outputs/stanford_area1_iterative_scale_3round_full_depth_metric_da3/accepted.pt \
  --split val --depth-support all-valid \
  --output results/stanford_area1/iterative_scale_3round_full_depth_metric_da3_val \
  --device cuda --batch-size 8 --inference-seed 42 \
  --bootstrap-repetitions 10000 --bootstrap-seed 42 \
  --allow-unverified-robust-comparator
```

Repeat the evaluator with `--split test` only after freezing the validation
checkpoint. Config/checkpoint/history/run-state SHA256 values are
`520b867193f4c351c4c99e0ed36a165695a37cf1b35afcd296aabe6cf3f26d59`,
`74f2797dc42a4e7e8359440ea9d305a073e7ec0d2fe0850fb1ab79877bb7ae6d`,
`7139c8b2ec333fa6d9d3b94cad2d73670216958a78979485b687c9dc8a91c959`,
and `d00d3d535f50eedf437724983bc1113d4973b0cd01c5eaa690ba4fa079036409`.
Validation/test summary SHA256 values are
`916a333e1d769b109ce0ea7ddda7cd4ed17277ffc37f9ff620759fe91abbe37e`
and `a9e67c2c43d34211282fbe56b595c8fc273c8c9a24ebc6bc684a4771341efb29`.
