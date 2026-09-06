# PriorBIMDA two-stage metric-depth pipeline

This variant keeps DA3 and DAv2 in metric-depth space while borrowing
PriorDA's separate coarse/fine optimization structure.

## Stage 1: global scale

Stage 1 uses the existing fixed-attention, three-update pseudo-Huber estimator.
It is trained for exactly three epochs and Stage 2 accepts only its
validation-selected `best.pt`. During Stage-2 training the complete Stage-1
module is frozen and forced to evaluation mode, including token dropout.

For every frame it exports:

- global scale `s` and `D_s = s D_DA3`;
- the learned fixed attention map `A`;
- `W_eff`, formed with the final per-head pseudo-Huber centers as
  `sum_h pi_h A_h / sqrt(1 + ((r-c_h)/delta)^2)`.

`A` and `W_eff` are exposed at both native scale-head token resolution and
full image resolution.

## Stage 2: conditioned DAv2 Metric Indoor

RGB uses the untouched pretrained DAv2 RGB patch embedding. A separate
three-channel convolution has exactly the same kernel and stride, is strictly
zero-initialized, and is added to RGB patch tokens before the class token,
positional encoding, and DINOv2 transformer.

The current generalization-oriented condition is:

```text
m, r = per-frame min(D_BIM), max(D_BIM) - min(D_BIM)
C1 = clip((D_s - m) / r, 0, 1)
C2 = W_eff / per-frame max(W_eff)
C3 = 0.5 * (clip(log(D_BIM) - log(D_s), -1.5, 1.5) / 1.5 + 1)
D_final = m + r * clip(D_DAv2 / 20 m, 0, 1)
```

The metric minimum/range comes from valid BIM pixels and is reused unchanged
for output de-normalization. Frames receiving full BIM dropout fall back to
the frozen Stage-1 depth minimum/range. All three condition channels are in
`[0,1]`; BIM-invalid pixels set `C2` and `C3` to zero while `C1` remains dense.
The completed fixed-train-statistics experiment remains reproducible with
`configs/stanford_area1_priorbimda_stage2_dav2_metric.yaml`.

Stage-2 training restores the F36 anchor's BIM augmentation configuration: a
four-pixel shift with probability `0.20`, 12% square dropout with probability
`0.15`, full dropout with probability `0.03`, and log-depth noise (`std=0.02`)
with probability `0.20`. The anchor's three-pixel edge-dilation setting is also
preserved, although it is inert for both three-channel models because neither
selected training batch consumes the separate BIM-edge tensor.

The refinement objective follows the repository's PriorDA fine-stage balance:
a dominant robust native-domain loss, plus lower-weight metric log-depth and
log-gradient terms. The native domain here is metric log depth rather than
support-relative disparity, so absolute scale is retained.

## Run

From the repository root, with the `priorbimda` environment active:

```bash
scripts/model/run_priorbimda_two_stage.sh
```

Or run Stage 2 after an existing Stage-1 `best.pt`:

```bash
python scripts/model/train_priorbimda_two_stage.py \
  --config configs/stanford_area1_priorbimda_stage2_dav2_metric_minmax_augmented.yaml \
  --device cuda
```

For an immutable run, copy the Stage-1 checkpoint SHA256 into
`model.frozen_scale.checkpoint_sha256` before Stage 2. Leaving it null is
supported for the first chained run; the observed digest is still recorded.
