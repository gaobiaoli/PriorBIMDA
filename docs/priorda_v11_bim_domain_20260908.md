# PriorDA v1.1 BIM fine-stage domain adaptation

Entry point: `scripts/model/train_priorda_bim_domain.py`.
Config: `configs/stanford_area1_priorda_v11_bim_domain_6plus12epoch_20260908.yaml`.

This is a separate experiment path; historical models/results are retained,
but no completion, KNN, scale-shift fit, coarse prediction or trust target is
computed by this dataset/model/trainer. Inputs are RGB, focal-corrected cached
DA3 metric depth, and dense BIM depth/mask. No GT-based BIM filtering is used.

The full official v1.1 ViT-B fine-network checkpoint is SHA256 verified and
strictly loaded before attaching three serial residual bottleneck blocks to
alpha_proj. The pretrained alpha projection is not reset. The adapter maps
768 -> 192 -> 192 -> 768 with GELUs and zero-initialized final projections.
The stack already includes residual additions; it is not added to F_c twice.

Condition order is [BIM mask, normalized prediction disparity, normalized BIM
disparity]. Shared min/range comes from valid BIM pixels, exactly as the
official metric-prior normalization. Zero range becomes one. Empty BIM support
fails explicitly; there is no invented fallback normalization. Nonpositive
normalized depth/disparity maps to zero, as in the original implementation;
there is no epsilon clamp or disparity cap. The original output ReLU is kept.
RGB follows the official BGR-uint8 to RGB/ImageNet preprocessing at the existing
504-pixel aligned dataset resolution. Final metric depth is min + range *
disparity2depth(network_output). Initialization preserves the official fine
network on these inputs, not exact equality to DA3 or the removed coarse pipeline.

Only ZoeDepth SILog is supervised, on all finite positive official GT pixels.
There are no anchor, residual, consistency, teacher or preservation losses.
SILog is computed in FP32 across each physical microbatch (2 frames), with
8-way gradient accumulation (effective batch 16). Partial accumulation windows
use their actual length. The loss uses correction=1 variance and the existing
ZoeDepth log epsilon 1e-7; a 1e-12 square-root floor guards exact perfect fits.

Schedule assumption (epochs were not specified in the new request): 6 epochs
Stage 1, then 12 epochs Stage 2. Constant AdamW group LRs are adapter 1e-4,
decoder 1e-5, pretrained alpha 5e-6, and last two ViT blocks 1e-6 in Stage 2.
Stage 2 continues the final Stage 1 weights and existing optimizer moments;
only the new ViT group is added. All other encoder parameters remain frozen.
Gradient checkpointing does not detach frozen blocks from condition gradients.
There is no automatic last-four/full-encoder experiment.

BIM-only corruption uses a per-sample CPU torch.Generator seeded with
42 + zero_based_epoch * dataset_size + sample_index. Workers are recreated
each epoch so epoch values propagate correctly. No random/NumPy global RNG is
used for augmentation. Categories are 55% clean, 20% local dropout, 15% shift,
10% both. Dropout deletes 10-30% of VALID pixels nearest a randomly selected
valid center; depth and mask shift together with zero padding, not wraparound.
RGB, DA3 and GT are never shifted or jittered. Sampling is a reproducible
uniform shuffle without replacement. Full BIM drop and DA3 perturbation are off.

Validation is performed after fine-tuning epochs; no zero-shot benchmark is
scheduled. Each stage's best checkpoint (validation frame-macro AbsRel) is
tested on Stanford only after both stages finish. Overall best/latest and
Stage 1 latest are also saved. The run directory holds status.json, receipt.json,
history.json, training_history.csv, epoch validations, and source/config snapshots.

Validation before launch: five unit tests passed; real-image FP32 outputs were
bitwise identical with/without the newly initialized adapter (max difference 0).
Adapter, alpha projection and decoder had finite nonzero gradients; frozen
parameters had no gradients. A complete two-stage, four-sample smoke test passed,
including nonzero gradients in the last two ViT blocks. Smoke checkpoints were
moved from the full system disk into the data disk without deleting them.
