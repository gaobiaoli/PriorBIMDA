# Reproduction workspace consolidation

Consolidated on 2026-09-05 from three temporary anchor workspaces into the
main `/home/bgao491/PriorBIMDA` checkout.

## Anchors

- `anchor_1c07d65`: based on commit `1c07d65` (`903-5`). Its F36 reproduction,
  deterministic epoch-1 comparisons, seed-42/batch-16 probe and retrain,
  Matterport zero-shot evaluations, configs, pipeline script, and checkpoints
  were moved into the normal `results/`, `configs/`, `scripts/`, and `outputs/`
  trees.
- `anchor_cbd3cef`: based on commit `cbd3cef` (`903-4`). Its deterministic
  epoch-1 comparisons, Matterport zero-shot evaluations, and checkpoints were
  moved into the normal result/output trees. The reproduction that originally
  used the generic calibrated-disagreement directory was renamed to
  `f36_anchor_cbd3cef_reproduction` in `results/stanford_area1` and
  `stanford_area1_f36_anchor_cbd3cef_reproduction` in `outputs` so that the
  main checkout's existing run was not overwritten.
- `anchor_latest_b578617_deterministic`: based on commit `b578617` (`904-2`).
  Its bicubic trace, strict deterministic smoke run, epoch-1 comparisons, and
  checkpoints were moved into the normal result/output trees.

## Working-tree source snapshots

Each anchor's modified tracked source files are retained below its
`working_tree/` directory. They are snapshots, not active imports. This keeps
the exact code used by the reproduction runs without replacing the newer,
independently modified source files in the main checkout. Use the anchor commit
listed above as the baseline when reconstructing a diff.

Materialized run configs and logs inside migrated artifacts intentionally keep
their original absolute workspace paths as provenance. The reusable pipeline
script in `scripts/model` was updated to point at the retained main checkout.
