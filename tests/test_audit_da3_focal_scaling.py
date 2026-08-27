from __future__ import annotations

import numpy as np

from scripts.analysis import audit_da3_focal_scaling as audit


def test_prepared_focal_factor_converts_canonical_depth_without_gt_fitting(tmp_path) -> None:
    sample = tmp_path / "sample.npz"
    canonical = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = canonical * 2.0
    np.savez_compressed(
        sample,
        base_depth=canonical,
        gt_depth=target,
        gt_valid=np.ones_like(target),
        intrinsic=np.asarray(
            [
                [600.0, 0.0, 1.0],
                [0.0, 600.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    old, corrected, factor, old_frame, corrected_frame = audit._load_record(
        {"id": "synthetic/frame_0", "sample": str(sample)},
        target_shape=(2, 2),
        all_valid_stanford=False,
    )

    assert factor == 2.0
    assert old[1] / old[0] == 0.5
    assert corrected[1] == 0.0
    assert old_frame == 0.5
    assert corrected_frame == 0.0
