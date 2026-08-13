import numpy as np
import pytest

from scripts.analysis.paired_bootstrap import (
    aggregate_metric,
    bootstrap_mean_differences,
    bootstrap_paired_differences,
)


def test_moving_block_bootstrap_is_reproducible_and_bounded() -> None:
    difference = np.asarray([-0.2, -0.1, 0.1, 0.2])
    first = bootstrap_mean_differences(difference, 100, block_size=2, seed=7)
    second = bootstrap_mean_differences(difference, 100, block_size=2, seed=7)
    assert np.array_equal(first, second)
    assert np.all(first >= difference.min())
    assert np.all(first <= difference.max())


def test_moving_block_bootstrap_rejects_invalid_block_size() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_differences(
            np.asarray([0.0, 1.0]),
            10,
            block_size=3,
            seed=1,
        )


def test_pixel_pooled_bootstrap_uses_valid_pixel_weights() -> None:
    candidate = np.asarray([0.1, 0.9])
    baseline = np.asarray([0.2, 0.8])
    weights = np.asarray([9.0, 1.0])
    assert aggregate_metric(candidate, weights, root_mean_square=False) == pytest.approx(0.18)
    differences = bootstrap_paired_differences(
        candidate,
        baseline,
        weights,
        samples=20,
        block_size=2,
        seed=3,
    )
    assert np.allclose(differences, -0.08)
