from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "model"
    / "train_frozen_huber_dav2_low_refiner.py"
)
SPEC = importlib.util.spec_from_file_location("train_frozen_huber_dav2_low_refiner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_masked_area_downsample_ignores_invalid_values() -> None:
    value = torch.tensor([[[[1.0, 999.0], [3.0, 5.0]]]])
    valid = torch.tensor([[[[True, False], [True, True]]]])

    target, support = MODULE.masked_area_downsample(value, valid, (1, 1))

    torch.testing.assert_close(target, torch.tensor([[[[3.0]]]]))
    assert bool(support.item())


def test_masked_area_downsample_marks_empty_cells_invalid() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    valid = torch.zeros_like(value, dtype=torch.bool)
    valid[..., :2, :2] = True

    target, support = MODULE.masked_area_downsample(value, valid, (2, 2))

    torch.testing.assert_close(target[..., 0, 0], torch.tensor([[2.5]]))
    assert support.tolist() == [[[[True, False], [False, False]]]]
    assert torch.count_nonzero(target[..., 1, 1]) == 0
