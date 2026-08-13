import pytest
import torch
from torch.utils.data import Dataset

from bim_priorda3.engine import build_loader


class _RegionDataset(Dataset):
    def __init__(self) -> None:
        self.records = [
            {"region": "A"},
            {"region": "A"},
            {"region": "B"},
            {"region": "B"},
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> int:
        return index


class _UnequalRegionDataset(Dataset):
    def __init__(self) -> None:
        self.records = [
            {"region": "A"},
            {"region": "A"},
            {"region": "A"},
            {"region": "A"},
            {"region": "B"},
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> int:
        return index


def _sample_order(generator: torch.Generator) -> list[int]:
    loader = build_loader(
        _RegionDataset(),
        batch_size=2,
        num_workers=0,
        shuffle=True,
        region_balanced=True,
        samples_per_epoch=8,
        generator=generator,
        persistent_workers=False,
    )
    return [int(index) for batch in loader for index in batch]


def test_loader_generator_is_independent_and_repeatable() -> None:
    first_generator = torch.Generator().manual_seed(42)
    first = _sample_order(first_generator)
    torch.rand(100)
    second_generator = torch.Generator().manual_seed(42)
    second = _sample_order(second_generator)
    assert first == second
    assert len(first) == 8


def test_loader_supports_sqrt_room_balance_without_oversampling_to_equal_rooms() -> None:
    loader = build_loader(
        _UnequalRegionDataset(),
        batch_size=1,
        num_workers=0,
        shuffle=True,
        region_balanced=True,
        region_balance_exponent=0.5,
        generator=torch.Generator().manual_seed(7),
        persistent_workers=False,
    )
    weights = loader.sampler.weights.tolist()
    assert weights[:4] == [0.5] * 4
    assert weights[4] == 1.0


@pytest.mark.parametrize("samples", [0, -1, True, 1.5])
def test_loader_rejects_invalid_samples_per_epoch(samples: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_loader(
            _RegionDataset(),
            batch_size=2,
            num_workers=0,
            shuffle=True,
            samples_per_epoch=samples,
        )


@pytest.mark.parametrize("exponent", [-0.1, 1.1])
def test_loader_rejects_invalid_region_balance_exponent(exponent: float) -> None:
    with pytest.raises(ValueError, match="region_balance_exponent"):
        build_loader(
            _RegionDataset(),
            batch_size=2,
            num_workers=0,
            shuffle=True,
            region_balanced=True,
            region_balance_exponent=exponent,
        )
