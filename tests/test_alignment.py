import torch

from bim_priorda3.models.alignment import RobustLocalAffineAlignment


def test_alignment_recovers_affine_depth() -> None:
    generator = torch.Generator().manual_seed(7)
    base = torch.rand((1, 1, 64, 64), generator=generator) * 2.5 + 0.5
    bim = 1.2 * base + 0.15
    valid = torch.ones_like(base)
    trust = torch.ones_like(base)
    edge = torch.zeros_like(base)
    module = RobustLocalAffineAlignment(
        kernel_size=9,
        downsample=1,
        min_support=0.01,
        scale_range=(0.5, 2.0),
    )
    coarse, support, scale, shift = module(base, bim, valid, trust, edge)
    center = (slice(None), slice(None), slice(10, -10), slice(10, -10))
    assert torch.mean(torch.abs(coarse[center] - bim[center])) < 1e-3
    assert torch.mean(torch.abs(scale[center] - 1.2)) < 1e-3
    assert torch.mean(torch.abs(shift[center] - 0.15)) < 1e-3
    assert torch.all(support[center] > 0)


def test_alignment_falls_back_without_trusted_bim() -> None:
    base = torch.rand(1, 1, 32, 32) + 0.5
    module = RobustLocalAffineAlignment(kernel_size=5, downsample=1)
    coarse, *_ = module(
        base,
        torch.ones_like(base),
        torch.zeros_like(base),
        torch.zeros_like(base),
        torch.zeros_like(base),
    )
    assert torch.allclose(coarse, base)
