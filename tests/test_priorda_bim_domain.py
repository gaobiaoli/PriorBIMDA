import torch
from torch import nn

from bim_priorda3.models.priorda_bim_domain import (
    AdaptedAlphaProjection, fine_condition, positive_reciprocal, PriorDABIMDomain,
)
from bim_priorda3.data.bim_domain import corrupt_bim
from bim_priorda3.models.dav2_dense4 import dense_silog_loss


def test_official_shared_normalization_and_order():
    bim = torch.tensor([[[[2., 4.], [6., 0.]]]])
    mask = (bim > 0).float()
    pred = torch.tensor([[[[1., 3.], [10., 2.]]]])
    condition, minimum, span = fine_condition(pred, bim, mask)
    assert minimum.item() == 2 and span.item() == 4
    assert torch.equal(condition[:, :1], mask)
    assert torch.equal(condition[:, 1:2], torch.tensor([[[[0., 4.], [.5, 0.]]]]))
    assert torch.equal(condition[:, 2:], torch.tensor([[[[0., 2.], [1., 0.]]]]))
    assert positive_reciprocal(torch.tensor([1e-9])).item() > 1e8  # no hidden cap


def test_adapter_exact_identity_and_live_gradients():
    torch.manual_seed(0)
    alpha = nn.Conv2d(3, 768, 14, 14)
    model = AdaptedAlphaProjection(alpha)
    condition = torch.randn(1, 3, 28, 28)
    assert torch.equal(model(condition), alpha(condition))
    model(condition).square().mean().backward()
    assert alpha.weight.grad.abs().sum() > 0
    for block in model.adapter:
        assert block.branch[-1].weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()
    model(condition).square().mean().backward()
    for block in model.adapter:
        assert block.branch[0].weight.grad.abs().sum() > 0


def test_corruption_reproducibility_distribution_and_alignment():
    depth = torch.arange(1, 1025).reshape(1, 32, 32).float()
    mask = torch.ones_like(depth)
    counts = [0] * 4
    for seed in range(1000):
        d, m, meta = corrupt_bim(depth, mask, seed=seed)
        other = corrupt_bim(depth, mask, seed=seed)
        assert torch.equal(d, other[0]) and torch.equal(m, other[1]) and meta == other[2]
        counts[meta["kind"]] += 1
        assert torch.equal(d > 0, m > 0)
        if meta["kind"] == 0:
            assert torch.equal(d, depth)
        if meta["kind"] == 1:
            assert .099 <= float((m == 0).float().mean()) <= .301
        if meta["kind"] == 2:
            expected = torch.roll(depth, (meta["dy"], meta["dx"]), (-2, -1))
            assert torch.equal(d[m > 0], expected[m > 0])
            assert meta["dx"] in (-4, -2, 2, 4) and meta["dy"] in (-4, -2, 2, 4)
    assert all(abs(count / 1000 - prob) < .04 for count, prob in zip(counts, (.55, .2, .15, .1)))
    assert torch.equal(mask, torch.ones_like(mask))  # no mutation


def test_stage_freezing_and_optimizer_partition():
    fine = nn.Module()
    fine.pretrained = nn.Module()
    fine.pretrained.patch_embed = nn.Module()
    fine.pretrained.patch_embed.proj = nn.Conv2d(3, 768, 1)
    fine.pretrained.patch_embed.alpha_proj = nn.Conv2d(3, 768, 1)
    fine.pretrained.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(12))
    fine.depth_head = nn.Linear(4, 1)
    model = PriorDABIMDomain(fine)
    for stage in (1, 2):
        groups = model.configure_stage(stage)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids))
        assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
        assert not fine.pretrained.patch_embed.proj.weight.requires_grad
        for i, block in enumerate(fine.pretrained.blocks):
            assert block.weight.requires_grad == (stage == 2 and i >= 10)
        assert [g["lr"] for g in groups] == ([1e-4, 1e-5, 5e-6] + ([1e-6] if stage == 2 else []))


def test_silog_metric_supervision_only_valid_gt():
    pred = torch.tensor([[[[2., 4., 100.]]]], requires_grad=True)
    gt = torch.tensor([[[[1., 3., 0.]]]])
    mask = gt > 0
    loss = dense_silog_loss(pred, gt, mask)["total"]
    g = (pred[mask] + 1e-7).log() - (gt[mask] + 1e-7).log()
    assert torch.allclose(loss, 10 * (g.var() + .15 * g.mean().square()).sqrt())
    loss.backward()
    assert pred.grad[0, 0, 0, 2] == 0
