"""PriorDA v1.1 fine network with a condition-only BIM domain adapter.

No completion, alignment, teacher, or metric residual is used here.
"""
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .priorda_v11_bim_adapter import _load_official_priorda_v11


def positive_reciprocal(value):
    # Exactly the official depth2disparity rule, including nonpositive inputs.
    result = torch.zeros_like(value)
    valid = value > 0
    result[valid] = 1.0 / value[valid]
    return result


def fine_condition(pred, bim, mask):
    if pred.shape != bim.shape or bim.shape != mask.shape:
        raise ValueError("All depth/mask shapes must agree")
    valid = mask.bool()
    if not bool(valid.flatten(1).any(1).all()):
        raise ValueError("Official prior normalization requires nonempty BIM support")
    if not bool((torch.isfinite(bim[valid]) & (bim[valid] > 0)).all()):
        raise ValueError("Invalid BIM depth on valid support")
    if not bool((torch.isfinite(pred) & (pred > 0)).all()):
        raise ValueError("D_pred must be finite positive dense metric depth")
    bim, pred = bim.float(), pred.float()
    minimum = bim.masked_fill(~valid, torch.inf).amin((-2, -1), keepdim=True)
    maximum = bim.masked_fill(~valid, -torch.inf).amax((-2, -1), keepdim=True)
    span = maximum - minimum
    span = torch.where(span == 0, torch.ones_like(span), span)
    q_pred = positive_reciprocal((pred - minimum) / span)
    q_bim = positive_reciprocal((bim.masked_fill(~valid, 0) - minimum) / span)
    return torch.cat((valid.float(), q_pred, q_bim), 1), minimum, span


class BottleneckResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(768, 192, 1), nn.GELU(),
            nn.Conv2d(192, 192, 3, padding=1), nn.GELU(),
            nn.Conv2d(192, 768, 1),
        )
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, feature):
        return feature + self.branch(feature)


class AdaptedAlphaProjection(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.pretrained = pretrained
        # Three serial residual blocks: F' = F + A(F), A = stack(F) - F.
        # Do NOT add F a second time outside this already-residual stack.
        self.adapter = nn.Sequential(*(BottleneckResBlock() for _ in range(3)))

    def forward(self, condition):
        return self.adapter(self.pretrained(condition))


class PriorDABIMDomain(nn.Module):
    def __init__(self, fine):
        super().__init__()
        self.fine = fine
        patch = fine.pretrained.patch_embed
        patch.alpha_proj = AdaptedAlphaProjection(patch.alpha_proj)
        self.stage = 1

    @classmethod
    def from_checkpoint(cls, repository: Path, path: Path):
        # STRICT load happens before adapter construction; no reinitialization
        # of RGB, alpha projection, encoder, decoder or output layers.
        return cls(_load_official_priorda_v11(repository=repository, checkpoint_path=path))

    def enable_gradient_checkpointing(self):
        # Preserve transformer structure and state keys. Frozen blocks still
        # require autograd w.r.t. the trainable condition entering the encoder.
        for block in self.fine.pretrained.blocks:
            original = block.forward
            def wrapped(*args, _original=original, **kwargs):
                if self.training and torch.is_grad_enabled():
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)
            block.forward = wrapped

    def configure_stage(self, stage, *, adapter_lr=1e-4, decoder_lr=1e-5,
                        alpha_lr=5e-6, vit_lr=1e-6):
        if stage not in (1, 2):
            raise ValueError("Only stages 1 and 2 (last two blocks) are supported")
        self.stage = stage
        self.requires_grad_(False)
        alpha = self.fine.pretrained.patch_embed.alpha_proj
        components = [("adapter", alpha.adapter, adapter_lr),
                      ("decoder", self.fine.depth_head, decoder_lr),
                      ("alpha_proj", alpha.pretrained, alpha_lr)]
        if stage == 2:
            components += [("vit_last2", self.fine.pretrained.blocks[-2:], vit_lr)]
        groups = []
        for name, module, lr in components:
            module.requires_grad_(True)
            groups.append(dict(name=name, params=list(module.parameters()), lr=lr))
        return groups

    def forward(self, batch):
        condition, minimum, span = fine_condition(
            batch["base_depth"], batch["bim_depth"], batch["bim_valid"])
        # Official raw2input expects BGR uint8 and performs RGB/ImageNet norm.
        image = (batch["rgb"][:, [2, 1, 0]] * 255).round().to(torch.uint8)
        q = self.fine(image, input_size=image.shape[-2], condition=condition,
                      device=image.device).float()
        depth = minimum + span * positive_reciprocal(q)
        return {"depth": depth, "normalized_disparity": q}
