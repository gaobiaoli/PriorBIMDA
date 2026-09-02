"""Training-only augmentations for cached depth foundation predictions."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def apply_bim_condition_dropout(
    condition: torch.Tensor,
    *,
    probability: float | None = None,
    applied: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the complete BIM condition for selected batch samples.

    Pass ``probability`` to sample a new per-sample mask, or pass ``applied``
    to reuse a previously sampled mask (for example in an equivariance second
    forward). Exactly one of the two arguments must be provided. The input
    condition is never modified in place.
    """

    if condition.ndim != 4:
        raise ValueError("BIM condition must have shape [B,C,H,W]")
    if (probability is None) == (applied is None):
        raise ValueError("Provide exactly one of probability or applied")
    batch_size = condition.shape[0]
    if applied is None:
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("BIM condition dropout probability must be in [0,1]")
        if probability == 0.0:
            return condition, torch.zeros(
                batch_size, dtype=torch.bool, device=condition.device
            )
        applied = torch.rand(batch_size, device=condition.device) < probability
    else:
        if applied.shape != (batch_size,) or applied.dtype != torch.bool:
            raise ValueError("BIM condition dropout mask must be bool with shape [B]")
        if applied.device != condition.device:
            raise ValueError("BIM condition and dropout mask must be on the same device")
    dropped = condition.masked_fill(applied[:, None, None, None], 0)
    return dropped, applied


def apply_da3_global_scale_perturbation(
    batch: Mapping[str, torch.Tensor],
    *,
    probability: float,
    log_range: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Multiply each selected DA3 depth map by one sampled global scale.

    For sample ``i``, ``log_q[i] ~ Uniform(-log_range, log_range)`` and
    ``base_depth[i]`` becomes ``exp(log_q[i]) * base_depth[i]``.  All other
    batch tensors are shared unchanged.  Callers must construct BIM/DA3 ratio
    features and scale-supervision targets from the returned batch so that
    ``z' = log(D_BIM / D_DA3')`` and ``c*' = c* - log_q`` hold automatically.

    Returns ``(augmented_batch, log_q, applied)`` with ``log_q`` shaped
    ``[B,1,1,1]`` and ``applied`` shaped ``[B]``.
    """

    probability = float(probability)
    log_range = float(log_range)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("DA3 global-scale perturbation probability must be in [0,1]")
    if not torch.isfinite(torch.tensor(log_range)) or log_range < 0.0:
        raise ValueError("DA3 global-scale perturbation log_range must be finite and non-negative")
    if "base_depth" not in batch:
        raise KeyError("DA3 global-scale perturbation requires batch['base_depth']")
    base_depth = batch["base_depth"]
    if base_depth.ndim != 4 or base_depth.shape[1] != 1:
        raise ValueError("DA3 base_depth must have shape [B,1,H,W]")

    batch_size = base_depth.shape[0]
    if probability == 0.0 or log_range == 0.0:
        # Preserve the exact RNG stream and tensor values for legacy configs in
        # which this new augmentation is absent or disabled.
        return (
            dict(batch),
            torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32, device=base_depth.device),
            torch.zeros(batch_size, dtype=torch.bool, device=base_depth.device),
        )
    applied = torch.rand(batch_size, device=base_depth.device) < probability
    log_q = torch.empty(
        (batch_size, 1, 1, 1), dtype=torch.float32, device=base_depth.device
    ).uniform_(-log_range, log_range)
    log_q = torch.where(applied[:, None, None, None], log_q, torch.zeros_like(log_q))
    augmented = dict(batch)
    augmented["base_depth"] = base_depth * log_q.exp().to(dtype=base_depth.dtype)
    return augmented, log_q, applied


__all__ = ["apply_bim_condition_dropout", "apply_da3_global_scale_perturbation"]
