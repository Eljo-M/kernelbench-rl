"""Token-level clipped group-relative objective, with an independently testable NumPy form."""

from __future__ import annotations

import math

import numpy as np


def grpo_loss_numpy(new_logps, old_logps, reference_logps, advantages, mask, *, clip=0.2, beta=0.04):
    """Average token losses per completion, then average completions (GRPO reduction).

    Inputs have shape [group, tokens], except advantages [group]. KL uses the
    nonnegative sampled estimator exp(ref-new) - (ref-new) - 1. Padding is masked.
    This function is a numerical reference, not a policy training implementation.
    """
    new, old, ref = [np.asarray(x, dtype=np.float64) for x in (new_logps, old_logps, reference_logps)]
    weights = np.asarray(mask, dtype=np.float64)
    advantage = np.asarray(advantages, dtype=np.float64)
    if new.ndim != 2 or new.shape != old.shape or new.shape != ref.shape or new.shape != weights.shape:
        raise ValueError("Log probabilities and mask must have matching [group, tokens] shapes")
    if advantage.shape != (new.shape[0],):
        raise ValueError("One advantage is required per completion")
    if new.shape[0] == 0 or not np.all((weights == 0) | (weights == 1)) or np.any(weights.sum(axis=1) == 0):
        raise ValueError("Each completion must contain at least one unmasked token")
    if not all(np.all(np.isfinite(x)) for x in (new, old, ref, advantage)):
        raise ValueError("Objective inputs must be finite")
    if not 0 < clip < 1 or beta < 0 or not math.isfinite(beta):
        raise ValueError("clip must be between zero and one; beta must be finite and nonnegative")
    ratio = np.exp(new - old)
    surrogate = np.minimum(ratio * advantage[:, None], np.clip(ratio, 1 - clip, 1 + clip) * advantage[:, None])
    delta = ref - new
    kl = np.expm1(delta) - delta
    losses = (-surrogate + beta * kl) * weights
    result = float(np.mean(losses.sum(axis=1) / weights.sum(axis=1)))
    if not math.isfinite(result):
        raise ValueError("Nonfinite objective; reduce the update size")
    return result


def grpo_loss_torch(new_logps, old_logps, reference_logps, advantages, mask, *, clip=0.2, beta=0.04):
    """Differentiable counterpart of grpo_loss_numpy; torch is an optional dependency."""
    import torch

    if new_logps.ndim != 2 or any(x.shape != new_logps.shape for x in (old_logps, reference_logps, mask)):
        raise ValueError("Log probabilities and mask must have matching [group, tokens] shapes")
    if advantages.shape != (new_logps.shape[0],) or new_logps.shape[0] == 0:
        raise ValueError("One advantage is required per completion")
    if not 0 < clip < 1 or beta < 0 or not math.isfinite(beta):
        raise ValueError("Invalid clipping or KL coefficient")
    if not bool(torch.all((mask == 0) | (mask == 1))) or bool(torch.any(mask.sum(dim=1) == 0)):
        raise ValueError("Invalid completion mask")
    if not all(bool(torch.all(torch.isfinite(x))) for x in (new_logps, old_logps, reference_logps, advantages)):
        raise ValueError("Objective inputs must be finite")
    ratio = torch.exp(new_logps - old_logps.detach())
    advantage = advantages.detach().unsqueeze(1)
    surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - clip, 1 + clip) * advantage)
    delta = reference_logps.detach() - new_logps
    kl = torch.expm1(delta) - delta
    loss = (((-surrogate + beta * kl) * mask).sum(dim=1) / mask.sum(dim=1)).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite objective; reduce the update size")
    return loss
