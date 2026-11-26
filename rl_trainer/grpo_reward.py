from __future__ import annotations

import math
import numpy as np


def compute_kernel_reward(
    compiled: bool,
    correct: bool,
    cheat_detected: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
) -> float:
    """Deterministic anti-cheat reward shaping for GPU kernel synthesis."""
    if cheat_detected:
        return -1.0
    if not compiled:
        return 0.0
    if not correct:
        return 0.15
    if speedup_vs_eager <= 1.0:
        return 0.50 + 0.50 * max(0.0, speedup_vs_eager)
    bonus_eager = math.log2(max(speedup_vs_eager, 1.0))
    bonus_compile = 0.5 * max(0.0, math.log2(max(speedup_vs_compile, 1e-6)))
    return min(4.0, 1.0 + bonus_eager + bonus_compile)


def compute_grpo_advantages(group_rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Group Relative Policy Optimization (GRPO) zero-mean, unit-std advantage normalization."""
    arr = np.asarray(group_rewards, dtype=np.float64)
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError("Group rewards must be a finite, one-dimensional sequence")
    if arr.size == 0:
        return []
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < eps:
        return [0.0 for _ in group_rewards]
    return [float((r - mean) / (std + eps)) for r in arr]
