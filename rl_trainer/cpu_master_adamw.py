"""CPU FP32 master-weights AdamW optimizer for BF16/FP16 full-parameter training on 8 GB GPUs.

Solves two laptop-GPU bottlenecks simultaneously:
1. Precision: Direct BF16 updates at LR=1e-6 vanish below BF16 ULP (~7.8e-3);
   maintaining FP32 master weights on CPU preserves exact small-LR updates.
2. VRAM: AdamW FP32 moments (m, v) and master weights live in host RAM (e.g. 32 GB)
   instead of competing for 8 GB GPU VRAM.

Preserves exact all-parameter trainability and full checkpoint save/resume of
both FP32 master weights and AdamW moments.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np


class CpuMasterAdamW:
    """CPU FP32 master-weight wrapper around `torch.optim.AdamW` for low-VRAM BF16 models."""

    def __init__(
        self,
        model_params: Iterable[Any],
        lr: float = 1e-6,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        import torch  # type: ignore

        self._torch = torch
        self.model_params: list[Any] = [p for p in model_params if getattr(p, "requires_grad", False)]
        if not self.model_params:
            raise ValueError("CpuMasterAdamW requires at least one trainable model parameter.")

        # Full FP32 CPU master copies of every trainable parameter
        self.master_params: list[Any] = [
            torch.nn.Parameter(p.detach().float().cpu().clone(), requires_grad=True)
            for p in self.model_params
        ]
        self.optimizer = torch.optim.AdamW(
            self.master_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def master_weights_sha256(self) -> str:
        h = hashlib.sha256()
        for m in self.master_params:
            h.update(m.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def copy_grads_and_step(self, max_grad_norm: float) -> tuple[bool, float]:
        """Copy unscaled CUDA/model grads to CPU FP32 masters, verify finiteness, step, and copy back.

        Returns `(updated: bool, fp32_weight_l2_delta: float)`.
        """
        torch = self._torch
        saw_grad = False
        for mp, cp in zip(self.model_params, self.master_params):
            if mp.grad is None:
                cp.grad = None
                continue
            saw_grad = True
            g_cpu = mp.grad.detach().float().cpu()
            if not bool(torch.isfinite(g_cpu).all()):
                self.zero_grad(set_to_none=True)
                return False, 0.0
            cp.grad = g_cpu

        if not saw_grad:
            return False, 0.0

        grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, max_grad_norm)
        if not bool(torch.isfinite(grad_norm)) or any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.master_params
        ):
            self.zero_grad(set_to_none=True)
            return False, 0.0
        before = [m.detach().clone() for m in self.master_params]
        self.optimizer.step()

        sq_sum = 0.0
        with torch.no_grad():
            for mp, cp, b in zip(self.model_params, self.master_params, before):
                diff = (cp.detach() - b).numpy()
                sq_sum += float(np.sum(diff * diff, dtype=np.float64))
                # Copy updated FP32 master weight back to model's device & BF16/FP16/FP32 dtype
                mp.data.copy_(cp.data.to(device=mp.device, dtype=mp.dtype))

        self.zero_grad(set_to_none=True)
        return True, float(np.sqrt(sq_sum))

    def state_dict(self) -> dict[str, Any]:
        return {
            "master_params": [m.detach().cpu().clone() for m in self.master_params],
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        torch = self._torch
        saved_masters = state["master_params"]
        if len(saved_masters) != len(self.master_params):
            raise ValueError(
                f"CpuMasterAdamW master_params count mismatch: {len(saved_masters)} vs {len(self.master_params)}"
            )
        with torch.no_grad():
            for mp, cp, saved in zip(self.model_params, self.master_params, saved_masters):
                cp.data.copy_(saved.to(device="cpu", dtype=torch.float32))
                mp.data.copy_(cp.data.to(device=mp.device, dtype=mp.dtype))
        self.optimizer.load_state_dict(state["optimizer"])
