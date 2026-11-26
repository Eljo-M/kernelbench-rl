"""Native PyTorch reference implementations for the 11 Triton-baseline operators.

All PyTorch imports are strictly lazy (inside functions) so importing this
module is safe on CPU-only environments without PyTorch installed.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for native torch references but is not installed in this environment."
        ) from exc
    return torch


def is_integer_numpy_array(arr: np.ndarray) -> bool:
    """Return True if a NumPy array holds integer (quantized/index) data that must stay integer."""
    return bool(np.issubdtype(arr.dtype, np.integer))


def resolve_input_dtype_plan(inputs: Sequence[np.ndarray], target_float_dtype: str) -> list[str]:
    """Pure-Python helper describing how each input array maps to a target dtype.

    Integer tensors (e.g. int8 qweights, int32 block tables, int64 targets) strictly
    preserve their original integer dtype; floating-point tensors map to `target_float_dtype`.
    """
    if target_float_dtype not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported target_float_dtype '{target_float_dtype}'. Expected 'fp32' or 'bf16'.")
    plan: list[str] = []
    for arr in inputs:
        if is_integer_numpy_array(arr):
            plan.append(str(arr.dtype))
        elif np.issubdtype(arr.dtype, np.floating):
            plan.append(target_float_dtype)
        else:
            plan.append(str(arr.dtype))
    return plan


def convert_numpy_inputs_to_torch(
    inputs: Sequence[np.ndarray],
    device: str = "cuda",
    float_dtype: str = "fp32",
) -> tuple[Any, ...]:
    """Convert NumPy input arrays to contiguous PyTorch tensors while preserving integer dtypes."""
    torch = _require_torch()
    if float_dtype == "fp32":
        tdtype = torch.float32
    elif float_dtype == "bf16":
        tdtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported float_dtype '{float_dtype}'. Expected 'fp32' or 'bf16'.")

    tensors = []
    for arr in inputs:
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if np.issubdtype(arr.dtype, np.floating):
            t = t.to(device=device, dtype=tdtype)
        else:
            # Strictly preserve integer dtypes (e.g., int8 qweights, int32/int64 indices)
            t = t.to(device=device)
        tensors.append(t.contiguous())
    return tuple(tensors)


def torch_inputs_to_numpy_fp32(tensors: Sequence[Any]) -> tuple[np.ndarray, ...]:
    """Convert PyTorch inputs back to NumPy for exact quantized-input reference computation.

    Floating-point tensors (e.g. bf16) are converted to fp32 NumPy arrays preserving their
    exact quantized values; integer tensors strictly preserve their integer dtype.
    """
    out: list[np.ndarray] = []
    for t in tensors:
        if hasattr(t, "is_floating_point") and t.is_floating_point():
            arr = t.detach().float().cpu().numpy()
        elif hasattr(t, "detach"):
            arr = t.detach().cpu().numpy()
        else:
            arr = np.asarray(t)
        out.append(np.ascontiguousarray(arr))
    return tuple(out)


def _torch_rmsnorm(x, weight, eps: float = 1e-6):
    torch = _require_torch()
    x_f32 = x.float()
    w_f32 = weight.float()
    rms = torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)
    return (x_f32 * rms * w_f32).to(x.dtype)


def _torch_layernorm(x, weight, bias, eps: float = 1e-5):
    torch = _require_torch()
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.var(dim=-1, keepdim=True, unbiased=False)
    y = (x_f32 - mean) * torch.rsqrt(var + eps)
    return (y * weight.float() + bias.float()).to(x.dtype)


def _torch_swiglu(x, gate):
    torch = _require_torch()
    g = gate.float()
    y = x.float() * (g * torch.sigmoid(g))
    return y.to(x.dtype)


def _torch_fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    torch = _require_torch()
    h = x.float() + residual.float()
    rms = torch.rsqrt(torch.mean(h * h, dim=-1, keepdim=True) + eps)
    return (h * rms * weight.float()).to(x.dtype)


def _torch_softmax(x):
    torch = _require_torch()
    x_f32 = x.float()
    shifted = x_f32 - x_f32.max(dim=-1, keepdim=True).values
    ex = torch.exp(shifted)
    return (ex / ex.sum(dim=-1, keepdim=True)).to(x.dtype)


def _torch_log_softmax(x):
    torch = _require_torch()
    x_f32 = x.float()
    m = x_f32.max(dim=-1, keepdim=True).values
    shifted = x_f32 - m
    lse = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    return (shifted - lse).to(x.dtype)


def _torch_rope_fwd(x, cos, sin):
    torch = _require_torch()
    half = x.shape[-1] // 2
    x1 = x[..., :half].float()
    x2 = x[..., half:].float()
    c = cos.float()[None, None, :, :]
    s = sin.float()[None, None, :, :]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


def _torch_matmul_bias_gelu(a, b, bias):
    torch = _require_torch()
    z = torch.matmul(a.float(), b.float()) + bias.float()
    k = math.sqrt(2.0 / math.pi)
    y = 0.5 * z * (1.0 + torch.tanh(k * (z + 0.044715 * z * z * z)))
    return y.to(a.dtype)


def _torch_int8_weight_only_gemm(x, qweight, scales):
    torch = _require_torch()
    if qweight.is_floating_point():
        raise TypeError(f"qweight must remain an integer tensor, got {qweight.dtype}")
    w_dequant = qweight.to(torch.float32) * scales.float().unsqueeze(0)
    return torch.matmul(x.float(), w_dequant).to(x.dtype)


def _torch_sdpa_causal(q, k, v):
    torch = _require_torch()
    qf, kf, vf = q.float(), k.float(), v.float()
    s = qf.shape[-2]
    d = qf.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    causal = torch.triu(torch.ones((s, s), device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    probs = torch.exp(shifted)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.matmul(probs, vf).to(q.dtype)


TORCH_REFERENCES: dict[str, Callable[..., Any]] = {
    "rmsnorm_2048": _torch_rmsnorm,
    "rmsnorm_4096": _torch_rmsnorm,
    "layernorm_2048": _torch_layernorm,
    "swiglu_4096": _torch_swiglu,
    "fused_add_rmsnorm_2048": _torch_fused_add_rmsnorm,
    "fused_softmax_2048": _torch_softmax,
    "log_softmax_2048": _torch_log_softmax,
    "rope_fwd_128": _torch_rope_fwd,
    "matmul_bias_gelu_512": _torch_matmul_bias_gelu,
    "int8_weight_only_gemm_512": _torch_int8_weight_only_gemm,
    "scaled_dot_product_attn_128": _torch_sdpa_causal,
}
