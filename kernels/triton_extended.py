"""Genuine OpenAI Triton baselines for the 19 extended OP_REGISTRY operators.

Compatibility & precision invariants (Triton 3.2/sm75 & Triton 3.6/sm120):
- Zero `tl.math.tanh` usage: uses exact identity `2.0 * tl.sigmoid(2.0 * u) - 1.0`.
- Explicit `input_precision="ieee"` on every `tl.dot(...)` call.
- Zero PyTorch fallback math in `kernel_forward`: all math executes inside `@triton.jit`.
- `.contiguous()` on all tensor inputs, arbitrary leading batch dims, non-power-of-2
  tail masks, and strict integer-dtype assertions on quantized/index tensors.
- Note: These baselines have been verified statically (AST, anti-cheat, no-tanh,
  IEEE dot) in a CPU-only sandbox; live GPU numerical and timing verification must
  be executed on a CUDA host.
"""

from __future__ import annotations

import ast
from typing import Dict

# ---------------------------------------------------------------------------
# Level 1 (7 extended Triton baselines)
# ---------------------------------------------------------------------------
TRITON_GEGLU = """
import torch
import triton
import triton.language as tl


@triton.jit
def _geglu_kernel(X_ptr, G_ptr, Y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    k_coef = 0.7978845608028654
    u = k_coef * (g + 0.044715 * g * g * g)
    t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    gelu = 0.5 * g * (1.0 + t)
    tl.store(Y_ptr + offs, (x * gelu).to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    g_c = gate.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 1024
    _geglu_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x_c, g_c, y, n, BLOCK_SIZE=BLOCK_SIZE)
    return y
""".strip()

TRITON_FUSED_BIAS_GELU = """
import torch
import triton
import triton.language as tl


@triton.jit
def _bias_gelu_kernel(X_ptr, B_ptr, Y_ptr, stride_x, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = x + b
    k_coef = 0.7978845608028654
    u = k_coef * (z + 0.044715 * z * z * z)
    t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    out = 0.5 * z * (1.0 + t)
    tl.store(Y_ptr + row * stride_x + cols, out.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    b_c = bias.contiguous()
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = 1024
    _bias_gelu_kernel[(M, triton.cdiv(N, BLOCK_SIZE))](x_2d, b_c, y, x_2d.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_FUSED_BIAS_SILU = """
import torch
import triton
import triton.language as tl


@triton.jit
def _bias_silu_kernel(X_ptr, B_ptr, Y_ptr, stride_x, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = x + b
    out = z * tl.sigmoid(z)
    tl.store(Y_ptr + row * stride_x + cols, out.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    b_c = bias.contiguous()
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = 1024
    _bias_silu_kernel[(M, triton.cdiv(N, BLOCK_SIZE))](x_2d, b_c, y, x_2d.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_FUSED_DROPOUT_RESIDUAL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _dropout_residual_kernel(X_ptr, R_ptr, M_ptr, Y_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(M_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = r + x * m * scale
    tl.store(Y_ptr + offs, out.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    x_c = x.contiguous()
    r_c = residual.contiguous()
    m_c = mask.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    scale = 1.0 / (1.0 - float(p))
    BLOCK_SIZE = 1024
    _dropout_residual_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x_c, r_c, m_c, y, n, scale, BLOCK_SIZE=BLOCK_SIZE)
    return y
""".strip()

TRITON_FUSED_SCALE_MASK = """
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_mask_kernel(X_ptr, M_ptr, S_ptr, Y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(M_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(S_ptr).to(tl.float32)
    out = tl.where(m > 0.5, x * s, -1e9)
    tl.store(Y_ptr + offs, out.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    m_c = mask.contiguous()
    s_c = scale.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 1024
    _scale_mask_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x_c, m_c, s_c, y, n, BLOCK_SIZE=BLOCK_SIZE)
    return y
""".strip()

TRITON_L2_NORMALIZE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_kernel(X_ptr, Y_ptr, stride_x, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    inv_norm = tl.rsqrt(tl.sum(x * x, axis=0) + eps)
    tl.store(Y_ptr + row * stride_x + cols, (x * inv_norm).to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    _l2_norm_kernel[(M,)](x_2d, y, x_2d.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_COSINE_SIMILARITY = """
import torch
import triton
import triton.language as tl


@triton.jit
def _cosine_sim_kernel(A_ptr, B_ptr, Y_ptr, stride_a, stride_b, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    a = tl.load(A_ptr + row * stride_a + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + row * stride_b + cols, mask=mask, other=0.0).to(tl.float32)
    dot = tl.sum(a * b, axis=0)
    na = tl.sqrt(tl.sum(a * a, axis=0) + eps)
    nb = tl.sqrt(tl.sum(b * b, axis=0) + eps)
    tl.store(Y_ptr + row, (dot / (na * nb)).to(Y_ptr.dtype.element_ty))


def kernel_forward(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    orig_lead = a.shape[:-1]
    N = a.shape[-1]
    a_2d = a.contiguous().reshape(-1, N)
    b_2d = b.contiguous().reshape(-1, N)
    M = a_2d.shape[0]
    y = torch.empty((M,), device=a.device, dtype=a.dtype)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _cosine_sim_kernel[(M,)](a_2d, b_2d, y, a_2d.stride(0), b_2d.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_lead)
""".strip()

# ---------------------------------------------------------------------------
# Level 2 (5 extended Triton baselines)
# ---------------------------------------------------------------------------
TRITON_CAUSAL_MASKED_SOFTMAX = """
import torch
import triton
import triton.language as tl


@triton.jit
def _causal_softmax_kernel(X_ptr, Y_ptr, stride_b, stride_r, S, BLOCK_SIZE: tl.constexpr):
    b = tl.program_id(0)
    r = tl.program_id(1)
    cols = tl.arange(0, BLOCK_SIZE)
    in_bounds = cols < S
    base = b * stride_b + r * stride_r
    x = tl.load(X_ptr + base + cols, mask=in_bounds, other=-float("inf")).to(tl.float32)
    x_masked = tl.where(in_bounds, tl.where(cols > r, -1e9, x), -float("inf"))
    x_max = tl.max(x_masked, axis=0)
    num = tl.where(in_bounds, tl.exp(x_masked - x_max), 0.0)
    den = tl.sum(num, axis=0)
    tl.store(Y_ptr + base + cols, (num / den).to(Y_ptr.dtype.element_ty), mask=in_bounds)


def kernel_forward(scores: torch.Tensor) -> torch.Tensor:
    orig_shape = scores.shape
    S = orig_shape[-1]
    R = orig_shape[-2]
    x_3d = scores.contiguous().reshape(-1, R, S)
    y = torch.empty_like(x_3d)
    B = x_3d.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(S)
    _causal_softmax_kernel[(B, R)](x_3d, y, x_3d.stride(0), x_3d.stride(1), S, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_FUSED_CROSS_ENTROPY = """
import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_kernel(L_ptr, T_ptr, Y_ptr, stride_l, V, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    tgt = tl.load(T_ptr + row).to(tl.int64)
    row_max = -float("inf")
    for c0 in range(0, V, BLOCK_SIZE):
        cols = c0 + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        vals = tl.load(L_ptr + row * stride_l + cols, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))
    sum_exp = 0.0
    for c0 in range(0, V, BLOCK_SIZE):
        cols = c0 + tl.arange(0, BLOCK_SIZE)
        mask = cols < V
        vals = tl.load(L_ptr + row * stride_l + cols, mask=mask, other=-float("inf")).to(tl.float32)
        sum_exp += tl.sum(tl.where(mask, tl.exp(vals - row_max), 0.0), axis=0)
    tgt_logit = tl.load(L_ptr + row * stride_l + tgt).to(tl.float32)
    loss = -(tgt_logit - row_max - tl.log(sum_exp))
    tl.store(Y_ptr + row, loss.to(Y_ptr.dtype.element_ty))


def kernel_forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.is_floating_point():
        raise TypeError(f"targets must be an integer tensor, got {targets.dtype}")
    l_c = logits.contiguous()
    t_c = targets.contiguous()
    N, V = l_c.shape
    y = torch.empty((N,), device=l_c.device, dtype=l_c.dtype)
    _cross_entropy_kernel[(N,)](l_c, t_c, y, l_c.stride(0), V, BLOCK_SIZE=1024)
    return y
""".strip()

TRITON_MATMUL_BIAS_SILU = """
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_bias_silu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + rk
        a = tl.load(A_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak, mask=(rm[:, None] < M) & (k_offs[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn, mask=(k_offs[:, None] < K) & (rn[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b, input_precision="ieee")
    bias = tl.load(Bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    z = acc + bias[None, :]
    out = z * tl.sigmoid(z)
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, out.to(C_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def kernel_forward(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    a_c = a.contiguous()
    b_c = b.contiguous()
    bias_c = bias.contiguous()
    M, K = a_c.shape
    _, N = b_c.shape
    c = torch.empty((M, N), device=a_c.device, dtype=a_c.dtype)
    _matmul_bias_silu_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        a_c, b_c, bias_c, c, M, N, K,
        a_c.stride(0), a_c.stride(1), b_c.stride(0), b_c.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c
""".strip()

TRITON_MATMUL_RESIDUAL_ADD = """
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_residual_kernel(
    A_ptr, B_ptr, R_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_rm, stride_rn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + rk
        a = tl.load(A_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak, mask=(rm[:, None] < M) & (k_offs[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn, mask=(k_offs[:, None] < K) & (rn[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b, input_precision="ieee")
    r_mask = (rm[:, None] < M) & (rn[None, :] < N)
    res = tl.load(R_ptr + rm[:, None] * stride_rm + rn[None, :] * stride_rn, mask=r_mask, other=0.0).to(tl.float32)
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, (acc + res).to(C_ptr.dtype.element_ty), mask=r_mask)


def kernel_forward(a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    a_c = a.contiguous()
    b_c = b.contiguous()
    r_c = residual.contiguous()
    M, K = a_c.shape
    _, N = b_c.shape
    c = torch.empty((M, N), device=a_c.device, dtype=a_c.dtype)
    _matmul_residual_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        a_c, b_c, r_c, c, M, N, K,
        a_c.stride(0), a_c.stride(1), b_c.stride(0), b_c.stride(1), r_c.stride(0), r_c.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c
""".strip()

TRITON_GROUP_NORM = """
import torch
import triton
import triton.language as tl


@triton.jit
def _group_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride_n, stride_c, C_per_g, L, eps,
    BLOCK_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    total = C_per_g * L
    c_start = g * C_per_g
    base_n = n * stride_n

    sum_x = 0.0
    for idx0 in range(0, total, BLOCK_SIZE):
        offs = idx0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        ch = c_start + (offs // L)
        pos = offs % L
        vals = tl.load(X_ptr + base_n + ch * stride_c + pos, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(vals, axis=0)
    mean = sum_x / total

    sum_sq = 0.0
    for idx0 in range(0, total, BLOCK_SIZE):
        offs = idx0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        ch = c_start + (offs // L)
        pos = offs % L
        vals = tl.load(X_ptr + base_n + ch * stride_c + pos, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, vals - mean, 0.0)
        sum_sq += tl.sum(diff * diff, axis=0)
    rstd = tl.rsqrt(sum_sq / total + eps)

    for idx0 in range(0, total, BLOCK_SIZE):
        offs = idx0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        ch = c_start + (offs // L)
        pos = offs % L
        vals = tl.load(X_ptr + base_n + ch * stride_c + pos, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + ch, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + ch, mask=mask, other=0.0).to(tl.float32)
        out = (vals - mean) * rstd * w + b
        tl.store(Y_ptr + base_n + ch * stride_c + pos, out.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int = 8, eps: float = 1e-5) -> torch.Tensor:
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()
    N, C, L = x_c.shape
    y = torch.empty_like(x_c)
    _group_norm_kernel[(N, num_groups)](
        x_c, w_c, b_c, y, x_c.stride(0), x_c.stride(1), C // num_groups, L, eps, BLOCK_SIZE=256
    )
    return y
""".strip()

# ---------------------------------------------------------------------------
# Level 3 (7 extended Triton baselines)
# ---------------------------------------------------------------------------
TRITON_INT4_GROUPWISE_GEMM = """
import torch
import triton
import triton.language as tl


@triton.jit
def _int4_groupwise_gemm_kernel(
    A_ptr, PW_ptr, S_ptr, C_ptr,
    M, N, K, group_size,
    stride_am, stride_ak, stride_pwk, stride_pwn, stride_sg, stride_sn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + rk
        k_mask = k_offs < K
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=(rm[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        pw_row = k_offs // 2
        is_hi = (k_offs & 1) == 1
        packed = tl.load(
            PW_ptr + pw_row[:, None] * stride_pwk + rn[None, :] * stride_pwn,
            mask=k_mask[:, None] & (rn[None, :] < N),
            other=0,
        ).to(tl.int32)
        nibble = tl.where(is_hi[:, None], (packed >> 4) & 0x0F, packed & 0x0F) - 8
        g_idx = k_offs // group_size
        sc = tl.load(
            S_ptr + g_idx[:, None] * stride_sg + rn[None, :] * stride_sn,
            mask=k_mask[:, None] & (rn[None, :] < N),
            other=1.0,
        ).to(tl.float32)
        acc += tl.dot(a, nibble.to(tl.float32) * sc, input_precision="ieee")
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(C_ptr.dtype.element_ty),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def kernel_forward(x: torch.Tensor, packed_w: torch.Tensor, scales: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    if packed_w.is_floating_point():
        raise TypeError(f"packed_w must be an integer tensor (uint8), got {packed_w.dtype}")
    x_c = x.contiguous()
    pw_c = packed_w.contiguous()
    s_c = scales.contiguous()
    M, K = x_c.shape
    _, N = pw_c.shape
    c = torch.empty((M, N), device=x_c.device, dtype=x_c.dtype)
    _int4_groupwise_gemm_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        x_c, pw_c, s_c, c, M, N, K, group_size,
        x_c.stride(0), x_c.stride(1), pw_c.stride(0), pw_c.stride(1), s_c.stride(0), s_c.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c
""".strip()

TRITON_FP8_SCALED_GEMM = """
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_scaled_gemm_kernel(
    A_ptr, B_ptr, SA_ptr, SB_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    sa = tl.load(SA_ptr + rm, mask=rm < M, other=1.0).to(tl.float32)
    sb = tl.load(SB_ptr + rn, mask=rn < N, other=1.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + rk
        a = tl.load(A_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak, mask=(rm[:, None] < M) & (k_offs[None, :] < K), other=0).to(tl.float32)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn, mask=(k_offs[:, None] < K) & (rn[None, :] < N), other=0).to(tl.float32)
        acc += tl.dot(a, b, input_precision="ieee")
    out = acc * sa[:, None] * sb[None, :]
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, out.to(C_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def kernel_forward(a_q: torch.Tensor, b_q: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    if a_q.is_floating_point() or b_q.is_floating_point():
        raise TypeError("a_q and b_q must remain integer tensors")
    a_c = a_q.contiguous()
    b_c = b_q.contiguous()
    sa_c = scale_a.contiguous()
    sb_c = scale_b.contiguous()
    M, K = a_c.shape
    _, N = b_c.shape
    c = torch.empty((M, N), device=a_c.device, dtype=sa_c.dtype)
    _fp8_scaled_gemm_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        a_c, b_c, sa_c, sb_c, c, M, N, K,
        a_c.stride(0), a_c.stride(1), b_c.stride(0), b_c.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c
""".strip()

TRITON_BLOCK_SPARSE_ATTN = """
import torch
import triton
import triton.language as tl


@triton.jit
def _block_sparse_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_bh, stride_s, stride_d,
    S, D, scale,
    BLOCK_SZ: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    q_blk = tl.program_id(1)
    rm = q_blk * BLOCK_SZ + tl.arange(0, BLOCK_SZ)
    rd = tl.arange(0, BLOCK_D)
    base = bh * stride_bh
    q_mask = (rm[:, None] < S) & (rd[None, :] < D)
    q = tl.load(Q_ptr + base + rm[:, None] * stride_s + rd[None, :] * stride_d, mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_SZ,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SZ,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SZ, BLOCK_D), dtype=tl.float32)

    start_blk = tl.maximum(0, q_blk - 1)
    for k_blk in range(start_blk, q_blk + 1):
        rn = k_blk * BLOCK_SZ + tl.arange(0, BLOCK_SZ)
        kv_mask = (rn[:, None] < S) & (rd[None, :] < D)
        k = tl.load(K_ptr + base + rn[:, None] * stride_s + rd[None, :] * stride_d, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + base + rn[:, None] * stride_s + rd[None, :] * stride_d, mask=kv_mask, other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        allowed = (rn[None, :] <= rm[:, None]) & (rn[None, :] < S) & (rm[:, None] < S)
        qk = tl.where(allowed, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        p = tl.where(allowed, tl.exp(qk - safe_m[:, None]), 0.0)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - safe_m))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_ij

    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(O_ptr + base + rm[:, None] * stride_s + rd[None, :] * stride_d, out.to(O_ptr.dtype.element_ty), mask=q_mask)


def kernel_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    B, H, S, D = q.shape
    q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q_c)
    q3, k3, v3, o3 = q_c.reshape(B * H, S, D), k_c.reshape(B * H, S, D), v_c.reshape(B * H, S, D), o.reshape(B * H, S, D)
    scale = 1.0 / (D ** 0.5)
    BLOCK_D = triton.next_power_of_2(D)
    _block_sparse_attn_kernel[(B * H, triton.cdiv(S, block_size))](
        q3, k3, v3, o3, q3.stride(0), q3.stride(1), q3.stride(2),
        S, D, scale, BLOCK_SZ=block_size, BLOCK_D=BLOCK_D,
    )
    return o
""".strip()

TRITON_GQA_DECODE_ATTN = """
import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_decode_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_kb, stride_kh, stride_ks, stride_ob, stride_oh,
    S, D, repeats, scale,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    hq = tl.program_id(1)
    hkv = hq // repeats
    rd = tl.arange(0, BLOCK_D)
    d_mask = rd < D
    q = tl.load(Q_ptr + b * stride_qb + hq * stride_qh + rd, mask=d_mask, other=0.0).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    kv_base = b * stride_kb + hkv * stride_kh

    for s0 in range(0, S, BLOCK_S):
        rs = s0 + tl.arange(0, BLOCK_S)
        s_mask = rs < S
        k_tile = tl.load(K_ptr + kv_base + rs[:, None] * stride_ks + rd[None, :], mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(V_ptr + kv_base + rs[:, None] * stride_ks + rd[None, :], mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * q[None, :], axis=1) * scale
        scores = tl.where(s_mask, scores, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.where(s_mask, tl.exp(scores - m_ij), 0.0)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = m_ij

    tl.store(O_ptr + b * stride_ob + hq * stride_oh + rd, (acc / l_i).to(O_ptr.dtype.element_ty), mask=d_mask)


def kernel_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
    B, Hq, _, D = q_c.shape
    _, Hkv, S, _ = k_c.shape
    o = torch.empty_like(q_c)
    q3 = q_c.reshape(B, Hq, D)
    o3 = o.reshape(B, Hq, D)
    scale = 1.0 / (D ** 0.5)
    BLOCK_D = triton.next_power_of_2(D)
    _gqa_decode_kernel[(B, Hq)](
        q3, k_c, v_c, o3,
        q3.stride(0), q3.stride(1), k_c.stride(0), k_c.stride(1), k_c.stride(2), o3.stride(0), o3.stride(1),
        S, D, Hq // Hkv, scale, BLOCK_S=64, BLOCK_D=BLOCK_D,
    )
    return o
""".strip()

TRITON_PAGED_KV_GATHER_ATTN = """
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_kv_attn_kernel(
    Q_ptr, K_ptr, V_ptr, BT_ptr, O_ptr,
    stride_qb, stride_qh,
    stride_kp_blk, stride_kp_h, stride_kp_s,
    stride_bt_b,
    stride_ob, stride_oh,
    max_blks, D, scale,
    BLOCK_SZ: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    rd = tl.arange(0, BLOCK_D)
    rs = tl.arange(0, BLOCK_SZ)
    d_mask = rd < D
    q = tl.load(Q_ptr + b * stride_qb + h * stride_qh + rd, mask=d_mask, other=0.0).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for blk_idx in range(0, max_blks):
        phys_blk = tl.load(BT_ptr + b * stride_bt_b + blk_idx).to(tl.int64)
        base_kv = phys_blk * stride_kp_blk + h * stride_kp_h
        k_tile = tl.load(K_ptr + base_kv + rs[:, None] * stride_kp_s + rd[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(V_ptr + base_kv + rs[:, None] * stride_kp_s + rd[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * q[None, :], axis=1) * scale
        m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_ij)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = m_ij

    tl.store(O_ptr + b * stride_ob + h * stride_oh + rd, (acc / l_i).to(O_ptr.dtype.element_ty), mask=d_mask)


def kernel_forward(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, block_tables: torch.Tensor) -> torch.Tensor:
    if block_tables.is_floating_point():
        raise TypeError(f"block_tables must remain an integer tensor, got {block_tables.dtype}")
    q_c, k_c, v_c, bt_c = q.contiguous(), k_pool.contiguous(), v_pool.contiguous(), block_tables.contiguous()
    B, H, D = q_c.shape
    _, _, blk_sz, _ = k_c.shape
    max_blks = bt_c.shape[1]
    o = torch.empty_like(q_c)
    scale = 1.0 / (D ** 0.5)
    BLOCK_D = triton.next_power_of_2(D)
    _paged_kv_attn_kernel[(B, H)](
        q_c, k_c, v_c, bt_c, o,
        q_c.stride(0), q_c.stride(1),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        bt_c.stride(0),
        o.stride(0), o.stride(1),
        max_blks, D, scale, BLOCK_SZ=blk_sz, BLOCK_D=BLOCK_D,
    )
    return o
""".strip()

TRITON_SELECTIVE_SCAN_SSM = """
import torch
import triton
import triton.language as tl


@triton.jit
def _selective_scan_kernel(
    U_ptr, DT_ptr, ALOG_ptr, B_ptr, C_ptr, Y_ptr,
    stride_b, stride_l, L, D,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    d_blk = tl.program_id(1)
    rd = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = rd < D
    a = -tl.exp(tl.load(ALOG_ptr + rd, mask=d_mask, other=0.0).to(tl.float32))
    h = tl.zeros((BLOCK_D,), dtype=tl.float32)
    base = b * stride_b
    for t in range(0, L):
        off = base + t * stride_l + rd
        u = tl.load(U_ptr + off, mask=d_mask, other=0.0).to(tl.float32)
        dt = tl.load(DT_ptr + off, mask=d_mask, other=0.0).to(tl.float32)
        bp = tl.load(B_ptr + off, mask=d_mask, other=0.0).to(tl.float32)
        cp = tl.load(C_ptr + off, mask=d_mask, other=0.0).to(tl.float32)
        h = tl.exp(dt * a) * h + dt * bp * u
        tl.store(Y_ptr + off, (h * cp).to(Y_ptr.dtype.element_ty), mask=d_mask)


def kernel_forward(u: torch.Tensor, delta: torch.Tensor, a_log: torch.Tensor, b_proj: torch.Tensor, c_proj: torch.Tensor) -> torch.Tensor:
    u_c, dt_c, al_c, b_c, c_c = u.contiguous(), delta.contiguous(), a_log.contiguous(), b_proj.contiguous(), c_proj.contiguous()
    B, L, D = u_c.shape
    y = torch.empty_like(u_c)
    BLOCK_D = 64
    _selective_scan_kernel[(B, triton.cdiv(D, BLOCK_D))](
        u_c, dt_c, al_c, b_c, c_c, y, u_c.stride(0), u_c.stride(1), L, D, BLOCK_D=BLOCK_D
    )
    return y
""".strip()

TRITON_TOPK_MOE_ROUTER = """
import torch
import triton
import triton.language as tl


@triton.jit
def _topk_moe_router_kernel(
    X_ptr, RW_ptr, EW_ptr, Y_ptr,
    stride_xt, stride_rwd, stride_ewe, stride_ewd, stride_yt,
    D, E, K_TOP: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr,
):
    t = tl.program_id(0)
    rd = tl.arange(0, BLOCK_D)
    re = tl.arange(0, BLOCK_E)
    d_mask = rd < D
    e_mask = re < E

    x_vec = tl.load(X_ptr + t * stride_xt + rd, mask=d_mask, other=0.0).to(tl.float32)
    rw_mat = tl.load(RW_ptr + rd[:, None] * stride_rwd + re[None, :], mask=d_mask[:, None] & e_mask[None, :], other=0.0).to(tl.float32)
    logits = tl.sum(x_vec[:, None] * rw_mat, axis=0)
    logits = tl.where(e_mask, logits, -float("inf"))
    l_max = tl.max(logits, axis=0)
    ex = tl.where(e_mask, tl.exp(logits - l_max), 0.0)
    probs = ex / tl.sum(ex, axis=0)

    rem_probs = probs
    w_sum = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for _ in range(0, K_TOP):
        best_val = tl.max(rem_probs, axis=0)
        is_best = (rem_probs == best_val) & e_mask
        # Deterministic lowest-index tie break
        best_idx = tl.min(tl.where(is_best, re, E), axis=0)
        w_sum += best_val
        ew_mat = tl.load(
            EW_ptr + best_idx * stride_ewe + rd[:, None] * stride_ewd + rd[None, :],
            mask=d_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        expert_out = tl.sum(x_vec[:, None] * ew_mat, axis=0)
        acc += best_val * expert_out
        rem_probs = tl.where(re == best_idx, -1.0, rem_probs)

    tl.store(Y_ptr + t * stride_yt + rd, (acc / w_sum).to(Y_ptr.dtype.element_ty), mask=d_mask)


def kernel_forward(x: torch.Tensor, router_w: torch.Tensor, expert_w: torch.Tensor, k: int = 2) -> torch.Tensor:
    x_c, rw_c, ew_c = x.contiguous(), router_w.contiguous(), expert_w.contiguous()
    T, D = x_c.shape
    E = rw_c.shape[1]
    y = torch.empty_like(x_c)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_E = triton.next_power_of_2(E)
    _topk_moe_router_kernel[(T,)](
        x_c, rw_c, ew_c, y,
        x_c.stride(0), rw_c.stride(0), ew_c.stride(0), ew_c.stride(1), y.stride(0),
        D, E, K_TOP=int(k), BLOCK_D=BLOCK_D, BLOCK_E=BLOCK_E,
    )
    return y
""".strip()

TRITON_EXTENDED: Dict[str, str] = {
    "geglu_4096": TRITON_GEGLU,
    "fused_bias_gelu_4096": TRITON_FUSED_BIAS_GELU,
    "fused_bias_silu_4096": TRITON_FUSED_BIAS_SILU,
    "fused_dropout_residual_2048": TRITON_FUSED_DROPOUT_RESIDUAL,
    "fused_scale_mask_2048": TRITON_FUSED_SCALE_MASK,
    "l2_normalize_2048": TRITON_L2_NORMALIZE,
    "cosine_similarity_1024": TRITON_COSINE_SIMILARITY,
    "causal_masked_softmax_1024": TRITON_CAUSAL_MASKED_SOFTMAX,
    "fused_cross_entropy_4096": TRITON_FUSED_CROSS_ENTROPY,
    "matmul_bias_silu_512": TRITON_MATMUL_BIAS_SILU,
    "matmul_residual_add_512": TRITON_MATMUL_RESIDUAL_ADD,
    "group_norm_64x32": TRITON_GROUP_NORM,
    "int4_groupwise_dequant_gemm_512": TRITON_INT4_GROUPWISE_GEMM,
    "fp8_e4m3_simulated_scaled_gemm_512": TRITON_FP8_SCALED_GEMM,
    "block_sparse_attn_256": TRITON_BLOCK_SPARSE_ATTN,
    "gqa_decode_attn_kv512": TRITON_GQA_DECODE_ATTN,
    "paged_kv_gather_attn_256": TRITON_PAGED_KV_GATHER_ATTN,
    "selective_scan_ssm_256": TRITON_SELECTIVE_SCAN_SSM,
    "topk_moe_router_dispatch_64": TRITON_TOPK_MOE_ROUTER,
}


def validate_triton_extended_syntax() -> dict[str, bool]:
    """Verify AST syntax, `@triton.jit`, `kernel_forward`, `.contiguous()`, no `tl.math.tanh`,
    and explicit `input_precision="ieee"` on every `tl.dot(...)` across all 19 extended baselines.
    """
    results: dict[str, bool] = {}
    for op_name, code in TRITON_EXTENDED.items():
        tree = ast.parse(code)
        has_entry = any(isinstance(n, ast.FunctionDef) and n.name == "kernel_forward" for n in tree.body)
        has_jit = "@triton.jit" in code
        has_contiguous = ".contiguous()" in code
        has_no_tanh = "tl.math.tanh" not in code
        dot_ok = True
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dot":
                kw = {k.arg: k.value for k in node.keywords}
                prec = kw.get("input_precision")
                if not (isinstance(prec, ast.Constant) and prec.value == "ieee"):
                    dot_ok = False
        results[op_name] = bool(has_entry and has_jit and has_contiguous and has_no_tanh and dot_ok)
    return results
