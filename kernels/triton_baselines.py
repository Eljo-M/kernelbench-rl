"""Hand-tuned OpenAI Triton baseline kernels for KernelBench-RL (CUDA GPU targets).

Design notes:
- Every `tl.dot(...)` explicitly specifies `input_precision="ieee"` so FP32 runs
  do not silently downgrade to TF32 and violate tight numerical tolerances.
- Every `kernel_forward(...)` enforces `.contiguous()` inputs, handles arbitrary
  leading batch dimensions where applicable, accumulates in `tl.float32`, and
  casts outputs cleanly back to the input floating dtype (`fp32` or `bf16`).
- Integer quantization weights (e.g. `qweight: torch.int8`) are asserted to
  remain non-floating integer tensors and dequantized inside registers.
- In CPU-only CI environments without CUDA/Triton installed, every baseline is
  validated statically via `validate_triton_baselines_syntax()`; live GPU
  numerical equivalence and CUDA Event timing are validated via
  `python3 -m kernels.validate_cuda`.
"""

from __future__ import annotations

import ast
from typing import Dict

TRITON_RMSNORM_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(X_ptr, W_ptr, Y_ptr, stride_x, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)
    y = x * rstd * w
    tl.store(Y_ptr + row * stride_x + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    w = weight.contiguous()
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 65536:
        raise ValueError(f"Unsupported hidden size N={N} > 65536 for single-block RMSNorm")
    _rmsnorm_fwd_kernel[(M,)](x_2d, w, y, x_2d.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_LAYERNORM_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd_kernel(X_ptr, W_ptr, B_ptr, Y_ptr, stride_x, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b
    tl.store(Y_ptr + row * stride_x + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    w = weight.contiguous()
    b = bias.contiguous()
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 65536:
        raise ValueError(f"Unsupported hidden size N={N} > 65536 for single-block LayerNorm")
    _layernorm_fwd_kernel[(M,)](x_2d, w, b, y, x_2d.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_SWIGLU_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(X_ptr, G_ptr, Y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu = g * tl.sigmoid(g)
    tl.store(Y_ptr + offs, (x * silu).to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    g_c = gate.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _swiglu_fwd_kernel[grid](x_c, g_c, y, n, BLOCK_SIZE=BLOCK_SIZE)
    return y
""".strip()

TRITON_FUSED_ADD_RMSNORM_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(X_ptr, R_ptr, W_ptr, Y_ptr, stride_x, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    rstd = tl.rsqrt(tl.sum(h * h, axis=0) / N + eps)
    tl.store(Y_ptr + row * stride_x + cols, (h * rstd * w).to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    r_2d = residual.contiguous().reshape(-1, orig_shape[-1])
    w = weight.contiguous()
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 65536:
        raise ValueError(f"Unsupported hidden size N={N} > 65536 for fused Add+RMSNorm")
    _add_rmsnorm_kernel[(M,)](x_2d, r_2d, w, y, x_2d.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_SOFTMAX_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd_kernel(X_ptr, Y_ptr, stride_x, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x_max = tl.max(x, axis=0)
    num = tl.exp(x - x_max)
    den = tl.sum(num, axis=0)
    tl.store(Y_ptr + row * stride_x + cols, (num / den).to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 65536:
        raise ValueError(f"Unsupported row size N={N} > 65536 for single-block Softmax")
    _softmax_fwd_kernel[(M,)](x_2d, y, x_2d.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_LOG_SOFTMAX_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax_kernel(X_ptr, Y_ptr, stride_x, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x_max = tl.max(x, axis=0)
    shifted = x - x_max
    lse = tl.log(tl.sum(tl.exp(shifted), axis=0))
    tl.store(Y_ptr + row * stride_x + cols, (shifted - lse).to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 65536:
        raise ValueError(f"Unsupported row size N={N} > 65536 for single-block LogSoftmax")
    _log_softmax_kernel[(M,)](x_2d, y, x_2d.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)
""".strip()

TRITON_ROPE_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(
    X_ptr, COS_ptr, SIN_ptr, Y_ptr,
    stride_bh, stride_s, stride_cos_s,
    HALF_D, BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    s = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < HALF_D
    base = bh * stride_bh + s * stride_s
    x1 = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X_ptr + base + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(COS_ptr + s * stride_cos_s + offs, mask=mask, other=1.0).to(tl.float32)
    sn = tl.load(SIN_ptr + s * stride_cos_s + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y_ptr + base + offs, (x1 * c - x2 * sn).to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(Y_ptr + base + HALF_D + offs, (x2 * c + x1 * sn).to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE expects 4D tensor [B, H, S, D] with even D, got {tuple(x.shape)}")
    B, H, S, D = x.shape
    x_c = x.contiguous()
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()
    y = torch.empty_like(x_c)
    x_3d = x_c.reshape(B * H, S, D)
    y_3d = y.reshape(B * H, S, D)
    HALF_D = D // 2
    BLOCK_D = triton.next_power_of_2(HALF_D)
    _rope_fwd_kernel[(B * H, S)](
        x_3d, cos_c, sin_c, y_3d,
        x_3d.stride(0), x_3d.stride(1), cos_c.stride(0),
        HALF_D, BLOCK_D=BLOCK_D,
    )
    return y
""".strip()

TRITON_MATMUL_BIAS_GELU_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_bias_gelu_kernel(
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
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(k_offs[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b, input_precision="ieee")
    bias = tl.load(Bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    z = acc + bias[None, :]
    k_coef = 0.7978845608028654
    inner = k_coef * (z + 0.044715 * z * z * z)
    t = tl.math.tanh(inner)
    out = 0.5 * z * (1.0 + t)
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        out.to(C_ptr.dtype.element_ty),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def kernel_forward(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    a_c = a.contiguous()
    b_c = b.contiguous()
    bias_c = bias.contiguous()
    M, K = a_c.shape
    K_b, N = b_c.shape
    if K != K_b:
        raise ValueError(f"Inner dimension mismatch: {K} vs {K_b}")
    c = torch.empty((M, N), device=a_c.device, dtype=a_c.dtype)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_bias_gelu_kernel[grid](
        a_c, b_c, bias_c, c, M, N, K,
        a_c.stride(0), a_c.stride(1), b_c.stride(0), b_c.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c
""".strip()

TRITON_INT8_WEIGHT_ONLY_GEMM_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _int8_wo_gemm_kernel(
    A_ptr, W_ptr, S_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    scales = tl.load(S_ptr + rn, mask=rn < N, other=1.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + rk
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        w_q = tl.load(
            W_ptr + k_offs[:, None] * stride_wk + rn[None, :] * stride_wn,
            mask=(k_offs[:, None] < K) & (rn[None, :] < N),
            other=0,
        ).to(tl.float32)
        acc += tl.dot(a, w_q * scales[None, :], input_precision="ieee")
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(C_ptr.dtype.element_ty),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def kernel_forward(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if qweight.is_floating_point():
        raise TypeError(f"qweight must be an integer tensor (e.g. torch.int8), got {qweight.dtype}")
    x_c = x.contiguous()
    w_c = qweight.contiguous()
    s_c = scales.contiguous()
    M, K = x_c.shape
    K_w, N = w_c.shape
    if K != K_w:
        raise ValueError(f"Inner dimension mismatch: {K} vs {K_w}")
    c = torch.empty((M, N), device=x_c.device, dtype=x_c.dtype)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _int8_wo_gemm_kernel[grid](
        x_c, w_c, s_c, c, M, N, K,
        x_c.stride(0), x_c.stride(1), w_c.stride(0), w_c.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c
""".strip()

TRITON_FLASH_CAUSAL_ATTN_BASELINE = """
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_causal_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_bh, stride_s, stride_d,
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    m_blk = tl.program_id(1)
    rm = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, BLOCK_D)
    base = bh * stride_bh
    q_mask = (rm[:, None] < S) & (rd[None, :] < D)
    q = tl.load(Q_ptr + base + rm[:, None] * stride_s + rd[None, :] * stride_d, mask=q_mask, other=0.0).to(tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    max_n = tl.minimum((m_blk + 1) * BLOCK_M, S)
    for n0 in range(0, max_n, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        kv_mask = (rn[:, None] < S) & (rd[None, :] < D)
        k = tl.load(K_ptr + base + rn[:, None] * stride_s + rd[None, :] * stride_d, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + base + rn[:, None] * stride_s + rd[None, :] * stride_d, mask=kv_mask, other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        causal_mask = (rn[None, :] <= rm[:, None]) & (rn[None, :] < S) & (rm[:, None] < S)
        qk = tl.where(causal_mask, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        p = tl.where(causal_mask, tl.exp(qk - safe_m[:, None]), 0.0)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - safe_m))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_ij
    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(O_ptr + base + rm[:, None] * stride_s + rd[None, :] * stride_d, out.to(O_ptr.dtype.element_ty), mask=q_mask)


def kernel_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q.ndim != 4:
        raise ValueError(f"Causal SDPA expects 4D [B, H, S, D] tensors, got {tuple(q.shape)}")
    B, H, S, D = q.shape
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    o = torch.empty_like(q_c)
    q3 = q_c.reshape(B * H, S, D)
    k3 = k_c.reshape(B * H, S, D)
    v3 = v_c.reshape(B * H, S, D)
    o3 = o.reshape(B * H, S, D)
    scale = 1.0 / (D ** 0.5)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * H, triton.cdiv(S, 32))
    _flash_causal_attn_kernel[grid](
        q3, k3, v3, o3,
        q3.stride(0), q3.stride(1), q3.stride(2),
        S, D, scale, BLOCK_M=32, BLOCK_N=32, BLOCK_D=BLOCK_D,
    )
    return o
""".strip()

TRITON_BASELINES: Dict[str, str] = {
    "rmsnorm_2048": TRITON_RMSNORM_BASELINE,
    "rmsnorm_4096": TRITON_RMSNORM_BASELINE,
    "layernorm_2048": TRITON_LAYERNORM_BASELINE,
    "swiglu_4096": TRITON_SWIGLU_BASELINE,
    "fused_add_rmsnorm_2048": TRITON_FUSED_ADD_RMSNORM_BASELINE,
    "fused_softmax_2048": TRITON_SOFTMAX_BASELINE,
    "log_softmax_2048": TRITON_LOG_SOFTMAX_BASELINE,
    "rope_fwd_128": TRITON_ROPE_BASELINE,
    "matmul_bias_gelu_512": TRITON_MATMUL_BIAS_GELU_BASELINE,
    "int8_weight_only_gemm_512": TRITON_INT8_WEIGHT_ONLY_GEMM_BASELINE,
    "scaled_dot_product_attn_128": TRITON_FLASH_CAUSAL_ATTN_BASELINE,
}


def validate_triton_baselines_syntax() -> dict[str, bool]:
    """Parse every Triton baseline with AST and verify `@triton.jit`, `kernel_forward`,
    `.contiguous()`, and explicit `input_precision="ieee"` on every `tl.dot(...)` call.
    """
    results: dict[str, bool] = {}
    for op_name, code in TRITON_BASELINES.items():
        tree = ast.parse(code)
        has_entry = any(isinstance(n, ast.FunctionDef) and n.name == "kernel_forward" for n in tree.body)
        has_triton_jit = "@triton.jit" in code
        has_contiguous = ".contiguous()" in code
        dot_ok = True
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dot":
                kw = {k.arg: k.value for k in node.keywords}
                prec = kw.get("input_precision")
                if not (isinstance(prec, ast.Constant) and prec.value == "ieee"):
                    dot_ok = False
        results[op_name] = bool(has_entry and has_triton_jit and has_contiguous and dot_ok)
    return results
