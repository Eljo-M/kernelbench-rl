from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    level: int
    description: str
    make_inputs: Callable[[np.random.Generator], tuple[np.ndarray, ...]]
    reference_fn: Callable[..., np.ndarray]
    flops: int
    bytes_moved: int
    rtol: float = 1e-4
    atol: float = 1e-4
    make_stress_inputs: Optional[Callable[[np.random.Generator], list[tuple[np.ndarray, ...]]]] = None

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / max(self.bytes_moved, 1)


# ---------------------------------------------------------------------------
# Level 1 Reference Implementations (Elementwise, Normalization, Activations)
# ---------------------------------------------------------------------------
def _ref_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _ref_layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    ez = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez)).astype(z.dtype)


def _ref_swiglu(x: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return x * (gate * _stable_sigmoid(gate))


def _ref_geglu(x: np.ndarray, gate: np.ndarray) -> np.ndarray:
    gelu = 0.5 * gate * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate + 0.044715 * (gate ** 3))))
    return x * gelu


def _ref_fused_add_rmsnorm(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    h = x + residual
    rms = np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps)
    return (h / rms) * weight


def _ref_bias_gelu(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = x + bias
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z ** 3))))


def _ref_bias_silu(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = x + bias
    return z * _stable_sigmoid(z)


def _ref_dropout_residual(x: np.ndarray, residual: np.ndarray, mask: np.ndarray, p: float = 0.1) -> np.ndarray:
    scale = np.float32(1.0 / (1.0 - p))
    return residual + x * mask * scale


def _ref_scale_mask(x: np.ndarray, mask: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.where(mask > 0.5, x * scale, np.float32(-1e9))


def _ref_l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


def _ref_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    na = np.sqrt(np.sum(a * a, axis=-1) + eps)
    nb = np.sqrt(np.sum(b * b, axis=-1) + eps)
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Level 2 Reference Implementations (Reductions, Attention, RoPE, Fused GEMM)
# ---------------------------------------------------------------------------
def _ref_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(shifted)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def _ref_causal_masked_softmax(scores: np.ndarray) -> np.ndarray:
    seq = scores.shape[-1]
    causal = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    masked = np.where(causal, np.float32(-1e9), scores)
    return _ref_softmax(masked)


def _ref_log_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    shifted = x - m
    lse = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - lse


def _ref_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    log_probs = _ref_log_softmax(logits)
    idx = np.arange(logits.shape[0])
    return -log_probs[idx, targets.astype(np.int64)]


def _ref_rope_fwd(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # x: [B, H, S, D], cos/sin: [S, D//2]
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return np.concatenate([out1, out2], axis=-1)


def _ref_matmul_gelu(a: np.ndarray, b: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = a @ b + bias
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z ** 3))))


def _ref_matmul_silu(a: np.ndarray, b: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = a @ b + bias
    return z * _stable_sigmoid(z)


def _ref_matmul_residual(a: np.ndarray, b: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return (a @ b) + residual


def _ref_sdpa_causal(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = q.shape[-1]
    scale = np.float32(1.0 / np.sqrt(d))
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    probs = _ref_causal_masked_softmax(scores)
    return np.matmul(probs, v)


def _ref_group_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, num_groups: int = 8, eps: float = 1e-5) -> np.ndarray:
    n, c, l = x.shape
    g = num_groups
    xg = x.reshape(n, g, c // g, l)
    mean = np.mean(xg, axis=(2, 3), keepdims=True)
    var = np.var(xg, axis=(2, 3), keepdims=True)
    y = ((xg - mean) / np.sqrt(var + eps)).reshape(n, c, l)
    return y * weight[None, :, None] + bias[None, :, None]


# ---------------------------------------------------------------------------
# Level 3 Reference Implementations (Quantized, Sparse, KV-Cache, SSM, MoE)
# ---------------------------------------------------------------------------
def _ref_int8_weight_only_gemm(x: np.ndarray, qweight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    w_fp32 = qweight.astype(np.float32) * scales[None, :]
    return x @ w_fp32


def _ref_int4_groupwise_dequant_gemm(
    x: np.ndarray, packed_w: np.ndarray, scales: np.ndarray, group_size: int = 64
) -> np.ndarray:
    # packed_w: [K//2, N] uint8 storing two signed int4 values offset by +8
    lo = (packed_w & np.uint8(0x0F)).astype(np.int32) - 8
    hi = ((packed_w >> np.uint8(4)) & np.uint8(0x0F)).astype(np.int32) - 8
    k_half, n = packed_w.shape
    unpacked = np.empty((k_half * 2, n), dtype=np.float32)
    unpacked[0::2, :] = lo
    unpacked[1::2, :] = hi
    num_groups = (k_half * 2) // group_size
    dequant = (unpacked.reshape(num_groups, group_size, n) * scales[:, None, :]).reshape(k_half * 2, n)
    return x @ dequant


def _ref_fp8_scaled_gemm(
    a_q: np.ndarray, b_q: np.ndarray, scale_a: np.ndarray, scale_b: np.ndarray
) -> np.ndarray:
    return (a_q.astype(np.float32) * scale_a[:, None]) @ (b_q.astype(np.float32) * scale_b[None, :])


def _ref_block_sparse_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray, block_size: int = 32) -> np.ndarray:
    # Causal + local block-window attention (attends to current block and previous block)
    s = q.shape[-2]
    d = q.shape[-1]
    scale = np.float32(1.0 / np.sqrt(d))
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    q_idx = np.arange(s)[:, None]
    k_idx = np.arange(s)[None, :]
    q_blk = q_idx // block_size
    k_blk = k_idx // block_size
    allowed = (k_idx <= q_idx) & (k_blk >= q_blk - 1)
    masked = np.where(allowed, scores, np.float32(-1e9))
    probs = _ref_softmax(masked)
    return np.matmul(probs, v)


def _ref_gqa_decode_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q: [B, Hq, 1, D], k/v: [B, Hkv, S, D]
    b, hq, _, d = q.shape
    hkv = k.shape[1]
    repeats = hq // hkv
    k_exp = np.repeat(k, repeats, axis=1)
    v_exp = np.repeat(v, repeats, axis=1)
    scale = np.float32(1.0 / np.sqrt(d))
    scores = np.matmul(q, np.swapaxes(k_exp, -1, -2)) * scale
    probs = _ref_softmax(scores)
    return np.matmul(probs, v_exp)


def _ref_paged_kv_gather_attn(
    q: np.ndarray, k_pool: np.ndarray, v_pool: np.ndarray, block_tables: np.ndarray
) -> np.ndarray:
    # q: [B, H, D], k_pool/v_pool: [NumBlocks, H, BlockSize, D], block_tables: [B, MaxBlocks]
    b, h, d = q.shape
    scale = np.float32(1.0 / np.sqrt(d))
    outputs = np.empty((b, h, d), dtype=np.float32)
    for i in range(b):
        blks = block_tables[i]
        # Gather blocks -> [MaxBlocks, H, BlockSize, D] -> [H, S, D]
        k_seq = np.transpose(k_pool[blks], (1, 0, 2, 3)).reshape(h, -1, d)
        v_seq = np.transpose(v_pool[blks], (1, 0, 2, 3)).reshape(h, -1, d)
        scores = np.einsum("hd,hsd->hs", q[i], k_seq) * scale
        probs = _ref_softmax(scores)
        outputs[i] = np.einsum("hs,hsd->hd", probs, v_seq)
    return outputs


def _ref_selective_scan_ssm(u: np.ndarray, delta: np.ndarray, a_log: np.ndarray, b_proj: np.ndarray, c_proj: np.ndarray) -> np.ndarray:
    # Diagonal selective scan: u, delta, b_proj, c_proj: [B, L, D]; a_log: [D]
    bsz, seqlen, dim = u.shape
    a = -np.exp(a_log).astype(np.float32)  # negative decay for stability
    h = np.zeros((bsz, dim), dtype=np.float32)
    ys = np.empty((bsz, seqlen, dim), dtype=np.float32)
    for t in range(seqlen):
        dt = delta[:, t, :]
        da = np.exp(dt * a[None, :])
        db_u = dt * b_proj[:, t, :] * u[:, t, :]
        h = da * h + db_u
        ys[:, t, :] = h * c_proj[:, t, :]
    return ys


def _ref_topk_moe_router(x: np.ndarray, router_w: np.ndarray, expert_w: np.ndarray, k: int = 2) -> np.ndarray:
    # x: [T, D], router_w: [D, E], expert_w: [E, D, D]
    logits = x @ router_w
    probs = _ref_softmax(logits)
    topk_idx = np.argpartition(-probs, kth=k - 1, axis=-1)[:, :k]
    rows = np.arange(x.shape[0])[:, None]
    topk_probs = probs[rows, topk_idx]
    topk_probs = topk_probs / np.sum(topk_probs, axis=-1, keepdims=True)
    out = np.zeros_like(x)
    for slot in range(k):
        e_ids = topk_idx[:, slot]
        w_slot = topk_probs[:, slot][:, None]
        for e in range(expert_w.shape[0]):
            sel = e_ids == e
            if np.any(sel):
                out[sel] += (x[sel] @ expert_w[e]) * w_slot[sel]
    return out


# ---------------------------------------------------------------------------
# Helper input generators for structured Level 2/3 operators
# ---------------------------------------------------------------------------
def _make_rope_inputs(rng: np.random.Generator, b: int = 8, h: int = 16, s: int = 128, d: int = 64):
    x = rng.standard_normal((b, h, s, d), dtype=np.float32)
    pos = np.arange(s, dtype=np.float32)[:, None]
    inv_freq = (1.0 / (10000.0 ** (np.arange(0, d, 2, dtype=np.float32) / d)))[None, :]
    angles = pos * inv_freq
    return (x, np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32))


def _make_int4_inputs(rng: np.random.Generator, m: int = 128, k: int = 512, n: int = 512, g: int = 64):
    x = rng.standard_normal((m, k), dtype=np.float32)
    packed = rng.integers(0, 256, size=(k // 2, n), dtype=np.uint8)
    scales = (rng.uniform(0.01, 0.2, size=(k // g, n))).astype(np.float32)
    return (x, packed, scales)


def _make_paged_kv_inputs(rng: np.random.Generator, b: int = 4, h: int = 8, d: int = 64, num_blocks: int = 64, blk_sz: int = 16, max_blks: int = 16):
    q = rng.standard_normal((b, h, d), dtype=np.float32)
    k_pool = rng.standard_normal((num_blocks, h, blk_sz, d), dtype=np.float32)
    v_pool = rng.standard_normal((num_blocks, h, blk_sz, d), dtype=np.float32)
    block_tables = rng.integers(0, num_blocks, size=(b, max_blks), dtype=np.int32)
    return (q, k_pool, v_pool, block_tables)


def _make_ssm_inputs(rng: np.random.Generator, b: int = 8, l: int = 256, d: int = 64):
    u = rng.standard_normal((b, l, d), dtype=np.float32)
    delta = rng.uniform(0.01, 0.2, size=(b, l, d)).astype(np.float32)
    a_log = rng.uniform(-1.0, 0.5, size=(d,)).astype(np.float32)
    b_proj = rng.standard_normal((b, l, d), dtype=np.float32) * np.float32(0.25)
    c_proj = rng.standard_normal((b, l, d), dtype=np.float32) * np.float32(0.25)
    return (u, delta, a_log, b_proj, c_proj)


# ---------------------------------------------------------------------------
# 30-Operator Canonical Registry (Level 1: 12, Level 2: 10, Level 3: 8)
# ---------------------------------------------------------------------------
OP_REGISTRY: dict[str, OperatorSpec] = {
    # --- LEVEL 1 (12 ops) ---
    "rmsnorm_2048": OperatorSpec(
        name="rmsnorm_2048",
        level=1,
        description="Fused RMSNorm over [512, 2048] fp32 activations",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 2048), dtype=np.float32),
            rng.standard_normal((2048,), dtype=np.float32),
        ),
        make_stress_inputs=lambda rng: [
            (rng.standard_normal((1, 2048), dtype=np.float32), rng.standard_normal((2048,), dtype=np.float32)),
            (rng.standard_normal((17, 771), dtype=np.float32), rng.standard_normal((771,), dtype=np.float32)),
        ],
        reference_fn=_ref_rmsnorm,
        flops=512 * 2048 * 5,
        bytes_moved=512 * 2048 * 4 * 2 + 2048 * 4,
    ),
    "rmsnorm_4096": OperatorSpec(
        name="rmsnorm_4096",
        level=1,
        description="Fused RMSNorm over [256, 4096] Llama-7B hidden dimension",
        make_inputs=lambda rng: (
            rng.standard_normal((256, 4096), dtype=np.float32),
            rng.standard_normal((4096,), dtype=np.float32),
        ),
        make_stress_inputs=lambda rng: [
            (rng.standard_normal((3, 3073), dtype=np.float32), rng.standard_normal((3073,), dtype=np.float32)),
        ],
        reference_fn=_ref_rmsnorm,
        flops=256 * 4096 * 5,
        bytes_moved=256 * 4096 * 4 * 2 + 4096 * 4,
    ),
    "layernorm_2048": OperatorSpec(
        name="layernorm_2048",
        level=1,
        description="Fused affine LayerNorm over [512, 2048] fp32 activations",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 2048), dtype=np.float32),
            rng.standard_normal((2048,), dtype=np.float32),
            rng.standard_normal((2048,), dtype=np.float32),
        ),
        make_stress_inputs=lambda rng: [
            (
                rng.standard_normal((11, 515), dtype=np.float32),
                rng.standard_normal((515,), dtype=np.float32),
                rng.standard_normal((515,), dtype=np.float32),
            ),
        ],
        reference_fn=_ref_layernorm,
        flops=512 * 2048 * 7,
        bytes_moved=512 * 2048 * 4 * 2 + 2048 * 4 * 2,
    ),
    "swiglu_4096": OperatorSpec(
        name="swiglu_4096",
        level=1,
        description="Fused SwiGLU activation over [512, 4096] fp32 tensors",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 4096), dtype=np.float32),
            rng.standard_normal((512, 4096), dtype=np.float32),
        ),
        make_stress_inputs=lambda rng: [
            (rng.standard_normal((1, 129), dtype=np.float32), rng.standard_normal((1, 129), dtype=np.float32)),
            (rng.standard_normal((13, 1025), dtype=np.float32), rng.standard_normal((13, 1025), dtype=np.float32)),
        ],
        reference_fn=_ref_swiglu,
        flops=512 * 4096 * 5,
        bytes_moved=512 * 4096 * 4 * 3,
    ),
    "geglu_4096": OperatorSpec(
        name="geglu_4096",
        level=1,
        description="Fused GeGLU (tanh-approx GeLU gate) over [512, 4096]",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 4096), dtype=np.float32),
            rng.standard_normal((512, 4096), dtype=np.float32),
        ),
        reference_fn=_ref_geglu,
        flops=512 * 4096 * 9,
        bytes_moved=512 * 4096 * 4 * 3,
    ),
    "fused_add_rmsnorm_2048": OperatorSpec(
        name="fused_add_rmsnorm_2048",
        level=1,
        description="Fused residual addition + RMSNorm over [512, 2048]",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 2048), dtype=np.float32),
            rng.standard_normal((512, 2048), dtype=np.float32),
            rng.standard_normal((2048,), dtype=np.float32),
        ),
        reference_fn=_ref_fused_add_rmsnorm,
        flops=512 * 2048 * 6,
        bytes_moved=512 * 2048 * 4 * 3 + 2048 * 4,
    ),
    "fused_bias_gelu_4096": OperatorSpec(
        name="fused_bias_gelu_4096",
        level=1,
        description="Fused bias add + tanh-approx GeLU over [512, 4096]",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 4096), dtype=np.float32),
            rng.standard_normal((4096,), dtype=np.float32),
        ),
        reference_fn=_ref_bias_gelu,
        flops=512 * 4096 * 9,
        bytes_moved=512 * 4096 * 4 * 2 + 4096 * 4,
    ),
    "fused_bias_silu_4096": OperatorSpec(
        name="fused_bias_silu_4096",
        level=1,
        description="Fused bias add + SiLU activation over [512, 4096]",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 4096), dtype=np.float32),
            rng.standard_normal((4096,), dtype=np.float32),
        ),
        reference_fn=_ref_bias_silu,
        flops=512 * 4096 * 5,
        bytes_moved=512 * 4096 * 4 * 2 + 4096 * 4,
    ),
    "fused_dropout_residual_2048": OperatorSpec(
        name="fused_dropout_residual_2048",
        level=1,
        description="Fused pre-generated dropout mask scaling + residual add [512, 2048]",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 2048), dtype=np.float32),
            rng.standard_normal((512, 2048), dtype=np.float32),
            (rng.random((512, 2048)) > 0.1).astype(np.float32),
        ),
        reference_fn=_ref_dropout_residual,
        flops=512 * 2048 * 3,
        bytes_moved=512 * 2048 * 4 * 4,
    ),
    "fused_scale_mask_2048": OperatorSpec(
        name="fused_scale_mask_2048",
        level=1,
        description="Fused attention score scale + boolean mask fill [512, 2048]",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 2048), dtype=np.float32),
            (rng.random((512, 2048)) > 0.25).astype(np.float32),
            np.array(0.125, dtype=np.float32),
        ),
        reference_fn=_ref_scale_mask,
        flops=512 * 2048 * 2,
        bytes_moved=512 * 2048 * 4 * 3,
    ),
    "l2_normalize_2048": OperatorSpec(
        name="l2_normalize_2048",
        level=1,
        description="Row-wise L2 embedding normalization over [512, 2048]",
        make_inputs=lambda rng: (rng.standard_normal((512, 2048), dtype=np.float32),),
        reference_fn=_ref_l2_normalize,
        flops=512 * 2048 * 4,
        bytes_moved=512 * 2048 * 4 * 2,
    ),
    "cosine_similarity_1024": OperatorSpec(
        name="cosine_similarity_1024",
        level=1,
        description="Batched row-wise cosine similarity over [512, 1024]",
        make_inputs=lambda rng: (
            rng.standard_normal((512, 1024), dtype=np.float32),
            rng.standard_normal((512, 1024), dtype=np.float32),
        ),
        reference_fn=_ref_cosine_similarity,
        flops=512 * 1024 * 6,
        bytes_moved=512 * 1024 * 4 * 2 + 512 * 4,
    ),
    # --- LEVEL 2 (10 ops) ---
    "fused_softmax_2048": OperatorSpec(
        name="fused_softmax_2048",
        level=2,
        description="Numerically stable row-wise Softmax over [512, 2048]",
        make_inputs=lambda rng: (rng.standard_normal((512, 2048), dtype=np.float32),),
        make_stress_inputs=lambda rng: [
            (rng.standard_normal((1, 127), dtype=np.float32),),
            (rng.standard_normal((19, 1029), dtype=np.float32) * np.float32(50.0),),
        ],
        reference_fn=_ref_softmax,
        flops=512 * 2048 * 5,
        bytes_moved=512 * 2048 * 4 * 2,
    ),
    "causal_masked_softmax_1024": OperatorSpec(
        name="causal_masked_softmax_1024",
        level=2,
        description="Fused upper-triangular causal mask + Softmax over [32, 128, 128]",
        make_inputs=lambda rng: (rng.standard_normal((32, 128, 128), dtype=np.float32),),
        reference_fn=_ref_causal_masked_softmax,
        flops=32 * 128 * 128 * 6,
        bytes_moved=32 * 128 * 128 * 4 * 2,
    ),
    "log_softmax_2048": OperatorSpec(
        name="log_softmax_2048",
        level=2,
        description="Numerically stable row-wise LogSoftmax over [512, 2048]",
        make_inputs=lambda rng: (rng.standard_normal((512, 2048), dtype=np.float32),),
        make_stress_inputs=lambda rng: [
            (rng.standard_normal((7, 333), dtype=np.float32) * np.float32(40.0),),
        ],
        reference_fn=_ref_log_softmax,
        flops=512 * 2048 * 5,
        bytes_moved=512 * 2048 * 4 * 2,
    ),
    "fused_cross_entropy_4096": OperatorSpec(
        name="fused_cross_entropy_4096",
        level=2,
        description="Online fused LogSoftmax + NLL cross-entropy loss [256, 4096]",
        make_inputs=lambda rng: (
            rng.standard_normal((256, 4096), dtype=np.float32),
            rng.integers(0, 4096, size=(256,), dtype=np.int64),
        ),
        reference_fn=_ref_cross_entropy,
        flops=256 * 4096 * 5,
        bytes_moved=256 * 4096 * 4 + 256 * 8 + 256 * 4,
    ),
    "rope_fwd_128": OperatorSpec(
        name="rope_fwd_128",
        level=2,
        description="Fused Rotary Positional Embedding forward pass [8, 16, 128, 64]",
        make_inputs=lambda rng: _make_rope_inputs(rng, 8, 16, 128, 64),
        make_stress_inputs=lambda rng: [_make_rope_inputs(rng, 2, 4, 37, 64)],
        reference_fn=_ref_rope_fwd,
        flops=8 * 16 * 128 * 64 * 3,
        bytes_moved=8 * 16 * 128 * 64 * 4 * 2 + 128 * 32 * 4 * 2,
    ),
    "matmul_bias_gelu_512": OperatorSpec(
        name="matmul_bias_gelu_512",
        level=2,
        description="Fused GEMM + Bias + tanh-approx GeLU [256, 512] x [512, 512]",
        make_inputs=lambda rng: (
            rng.standard_normal((256, 512), dtype=np.float32),
            rng.standard_normal((512, 512), dtype=np.float32),
            rng.standard_normal((512,), dtype=np.float32),
        ),
        make_stress_inputs=lambda rng: [
            (
                rng.standard_normal((7, 65), dtype=np.float32),
                rng.standard_normal((65, 33), dtype=np.float32),
                rng.standard_normal((33,), dtype=np.float32),
            ),
        ],
        reference_fn=_ref_matmul_gelu,
        flops=2 * 256 * 512 * 512 + 256 * 512 * 9,
        bytes_moved=(256 * 512 + 512 * 512 + 256 * 512 + 512) * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "matmul_bias_silu_512": OperatorSpec(
        name="matmul_bias_silu_512",
        level=2,
        description="Fused GEMM + Bias + SiLU epilogue [256, 512] x [512, 512]",
        make_inputs=lambda rng: (
            rng.standard_normal((256, 512), dtype=np.float32),
            rng.standard_normal((512, 512), dtype=np.float32),
            rng.standard_normal((512,), dtype=np.float32),
        ),
        reference_fn=_ref_matmul_silu,
        flops=2 * 256 * 512 * 512 + 256 * 512 * 5,
        bytes_moved=(256 * 512 + 512 * 512 + 256 * 512 + 512) * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "matmul_residual_add_512": OperatorSpec(
        name="matmul_residual_add_512",
        level=2,
        description="Fused GEMM + residual skip-connection epilogue [256, 512] x [512, 512]",
        make_inputs=lambda rng: (
            rng.standard_normal((256, 512), dtype=np.float32),
            rng.standard_normal((512, 512), dtype=np.float32),
            rng.standard_normal((256, 512), dtype=np.float32),
        ),
        reference_fn=_ref_matmul_residual,
        flops=2 * 256 * 512 * 512 + 256 * 512,
        bytes_moved=(256 * 512 * 3 + 512 * 512) * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "scaled_dot_product_attn_128": OperatorSpec(
        name="scaled_dot_product_attn_128",
        level=2,
        description="Causal Scaled Dot-Product Attention [4, 8, 128, 64]",
        make_inputs=lambda rng: (
            rng.standard_normal((4, 8, 128, 64), dtype=np.float32),
            rng.standard_normal((4, 8, 128, 64), dtype=np.float32),
            rng.standard_normal((4, 8, 128, 64), dtype=np.float32),
        ),
        reference_fn=_ref_sdpa_causal,
        flops=4 * 8 * (4 * 128 * 128 * 64 + 5 * 128 * 128),
        bytes_moved=4 * 8 * 128 * 64 * 4 * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "group_norm_64x32": OperatorSpec(
        name="group_norm_64x32",
        level=2,
        description="Fused GroupNorm (8 groups) over [16, 32, 64]",
        make_inputs=lambda rng: (
            rng.standard_normal((16, 32, 64), dtype=np.float32),
            rng.standard_normal((32,), dtype=np.float32),
            rng.standard_normal((32,), dtype=np.float32),
        ),
        reference_fn=_ref_group_norm,
        flops=16 * 32 * 64 * 7,
        bytes_moved=16 * 32 * 64 * 4 * 2 + 32 * 4 * 2,
    ),
    # --- LEVEL 3 (8 ops) ---
    "int8_weight_only_gemm_512": OperatorSpec(
        name="int8_weight_only_gemm_512",
        level=3,
        description="Per-channel symmetric INT8 weight-only dequant + GEMM [128, 512] x [512, 512]",
        make_inputs=lambda rng: (
            rng.standard_normal((128, 512), dtype=np.float32),
            rng.integers(-127, 128, size=(512, 512), dtype=np.int8),
            rng.uniform(0.005, 0.05, size=(512,)).astype(np.float32),
        ),
        reference_fn=_ref_int8_weight_only_gemm,
        flops=2 * 128 * 512 * 512 + 512 * 512,
        bytes_moved=128 * 512 * 4 * 2 + 512 * 512 * 1 + 512 * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "int4_groupwise_dequant_gemm_512": OperatorSpec(
        name="int4_groupwise_dequant_gemm_512",
        level=3,
        description="Packed INT4 groupwise (G=64) dequant + GEMM [128, 512] x [512, 512]",
        make_inputs=lambda rng: _make_int4_inputs(rng, 128, 512, 512, 64),
        reference_fn=_ref_int4_groupwise_dequant_gemm,
        flops=2 * 128 * 512 * 512 + 512 * 512 * 3,
        bytes_moved=128 * 512 * 4 * 2 + (512 * 512) // 2 + (512 // 64) * 512 * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "fp8_e4m3_simulated_scaled_gemm_512": OperatorSpec(
        name="fp8_e4m3_simulated_scaled_gemm_512",
        level=3,
        description="Row/col-scaled FP8-range quantized GEMM [128, 512] x [512, 512]",
        make_inputs=lambda rng: (
            rng.integers(-120, 121, size=(128, 512), dtype=np.int8),
            rng.integers(-120, 121, size=(512, 512), dtype=np.int8),
            rng.uniform(0.01, 0.05, size=(128,)).astype(np.float32),
            rng.uniform(0.01, 0.05, size=(512,)).astype(np.float32),
        ),
        reference_fn=_ref_fp8_scaled_gemm,
        flops=2 * 128 * 512 * 512 + 128 * 512 + 512 * 512,
        bytes_moved=128 * 512 * 1 + 512 * 512 * 1 + 128 * 512 * 4 + (128 + 512) * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "block_sparse_attn_256": OperatorSpec(
        name="block_sparse_attn_256",
        level=3,
        description="Causal local block-sparse attention (block=32) [2, 4, 256, 64]",
        make_inputs=lambda rng: (
            rng.standard_normal((2, 4, 256, 64), dtype=np.float32),
            rng.standard_normal((2, 4, 256, 64), dtype=np.float32),
            rng.standard_normal((2, 4, 256, 64), dtype=np.float32),
        ),
        reference_fn=_ref_block_sparse_attn,
        flops=2 * 4 * (4 * 256 * 64 * 64),  # active 2-block window
        bytes_moved=2 * 4 * 256 * 64 * 4 * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "gqa_decode_attn_kv512": OperatorSpec(
        name="gqa_decode_attn_kv512",
        level=3,
        description="Grouped-Query Attention decode (16 Q heads, 4 KV heads, S=512)",
        make_inputs=lambda rng: (
            rng.standard_normal((4, 16, 1, 64), dtype=np.float32),
            rng.standard_normal((4, 4, 512, 64), dtype=np.float32),
            rng.standard_normal((4, 4, 512, 64), dtype=np.float32),
        ),
        reference_fn=_ref_gqa_decode_attn,
        flops=4 * 16 * (4 * 512 * 64 + 5 * 512),
        bytes_moved=(4 * 16 * 64 * 2 + 4 * 4 * 512 * 64 * 2) * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "paged_kv_gather_attn_256": OperatorSpec(
        name="paged_kv_gather_attn_256",
        level=3,
        description="PagedAttention block-table KV gather + attention (B=4, H=8, S=256)",
        make_inputs=lambda rng: _make_paged_kv_inputs(rng, 4, 8, 64, 64, 16, 16),
        reference_fn=_ref_paged_kv_gather_attn,
        flops=4 * 8 * (4 * 256 * 64 + 5 * 256),
        bytes_moved=(4 * 8 * 64 * 2 + 4 * 8 * 256 * 64 * 2) * 4 + 4 * 16 * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "selective_scan_ssm_256": OperatorSpec(
        name="selective_scan_ssm_256",
        level=3,
        description="Mamba-style diagonal selective state-space scan [8, 256, 64]",
        make_inputs=lambda rng: _make_ssm_inputs(rng, 8, 256, 64),
        reference_fn=_ref_selective_scan_ssm,
        flops=8 * 256 * 64 * 8,
        bytes_moved=8 * 256 * 64 * 4 * 5 + 64 * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
    "topk_moe_router_dispatch_64": OperatorSpec(
        name="topk_moe_router_dispatch_64",
        level=3,
        description="MoE Top-2 softmax gating + weighted expert dispatch [256, 64] x 8 experts",
        make_inputs=lambda rng: (
            rng.standard_normal((256, 64), dtype=np.float32),
            rng.standard_normal((64, 8), dtype=np.float32),
            rng.standard_normal((8, 64, 64), dtype=np.float32) * np.float32(0.1),
        ),
        reference_fn=_ref_topk_moe_router,
        flops=2 * 256 * 64 * 8 + 2 * 2 * 256 * 64 * 64,
        bytes_moved=(256 * 64 * 2 + 64 * 8 + 8 * 64 * 64) * 4,
        rtol=1e-3,
        atol=1e-3,
    ),
}


from kernels.extended_stress import make_extended_stress_cases
from kernels.triton_extended import TRITON_EXTENDED
for _name in TRITON_EXTENDED:
    OP_REGISTRY[_name] = replace(OP_REGISTRY[_name],
        make_stress_inputs=lambda rng, name=_name: make_extended_stress_cases(name, rng))


def get_operators_by_level(level: int) -> list[OperatorSpec]:
    return [spec for spec in OP_REGISTRY.values() if spec.level == level]


def verify_registry(seed: int = 42) -> bool:
    """Validate determinism, finiteness, positive FLOP/byte models, and stress shapes for all ops."""
    ok = True
    for name, spec in OP_REGISTRY.items():
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        in1 = spec.make_inputs(rng1)
        in2 = spec.make_inputs(rng2)
        out1 = spec.reference_fn(*in1)
        out2 = spec.reference_fn(*in2)
        if not np.array_equal(out1, out2):
            print(f"[FAIL] {name}: non-deterministic reference output")
            ok = False
            continue
        if not np.all(np.isfinite(out1)):
            print(f"[FAIL] {name}: non-finite values in reference output")
            ok = False
            continue
        if spec.flops <= 0 or spec.bytes_moved <= 0:
            print(f"[FAIL] {name}: invalid flops/bytes_moved")
            ok = False
            continue
        if spec.make_stress_inputs is not None:
            for idx, s_in in enumerate(spec.make_stress_inputs(np.random.default_rng(seed + 1))):
                s_out = spec.reference_fn(*s_in)
                if not np.all(np.isfinite(s_out)):
                    print(f"[FAIL] {name}: non-finite stress case {idx}")
                    ok = False
                    break
        print(
            f"[OK] L{spec.level} {name:<34} shape={str(tuple(out1.shape)):<18} "
            f"AI={spec.arithmetic_intensity:6.2f} FLOP/B"
        )
    print(f"\nVerified {len(OP_REGISTRY)} operators (L1={len(get_operators_by_level(1))}, "
          f"L2={len(get_operators_by_level(2))}, L3={len(get_operators_by_level(3))}).")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelBench-RL Operator Registry CLI")
    parser.add_argument("--verify", action="store_true", help="Run determinism & finiteness verification on all ops")
    parser.add_argument("--list", action="store_true", help="List registered operators by level")
    args = parser.parse_args()
    if args.list:
        for lvl in (1, 2, 3):
            print(f"Level {lvl}: {[op.name for op in get_operators_by_level(lvl)]}")
    if args.verify:
        if not verify_registry():
            sys.exit(1)


if __name__ == "__main__":
    main()
