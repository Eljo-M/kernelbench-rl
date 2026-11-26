"""Native lazy-import PyTorch references for the 19 extended OP_REGISTRY operators.

Covers every operator in `kernels.op_registry.OP_REGISTRY` not already present
in `kernels.torch_references.TORCH_REFERENCES`. All PyTorch imports are lazy
so this module imports cleanly on CPU-only environments without PyTorch.
Integer quantization/index inputs (`int8`, `uint8`, `int32`, `int64`) are
strictly preserved as integer tensors.
"""

from __future__ import annotations

import math
from typing import Any, Callable


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for native torch references but is not installed."
        ) from exc
    return torch


def _tanh_via_sigmoid_torch(u):
    torch = _require_torch()
    return 2.0 * torch.sigmoid(2.0 * u) - 1.0


def _stable_softmax_torch(x):
    torch = _require_torch()
    xf = x.float()
    shifted = xf - xf.max(dim=-1, keepdim=True).values
    ex = torch.exp(shifted)
    return ex / ex.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Level 1 (7 extended ops)
# ---------------------------------------------------------------------------
def _torch_geglu(x, gate):
    g = gate.float()
    k = math.sqrt(2.0 / math.pi)
    gelu = 0.5 * g * (1.0 + _tanh_via_sigmoid_torch(k * (g + 0.044715 * g * g * g)))
    return (x.float() * gelu).to(x.dtype)


def _torch_bias_gelu(x, bias):
    z = x.float() + bias.float()
    k = math.sqrt(2.0 / math.pi)
    y = 0.5 * z * (1.0 + _tanh_via_sigmoid_torch(k * (z + 0.044715 * z * z * z)))
    return y.to(x.dtype)


def _torch_bias_silu(x, bias):
    torch = _require_torch()
    z = x.float() + bias.float()
    return (z * torch.sigmoid(z)).to(x.dtype)


def _torch_dropout_residual(x, residual, mask, p: float = 0.1):
    scale = 1.0 / (1.0 - p)
    return (residual.float() + x.float() * mask.float() * scale).to(x.dtype)


def _torch_scale_mask(x, mask, scale):
    torch = _require_torch()
    xf = x.float()
    mf = mask.float()
    sf = scale.float() if hasattr(scale, "float") else float(scale)
    neg = torch.tensor(-1e9, device=x.device, dtype=torch.float32)
    return torch.where(mf > 0.5, xf * sf, neg).to(x.dtype)


def _torch_l2_normalize(x, eps: float = 1e-12):
    torch = _require_torch()
    xf = x.float()
    inv = torch.rsqrt(torch.sum(xf * xf, dim=-1, keepdim=True) + eps)
    return (xf * inv).to(x.dtype)


def _torch_cosine_similarity(a, b, eps: float = 1e-8):
    torch = _require_torch()
    af, bf = a.float(), b.float()
    dot = torch.sum(af * bf, dim=-1)
    na = torch.sqrt(torch.sum(af * af, dim=-1) + eps)
    nb = torch.sqrt(torch.sum(bf * bf, dim=-1) + eps)
    return (dot / (na * nb)).to(a.dtype)


# ---------------------------------------------------------------------------
# Level 2 (5 extended ops)
# ---------------------------------------------------------------------------
def _torch_causal_masked_softmax(scores):
    torch = _require_torch()
    s = scores.shape[-1]
    causal = torch.triu(torch.ones((s, s), device=scores.device, dtype=torch.bool), diagonal=1)
    masked = torch.where(causal, torch.tensor(-1e9, device=scores.device, dtype=torch.float32), scores.float())
    return _stable_softmax_torch(masked).to(scores.dtype)


def _torch_cross_entropy(logits, targets):
    torch = _require_torch()
    if targets.is_floating_point():
        raise TypeError(f"targets must remain an integer tensor, got {targets.dtype}")
    xf = logits.float()
    m = xf.max(dim=-1, keepdim=True).values
    shifted = xf - m
    lse = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_probs = shifted - lse
    idx = torch.arange(logits.shape[0], device=logits.device)
    return (-log_probs[idx, targets.long()]).to(logits.dtype)


def _torch_matmul_bias_silu(a, b, bias):
    torch = _require_torch()
    z = torch.matmul(a.float(), b.float()) + bias.float()
    return (z * torch.sigmoid(z)).to(a.dtype)


def _torch_matmul_residual(a, b, residual):
    torch = _require_torch()
    return (torch.matmul(a.float(), b.float()) + residual.float()).to(a.dtype)


def _torch_group_norm(x, weight, bias, num_groups: int = 8, eps: float = 1e-5):
    torch = _require_torch()
    n, c, l = x.shape
    xg = x.float().reshape(n, num_groups, c // num_groups, l)
    mean = xg.mean(dim=(2, 3), keepdim=True)
    var = xg.var(dim=(2, 3), keepdim=True, unbiased=False)
    y = ((xg - mean) * torch.rsqrt(var + eps)).reshape(n, c, l)
    return (y * weight.float()[None, :, None] + bias.float()[None, :, None]).to(x.dtype)


# ---------------------------------------------------------------------------
# Level 3 (7 extended ops)
# ---------------------------------------------------------------------------
def _torch_int4_groupwise_dequant_gemm(x, packed_w, scales, group_size: int = 64):
    torch = _require_torch()
    if packed_w.is_floating_point():
        raise TypeError(f"packed_w must remain an integer tensor (uint8), got {packed_w.dtype}")
    pw = packed_w.to(torch.int32)
    lo = (pw & 0x0F) - 8
    hi = ((pw >> 4) & 0x0F) - 8
    k_half, n = packed_w.shape
    unpacked = torch.empty((k_half * 2, n), device=x.device, dtype=torch.float32)
    unpacked[0::2, :] = lo.float()
    unpacked[1::2, :] = hi.float()
    num_groups = (k_half * 2) // group_size
    dequant = (unpacked.reshape(num_groups, group_size, n) * scales.float()[:, None, :]).reshape(k_half * 2, n)
    return torch.matmul(x.float(), dequant).to(x.dtype)


def _torch_fp8_scaled_gemm(a_q, b_q, scale_a, scale_b):
    torch = _require_torch()
    if a_q.is_floating_point() or b_q.is_floating_point():
        raise TypeError(f"a_q and b_q must remain integer tensors, got {a_q.dtype}, {b_q.dtype}")
    a_f = a_q.to(torch.float32) * scale_a.float()[:, None]
    b_f = b_q.to(torch.float32) * scale_b.float()[None, :]
    return torch.matmul(a_f, b_f).to(scale_a.dtype)


def _torch_block_sparse_attn(q, k, v, block_size: int = 32):
    torch = _require_torch()
    qf, kf, vf = q.float(), k.float(), v.float()
    s = qf.shape[-2]
    d = qf.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    q_idx = torch.arange(s, device=q.device)[:, None]
    k_idx = torch.arange(s, device=q.device)[None, :]
    allowed = (k_idx <= q_idx) & ((k_idx // block_size) >= (q_idx // block_size) - 1)
    masked = torch.where(allowed, scores, torch.tensor(-1e9, device=q.device, dtype=torch.float32))
    probs = _stable_softmax_torch(masked)
    return torch.matmul(probs, vf).to(q.dtype)


def _torch_gqa_decode_attn(q, k, v):
    torch = _require_torch()
    qf, kf, vf = q.float(), k.float(), v.float()
    hq = qf.shape[1]
    hkv = kf.shape[1]
    d = qf.shape[-1]
    repeats = hq // hkv
    k_exp = kf.repeat_interleave(repeats, dim=1)
    v_exp = vf.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(qf, k_exp.transpose(-1, -2)) * (1.0 / math.sqrt(d))
    probs = _stable_softmax_torch(scores)
    return torch.matmul(probs, v_exp).to(q.dtype)


def _torch_paged_kv_gather_attn(q, k_pool, v_pool, block_tables):
    torch = _require_torch()
    if block_tables.is_floating_point():
        raise TypeError(f"block_tables must remain an integer tensor, got {block_tables.dtype}")
    qf, kf, vf = q.float(), k_pool.float(), v_pool.float()
    b, h, d = qf.shape
    scale = 1.0 / math.sqrt(d)
    outputs = torch.empty((b, h, d), device=q.device, dtype=torch.float32)
    for i in range(b):
        blks = block_tables[i].long()
        k_seq = kf[blks].permute(1, 0, 2, 3).reshape(h, -1, d)
        v_seq = vf[blks].permute(1, 0, 2, 3).reshape(h, -1, d)
        scores = torch.einsum("hd,hsd->hs", qf[i], k_seq) * scale
        probs = _stable_softmax_torch(scores)
        outputs[i] = torch.einsum("hs,hsd->hd", probs, v_seq)
    return outputs.to(q.dtype)


def _torch_selective_scan_ssm(u, delta, a_log, b_proj, c_proj):
    torch = _require_torch()
    uf, dtf, bf, cf = u.float(), delta.float(), b_proj.float(), c_proj.float()
    bsz, seqlen, dim = uf.shape
    a = -torch.exp(a_log.float())
    h = torch.zeros((bsz, dim), device=u.device, dtype=torch.float32)
    ys = torch.empty((bsz, seqlen, dim), device=u.device, dtype=torch.float32)
    for t in range(seqlen):
        dt = dtf[:, t, :]
        h = torch.exp(dt * a[None, :]) * h + dt * bf[:, t, :] * uf[:, t, :]
        ys[:, t, :] = h * cf[:, t, :]
    return ys.to(u.dtype)


def _torch_topk_moe_router(x, router_w, expert_w, k: int = 2):
    torch = _require_torch()
    xf, rw, ew = x.float(), router_w.float(), expert_w.float()
    probs = _stable_softmax_torch(torch.matmul(xf, rw))
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    topk_probs = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    # Batched gather/matmul avoids data-dependent host synchronization and
    # supports CUDA graph capture for the same timing path as the candidate.
    selected_weights = ew[topk_idx]
    expert_outputs = torch.matmul(xf[:, None, None, :], selected_weights).squeeze(-2)
    out = (expert_outputs * topk_probs.unsqueeze(-1)).sum(dim=1)
    return out.to(x.dtype)


TORCH_EXTENDED: dict[str, Callable[..., Any]] = {
    "geglu_4096": _torch_geglu,
    "fused_bias_gelu_4096": _torch_bias_gelu,
    "fused_bias_silu_4096": _torch_bias_silu,
    "fused_dropout_residual_2048": _torch_dropout_residual,
    "fused_scale_mask_2048": _torch_scale_mask,
    "l2_normalize_2048": _torch_l2_normalize,
    "cosine_similarity_1024": _torch_cosine_similarity,
    "causal_masked_softmax_1024": _torch_causal_masked_softmax,
    "fused_cross_entropy_4096": _torch_cross_entropy,
    "matmul_bias_silu_512": _torch_matmul_bias_silu,
    "matmul_residual_add_512": _torch_matmul_residual,
    "group_norm_64x32": _torch_group_norm,
    "int4_groupwise_dequant_gemm_512": _torch_int4_groupwise_dequant_gemm,
    "fp8_e4m3_simulated_scaled_gemm_512": _torch_fp8_scaled_gemm,
    "block_sparse_attn_256": _torch_block_sparse_attn,
    "gqa_decode_attn_kv512": _torch_gqa_decode_attn,
    "paged_kv_gather_attn_256": _torch_paged_kv_gather_attn,
    "selective_scan_ssm_256": _torch_selective_scan_ssm,
    "topk_moe_router_dispatch_64": _torch_topk_moe_router,
}
