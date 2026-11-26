"""Small non-power-of-two cases for the extended operator contracts."""
import numpy as np


def make_extended_stress_cases(name, rng):
    from kernels.op_registry import _make_paged_kv_inputs, _make_ssm_inputs
    f = lambda shape: rng.standard_normal(shape, dtype=np.float32)
    if name == "geglu_4096":
        args = (f((3, 19)), f((3, 19)))
    elif name in {"fused_bias_gelu_4096", "fused_bias_silu_4096"}:
        args = (f((3, 19)), f((19,)))
    elif name == "fused_dropout_residual_2048":
        args = (f((3, 19)), f((3, 19)), rng.integers(0, 2, (3, 19)).astype(np.float32))
    elif name == "fused_scale_mask_2048":
        args = (f((3, 19)), rng.integers(0, 2, (3, 19)).astype(np.float32), np.array([.125], np.float32))
    elif name == "l2_normalize_2048":
        return [(f((3, 19)),), (np.zeros((2, 7), np.float32),)]
    elif name == "cosine_similarity_1024":
        args = (f((3, 19)), f((3, 19)))
    elif name == "causal_masked_softmax_1024":
        args = (f((2, 3, 17, 17)),)
    elif name == "fused_cross_entropy_4096":
        args = (f((11, 35)), rng.integers(0, 35, 11, dtype=np.int64))
    elif name == "matmul_bias_silu_512":
        args = (f((17, 23)), f((23, 19)), f((19,)))
    elif name == "matmul_residual_add_512":
        args = (f((17, 23)), f((23, 19)), f((17, 19)))
    elif name == "group_norm_64x32":
        args = (f((2, 16, 7)), f((16,)), f((16,)))
    elif name == "int4_groupwise_dequant_gemm_512":
        args = (f((17, 128)), rng.integers(0, 256, (64, 29), dtype=np.uint8),
                rng.uniform(.005, .02, (2, 29)).astype(np.float32))
    elif name == "fp8_e4m3_simulated_scaled_gemm_512":
        args = (rng.integers(-16, 16, (17, 23), dtype=np.int8),
                rng.integers(-16, 16, (23, 19), dtype=np.int8),
                rng.uniform(.001, .02, 17).astype(np.float32),
                rng.uniform(.001, .02, 19).astype(np.float32))
    elif name == "block_sparse_attn_256":
        args = (f((1, 2, 37, 32)), f((1, 2, 37, 32)), f((1, 2, 37, 32)))
    elif name == "gqa_decode_attn_kv512":
        args = (f((1, 4, 1, 32)), f((1, 2, 43, 32)), f((1, 2, 43, 32)))
    elif name == "paged_kv_gather_attn_256":
        args = _make_paged_kv_inputs(rng, 1, 2, 32, 5, 8, 3)
    elif name == "selective_scan_ssm_256":
        args = _make_ssm_inputs(rng, 2, 17, 13)
    elif name == "topk_moe_router_dispatch_64":
        args = (f((17, 13)), f((13, 5)), f((5, 13, 13)) * np.float32(.1))
    else:
        raise KeyError(name)
    return [args]
