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
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_bias_silu_kernel[grid](a_c, b_c, bias_c, c, M, N, K,
                                    a_c.stride(0), a_c.stride(1), b_c.stride(0), b_c.stride(1), c.stride(0), c.stride(1),
                                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return c