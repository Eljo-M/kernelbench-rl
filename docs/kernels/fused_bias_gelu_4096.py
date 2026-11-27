import torch
import triton
import triton.language as tl


@triton.jit
def _bias_gelu_kernel(X_ptr, B_ptr, Y_ptr, stride_x, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = x + b
    k_coef = 0.7978845608028654
    inner = k_coef * (z + 0.044715 * z * z * z)
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    out = 0.5 * z * (1.0 + t)
    tl.store(Y_ptr + row * stride_x + cols, out.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_forward(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, orig_shape[-1])
    b_c = bias.contiguous()
    y = torch.empty_like(x_2d)
    M, N = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 65536:
        raise ValueError(f"Bias size N={N} > 65536 for single-block GeLU")
    _bias_gelu_kernel[(M,)](x_2d, b_c, y, x_2d.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y.reshape(orig_shape)