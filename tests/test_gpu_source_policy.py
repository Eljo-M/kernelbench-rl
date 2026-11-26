import unittest

from kernels.triton_baselines import TRITON_BASELINES
from kernels.triton_extended import TRITON_EXTENDED
from sandbox_runner.gpu_source_policy import SECURITY_BOUNDARY_NOTE, check_gpu_source_policy


class TestGpuSourcePolicy(unittest.TestCase):
    def test_allows_pure_jit_math_helpers_with_a_real_launched_kernel(self):
        source = TRITON_BASELINES["rmsnorm_2048"]
        source += "\n@triton.jit\ndef helper(value):\n    return tl.sigmoid(value)\n"
        self.assertTrue(check_gpu_source_policy(source).allowed)

    def test_rejects_chained_and_annotated_torch_alias_fallbacks(self):
        source = TRITON_BASELINES["rmsnorm_2048"]
        for extra in ("    value = x.contiguous().reshape(x.shape)\n    return value * weight\n",
                      "    t: object = torch\n    return t.mean(x, dim=-1)\n",
                      "    reader = open\n    return reader('private.txt').read()\n"):
            self.assertFalse(check_gpu_source_policy(source + "\ndef fallback(x, weight):\n" + extra).allowed)

    def test_all_30_real_triton_baselines_pass_strict_policy(self):
        all_30 = {**TRITON_BASELINES, **TRITON_EXTENDED}
        self.assertEqual(len(all_30), 30)
        for op_name, code in all_30.items():
            res = check_gpu_source_policy(code)
            self.assertTrue(
                res.allowed,
                msg=f"Legitimate Triton baseline '{op_name}' was rejected: {res.violations}",
            )
            self.assertGreaterEqual(len(res.jit_kernels), 1, msg=f"{op_name} missing jit kernel")
            self.assertGreaterEqual(len(res.launched_kernels), 1, msg=f"{op_name} missing launched kernel")
            self.assertEqual(res.security_boundary, SECURITY_BOUNDARY_NOTE)

    def test_rejects_plain_torch_matmul_mean_rsqrt_fallback(self):
        code = """
import torch
def kernel_forward(x, weight, eps=1e-6):
    rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * rms * weight
"""
        res = check_gpu_source_policy(code)
        self.assertFalse(res.allowed)
        self.assertTrue(any("rsqrt" in v or "mean" in v for v in res.violations))

    def test_rejects_torch_alias_and_from_import_fallback(self):
        alias_code = """
import torch as t
import triton
import triton.language as tl

@triton.jit
def _k(X, Y, N, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(Y + o, tl.load(X + o, mask=o < N), mask=o < N)

def kernel_forward(a, b):
    mm = t.matmul
    return mm(a, b)
"""
        res = check_gpu_source_policy(alias_code)
        self.assertFalse(res.allowed)
        self.assertTrue(any("matmul" in v for v in res.violations))

        from_code = """
from torch import matmul as mm
def kernel_forward(a, b):
    return mm(a, b)
"""
        res2 = check_gpu_source_policy(from_code)
        self.assertFalse(res2.allowed)
        self.assertTrue(any("matmul" in v for v in res2.violations))

    def test_rejects_dummy_noop_triton_launch_plus_torch_fallback(self):
        code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _noop(X):
    pass

def kernel_forward(a, b):
    _noop[(1,)](a)
    return a @ b
"""
        res = check_gpu_source_policy(code)
        self.assertFalse(res.allowed)
        self.assertTrue(any("dummy/no-op" in v for v in res.violations))
        self.assertTrue(any("matrix multiplication" in v for v in res.violations))

    def test_rejects_nested_helper_doing_tensor_fallback(self):
        code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _copy(X, Y, N, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(Y + o, tl.load(X + o, mask=o < N), mask=o < N)

def kernel_forward(x, gate):
    def _sneaky(a, b):
        return a.sum(dim=-1)
    y = torch.empty_like(x)
    _copy[(1,)](x, y, x.numel(), B=64)
    return _sneaky(y, gate)
"""
        res = check_gpu_source_policy(code)
        self.assertFalse(res.allowed)
        self.assertTrue(any("sum" in v for v in res.violations))

    def test_rejects_output_memoization_dunder_escape_and_forbidden_imports(self):
        memo_code = """
import torch
import triton
import triton.language as tl

CACHE = {}

@triton.jit
def _k(X, Y, N, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(Y + o, tl.load(X + o, mask=o < N), mask=o < N)

def kernel_forward(x):
    y = torch.empty_like(x)
    _k[(1,)](x, y, x.numel(), B=64)
    CACHE["last"] = y
    return y
"""
        self.assertFalse(check_gpu_source_policy(memo_code).allowed)

        dunder_code = """
import torch
def kernel_forward(x):
    return x.__class__
"""
        self.assertFalse(check_gpu_source_policy(dunder_code).allowed)

        imp_code = """
import os
import torch
def kernel_forward(x):
    return x
"""
        self.assertFalse(check_gpu_source_policy(imp_code).allowed)

    def test_rejects_unlaunched_valid_jit_kernel(self):
        code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _unused(X, Y, N, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(Y + o, tl.load(X + o, mask=o < N), mask=o < N)

def kernel_forward(x):
    return x.contiguous()
"""
        res = check_gpu_source_policy(code)
        self.assertFalse(res.allowed)
        self.assertTrue(any("does not launch" in v for v in res.violations))

    def test_allows_legitimate_host_helper_launching_triton_kernel(self):
        code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _scale_k(X, Y, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    m = o < n
    tl.store(Y + o, tl.load(X + o, mask=m) * 2.0, mask=m)

def _launch(x_c, y):
    n = x_c.numel()
    _scale_k[(triton.cdiv(n, 256),)](x_c, y, n, B=256)

def kernel_forward(x):
    x_c = x.contiguous()
    y = torch.empty_like(x_c)
    _launch(x_c, y)
    return y
"""
        res = check_gpu_source_policy(code)
        self.assertTrue(res.allowed, msg=f"Unexpected violations: {res.violations}")
        self.assertEqual(res.launched_kernels, ["_scale_k"])


if __name__ == "__main__":
    unittest.main()
