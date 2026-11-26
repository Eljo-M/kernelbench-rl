import unittest

import numpy as np

from kernels.op_registry import OP_REGISTRY
from kernels.torch_extended import TORCH_EXTENDED
from kernels.torch_references import (
    TORCH_REFERENCES,
    convert_numpy_inputs_to_torch,
    resolve_input_dtype_plan,
    torch_inputs_to_numpy_fp32,
)
from kernels.triton_baselines import TRITON_BASELINES, CORE_TRITON_BASELINES
from kernels.triton_extended import TRITON_EXTENDED, validate_triton_extended_syntax
from sandbox_runner.anti_cheat import analyze_candidate_source

try:
    import torch  # type: ignore

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestTrack2Extended19Ops(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "PyTorch required")
    def test_signed_scan_prefix_form_handles_zero_and_mixed_inputs(self):
        rng = np.random.default_rng(9)
        u = rng.standard_normal((2, 33, 7), dtype=np.float32)
        dt = rng.uniform(0, 0.1, u.shape).astype(np.float32)
        alog = rng.standard_normal(7, dtype=np.float32)
        b = rng.standard_normal(u.shape, dtype=np.float32)
        c = rng.standard_normal(u.shape, dtype=np.float32)
        spec = OP_REGISTRY["selective_scan_ssm_256"]
        for data in ((u, dt, alog, b, c), (u*0, dt*0, alog, b, c)):
            expected = spec.reference_fn(*data)
            actual = TORCH_EXTENDED[spec.name](*(torch.from_numpy(a) for a in data)).numpy()
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertTrue(np.allclose(actual, expected, rtol=spec.rtol, atol=spec.atol))

    def test_exact_19_remaining_ops_covered_with_zero_overlap(self):
        existing = set(CORE_TRITON_BASELINES.keys())
        ext_triton = set(TRITON_EXTENDED.keys())
        ext_torch = set(TORCH_EXTENDED.keys())
        all_reg = set(OP_REGISTRY.keys())

        self.assertEqual(len(ext_triton), 19)
        self.assertEqual(ext_triton, ext_torch)
        self.assertEqual(existing & ext_triton, set())
        self.assertEqual(existing | ext_triton, all_reg)
        self.assertEqual(set(TORCH_REFERENCES.keys()) | ext_torch, all_reg)

    def test_all_19_triton_baselines_pass_ast_ieee_no_tanh_and_anti_cheat(self):
        syntax_map = validate_triton_extended_syntax()
        self.assertEqual(len(syntax_map), 19)
        for op_name, ok in syntax_map.items():
            self.assertTrue(ok, msg=f"Extended Triton baseline '{op_name}' failed static checks")
            code = TRITON_EXTENDED[op_name]
            self.assertNotIn("tl.math.tanh", code)
            self.assertNotIn("torch.nn.functional", code)
            cheat = analyze_candidate_source(code)
            self.assertFalse(cheat.cheat_detected, msg=f"Baseline '{op_name}' tripped anti-cheat: {cheat.reason}")

    def test_integer_inputs_preserved_across_quantized_and_indexed_extended_ops(self):
        rng = np.random.default_rng(42)
        int_ops = {
            "fused_cross_entropy_4096": ["bf16", "int64"],
            "int4_groupwise_dequant_gemm_512": ["bf16", "uint8", "bf16"],
            "fp8_e4m3_simulated_scaled_gemm_512": ["int8", "int8", "bf16", "bf16"],
            "paged_kv_gather_attn_256": ["bf16", "bf16", "bf16", "int32"],
        }
        for op_name, expected_plan in int_ops.items():
            inputs = OP_REGISTRY[op_name].make_inputs(rng)
            self.assertEqual(resolve_input_dtype_plan(inputs, "bf16"), expected_plan, msg=f"Integer preservation failed for {op_name}")

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in this sandbox; runs automatically on Torch CPU/GPU hosts.")
    def test_cpu_pytorch_vs_numpy_equivalence_all_19_ops_fp32_and_bf16(self):
        for op_name, torch_fn in TORCH_EXTENDED.items():
            spec = OP_REGISTRY[op_name]
            rng = np.random.default_rng(42)
            cases = [spec.make_inputs(rng)]
            if spec.make_stress_inputs is not None:
                cases.extend(spec.make_stress_inputs(np.random.default_rng(43)))

            for dtype in ("fp32", "bf16"):
                rtol = spec.rtol if dtype == "fp32" else max(spec.rtol, 2e-2)
                atol = spec.atol if dtype == "fp32" else max(spec.atol, 2e-2)
                for idx, np_in in enumerate(cases):
                    t_in = convert_numpy_inputs_to_torch(np_in, device="cpu", float_dtype=dtype)
                    quant_np_in = torch_inputs_to_numpy_fp32(t_in)
                    np_out = spec.reference_fn(*quant_np_in)
                    with torch.no_grad():
                        t_out = torch_fn(*t_in)
                    t_np = t_out.detach().float().cpu().numpy()
                    self.assertEqual(t_np.shape, np_out.shape, msg=f"{op_name} ({dtype}) case {idx} shape mismatch")
                    self.assertTrue(
                        np.allclose(t_np, np_out, rtol=rtol, atol=atol),
                        msg=f"{op_name} ({dtype}) case {idx} max_err={np.max(np.abs(t_np - np_out)):.3e}",
                    )


if __name__ == "__main__":
    unittest.main()
