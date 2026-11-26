import unittest

import numpy as np

from kernels.op_registry import OP_REGISTRY, get_operators_by_level, verify_registry
from kernels.triton_baselines import TRITON_BASELINES, validate_triton_baselines_syntax
from sandbox_runner.anti_cheat import analyze_candidate_source
from sandbox_runner.verifier import verify_candidate_source


class TestTrack2OperatorRegistryAndBaselines(unittest.TestCase):
    def test_operator_counts_by_tier(self):
        self.assertEqual(len(OP_REGISTRY), 30)
        self.assertEqual(len(get_operators_by_level(1)), 12)
        self.assertEqual(len(get_operators_by_level(2)), 10)
        self.assertEqual(len(get_operators_by_level(3)), 8)

    def test_all_30_operators_deterministic_and_finite(self):
        self.assertTrue(verify_registry(seed=42))

    def test_analytical_roofline_metrics_positive(self):
        for name, spec in OP_REGISTRY.items():
            self.assertGreater(spec.flops, 0, msg=f"{name} flops must be > 0")
            self.assertGreater(spec.bytes_moved, 0, msg=f"{name} bytes_moved must be > 0")
            self.assertGreater(spec.arithmetic_intensity, 0.0, msg=f"{name} AI must be > 0")

    def test_triton_baselines_registered_valid_syntax_and_no_cheat(self):
        self.assertGreaterEqual(len(TRITON_BASELINES), 10)
        syntax_results = validate_triton_baselines_syntax()
        for op_name, valid in syntax_results.items():
            self.assertIn(op_name, OP_REGISTRY)
            self.assertTrue(valid, msg=f"Triton baseline for {op_name} missing @triton.jit or kernel_forward")
            cheat = analyze_candidate_source(TRITON_BASELINES[op_name])
            self.assertFalse(
                cheat.cheat_detected,
                msg=f"Triton baseline for {op_name} tripped anti-cheat: {cheat.reason}",
            )

    def test_verifier_compatibility_across_levels(self):
        # Level 1: layernorm_2048
        ln_code = (
            "import numpy as np\n"
            "def kernel_forward(x, w, b, eps=1e-5):\n"
            "    m = np.mean(x, axis=-1, keepdims=True)\n"
            "    v = np.var(x, axis=-1, keepdims=True)\n"
            "    return ((x - m) / np.sqrt(v + eps)) * w + b\n"
        )
        res_l1 = verify_candidate_source("layernorm_2048", ln_code, run_id="t2-l1")
        self.assertEqual(res_l1.status, "PASS")
        self.assertTrue(res_l1.correct)

        # Level 2: rope_fwd_128
        rope_code = (
            "import numpy as np\n"
            "def kernel_forward(x, cos, sin):\n"
            "    h = x.shape[-1] // 2\n"
            "    x1, x2 = x[..., :h], x[..., h:]\n"
            "    c, s = cos[None, None, :, :], sin[None, None, :, :]\n"
            "    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)\n"
        )
        res_l2 = verify_candidate_source("rope_fwd_128", rope_code, run_id="t2-l2")
        self.assertEqual(res_l2.status, "PASS")
        self.assertTrue(res_l2.correct)

        # Level 3: int8_weight_only_gemm_512
        i8_code = (
            "import numpy as np\n"
            "def kernel_forward(x, qw, sc):\n"
            "    return x @ (qw.astype(np.float32) * sc[None, :])\n"
        )
        res_l3 = verify_candidate_source("int8_weight_only_gemm_512", i8_code, run_id="t2-l3")
        self.assertEqual(res_l3.status, "PASS")
        self.assertTrue(res_l3.correct)


if __name__ == "__main__":
    unittest.main()
