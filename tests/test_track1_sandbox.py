import unittest

from sandbox_runner.anti_cheat import analyze_candidate_source
from sandbox_runner.verifier import verify_candidate_source


class TestTrack1SandboxHardening(unittest.TestCase):
    def test_blocks_eval_exec_and_dunder_escape(self):
        self.assertTrue(analyze_candidate_source("def kernel_forward(x): return eval('x')").cheat_detected)
        self.assertTrue(analyze_candidate_source("def kernel_forward(x): return x.__class__").cheat_detected)
        self.assertTrue(analyze_candidate_source("import subprocess\ndef kernel_forward(x): return x").cheat_detected)

    def test_blocks_global_cache_memoization_cheat(self):
        code = (
            "CACHE = {}\n"
            "def kernel_forward(x, w):\n"
            "    CACHE['k'] = x\n"
            "    return x\n"
        )
        rep = analyze_candidate_source(code)
        self.assertTrue(rep.cheat_detected)
        self.assertIn("Memoization", rep.reason)

    def test_blocks_in_place_input_mutation(self):
        code = (
            "import numpy as np\n"
            "def kernel_forward(x, w, eps=1e-6):\n"
            "    x[:] = 0.0\n"
            "    return x\n"
        )
        res = verify_candidate_source("rmsnorm_2048", code, run_id="t1-mutate")
        self.assertEqual(res.status, "CHEAT")
        self.assertTrue(res.cheat_detected)
        self.assertEqual(res.reward, -1.0)

    def test_multi_shape_stress_catches_hardcoded_dimensions(self):
        # Hardcodes shape (512, 2048) -> fails on stress variant (17, 771)
        code = (
            "import numpy as np\n"
            "def kernel_forward(x, w, eps=1e-6):\n"
            "    assert x.shape == (512, 2048)\n"
            "    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)) * w\n"
        )
        res = verify_candidate_source("rmsnorm_2048", code, run_id="t1-hardcoded", stress_test=True)
        self.assertEqual(res.status, "RUNTIME_ERROR")
        self.assertFalse(res.correct)

    def test_softmax_overflow_stability_stress_test(self):
        # Naive exp(x)/sum(exp(x)) overflows to NaN on *50.0 stress input!
        naive = (
            "import numpy as np\n"
            "def kernel_forward(x):\n"
            "    e = np.exp(x)\n"
            "    return e / np.sum(e, axis=-1, keepdims=True)\n"
        )
        res_bad = verify_candidate_source("fused_softmax_2048", naive, run_id="t1-naive-sm", stress_test=True)
        self.assertEqual(res_bad.status, "NUMERICAL_MISMATCH")

        stable = (
            "import numpy as np\n"
            "def kernel_forward(x):\n"
            "    s = x - np.max(x, axis=-1, keepdims=True)\n"
            "    e = np.exp(s)\n"
            "    return e / np.sum(e, axis=-1, keepdims=True)\n"
        )
        res_ok = verify_candidate_source("fused_softmax_2048", stable, run_id="t1-stable-sm", stress_test=True)
        self.assertEqual(res_ok.status, "PASS")
        self.assertTrue(res_ok.correct)

    def test_missing_entrypoint_is_compile_error(self):
        res = verify_candidate_source("rmsnorm_2048", "def wrong_name(x): return x", run_id="t1-no-entry")
        self.assertEqual(res.status, "COMPILE_ERROR")
        self.assertEqual(res.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
