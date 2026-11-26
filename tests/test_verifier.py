import json
import unittest
from sandbox_runner.verifier import verify_candidate_source


class VerifierBoundaryTests(unittest.TestCase):
    def test_infinite_candidate_is_killed(self):
        row = verify_candidate_source("rmsnorm_2048", "def kernel_forward(x,w):\n    while True: pass", timeout_secs=1)
        self.assertEqual(row.status, "TIMEOUT")
        json.dumps(row.to_dict(), allow_nan=False)

    def test_input_mutation_cannot_corrupt_reference(self):
        row = verify_candidate_source("rmsnorm_2048", "def kernel_forward(x,w):\n    x[:] = 0\n    return x")
        self.assertEqual(row.status, "CHEAT")
        self.assertTrue(row.cheat_detected)

    def test_strict_json_and_missing_compiler_measurement(self):
        source = "import numpy as np\ndef kernel_forward(x,w):\n    return x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*w"
        row = verify_candidate_source("rmsnorm_2048", source)
        self.assertEqual(row.status, "PASS", row.error_message)
        self.assertEqual(row.compile_ms, 0)
        self.assertEqual(row.speedup_vs_compile, 0)
        json.dumps(row.to_dict(), allow_nan=False)

    def test_file_and_introspection_access_rejected(self):
        for source in ("import numpy as np\nx=np.load('secret')", "x=().__class__", "x=open('secret')", "from numpy import ctypeslib"):
            self.assertEqual(verify_candidate_source("rmsnorm_2048", source).status, "CHEAT")

    def test_timeout_validation(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                verify_candidate_source("rmsnorm_2048", "", timeout_secs=value)

    def test_cached_output_cannot_pass_fresh_timing_inputs(self):
        source = """import numpy as np
def kernel_forward(x, w, cache=[None]):
    if cache[0] is None:
        cache[0] = x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*w
    return cache[0]
"""
        # A single correctness case deliberately passes. Timed calls must still
        # use new values and verify their outputs outside the measured interval.
        row = verify_candidate_source("rmsnorm_2048", source, stress_test=False)
        self.assertEqual(row.status, "RUNTIME_ERROR")
        self.assertIn("during timing", row.error_message)
        self.assertEqual(row.candidate_ms, 0)

    def test_delayed_input_mutation_cannot_pass_timing(self):
        source = """import numpy as np
def kernel_forward(x, w, calls=[0]):
    calls[0] += 1
    out = x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*w
    if calls[0] > 1:
        x[:] = 0
    return out
"""
        row = verify_candidate_source("rmsnorm_2048", source, stress_test=False)
        self.assertEqual(row.status, "RUNTIME_ERROR")
        self.assertIn("mutated input during timing", row.error_message)
        self.assertEqual(row.speedup_vs_eager, 0)

    def test_numpy_namespace_mutation_rejected(self):
        row = verify_candidate_source("rmsnorm_2048", "import numpy as np\nnp.sqrt = np.abs")
        self.assertEqual(row.status, "CHEAT")

    def test_syntax_error_is_a_compile_failure(self):
        row = verify_candidate_source("rmsnorm_2048", "def kernel_forward(:")
        self.assertEqual(row.status, "COMPILE_ERROR")
        self.assertFalse(row.compiled)
        self.assertFalse(row.cheat_detected)

    def test_unknown_operator_and_invalid_seed_rejected_before_worker(self):
        with self.assertRaises(ValueError):
            verify_candidate_source("missing_operator", "")
        with self.assertRaises(ValueError):
            verify_candidate_source("rmsnorm_2048", "", seed=-1)
