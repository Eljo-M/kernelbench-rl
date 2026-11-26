import inspect
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sandbox_runner.gpu_verifier import (
    _parse_strict_worker_json,
    build_sanitized_env,
    grade_candidate,
    validate_worker_payload,
    verify_candidate_gpu,
)
from sandbox_runner.gpu_worker import (
    ISOLATION_BOUNDARY,
    grade_candidate_in_worker,
    load_candidate_from_file,
)


class TestGpuVerifierAndWorker(unittest.TestCase):
    def test_sanitized_env_strips_unlisted_vars(self):
        fake_env = {
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "0",
            "UNLISTED_PRIVATE_VAR": "private_val_1",
        }
        clean = build_sanitized_env(base_env=fake_env)
        self.assertEqual(clean["PATH"], "/usr/bin")
        self.assertEqual(clean["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(clean["PYTHONNOUSERSITE"], "1")
        self.assertNotIn("UNLISTED_PRIVATE_VAR", clean)

    def test_load_candidate_from_real_temp_file_supports_inspect_getsource(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "real_cand.py"
            p.write_text("def kernel_forward(x):\n    return x + 1\n", encoding="utf-8")
            fn = load_candidate_from_file(p, module_name="kb_test_real_file_mod")
            src = inspect.getsource(fn)
            self.assertIn("def kernel_forward(x):", src)
            self.assertEqual(fn(4), 5)

    def test_strict_payload_and_returncode_validation_rejects_partial_or_mismatched_dicts(self):
        # Partial dict with only status and op_name must be rejected
        partial = '{"status":"PASS","op_name":"rmsnorm_2048"}'
        self.assertIsNone(_parse_strict_worker_json(partial, expected_run_id="r1", expected_op_name="rmsnorm_2048", expected_level=1, returncode=0))

        full = {
            "run_id": "r1",
            "op_name": "rmsnorm_2048",
            "level": 1,
            "backend": "triton",
            "status": "PASS",
            "compiled": True,
            "correct": True,
            "cheat_detected": False,
            "max_abs_err": 1e-5,
            "eager_ms": 0.2,
            "compile_ms": 0.15,
            "candidate_ms": 0.1,
            "speedup_vs_eager": 2.0,
            "speedup_vs_compile": 1.5,
            "reward": 2.29,
        }
        self.assertTrue(validate_worker_payload(full, expected_run_id="r1", expected_op_name="rmsnorm_2048", expected_level=1))
        for change in ({"correct": False}, {"candidate_ms": -1}, {"speedup_vs_eager": 20}, {"cheat_detected": True}):
            self.assertFalse(validate_worker_payload(dict(full, **change), expected_run_id="r1", expected_op_name="rmsnorm_2048", expected_level=1))
        # Wrong run_id rejected
        self.assertFalse(validate_worker_payload(full, expected_run_id="wrong", expected_op_name="rmsnorm_2048", expected_level=1))
        # Non-zero exit code with PASS rejected
        self.assertIsNone(_parse_strict_worker_json(json.dumps(full), expected_run_id="r1", expected_op_name="rmsnorm_2048", expected_level=1, returncode=1))

    def test_finite_timeout_seed_and_dtype_validation(self):
        with self.assertRaises(ValueError):
            verify_candidate_gpu("rmsnorm_2048", "def kernel_forward(x, w): return x", timeout_secs=float("inf"))
        with self.assertRaises(ValueError):
            verify_candidate_gpu("rmsnorm_2048", "def kernel_forward(x, w): return x", timeout_secs=float("nan"))
        with self.assertRaises(ValueError):
            verify_candidate_gpu("rmsnorm_2048", "def kernel_forward(x, w): return x", seed=-1)
        with self.assertRaises(ValueError):
            verify_candidate_gpu("rmsnorm_2048", "def kernel_forward(x, w): return x", float_dtype="int4")

    def test_supervisor_handles_timeout_and_oom_with_source_hash(self):
        src = "def kernel_forward(x, w): return x\n"
        with mock.patch("sandbox_runner.gpu_verifier.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1.0)):
            res = verify_candidate_gpu("rmsnorm_2048", src, timeout_secs=1.0, run_id="t-to")
            self.assertEqual(res["status"], "TIMEOUT")
            self.assertEqual(res["candidate_source"], src)
            self.assertEqual(len(res["candidate_sha256"]), 64)
            json.dumps(res, allow_nan=False)

        fake_oom = subprocess.CompletedProcess(args=["python"], returncode=137, stdout="", stderr="Killed")
        with mock.patch("sandbox_runner.gpu_verifier.subprocess.run", return_value=fake_oom):
            res_oom = verify_candidate_gpu("rmsnorm_2048", src, timeout_secs=5.0, run_id="t-oom")
            self.assertEqual(res_oom["status"], "OOM")
            json.dumps(res_oom, allow_nan=False)

    def test_real_isolated_child_worker_runs_via_repo_root_bootstrap(self):
        valid_np = (
            "import numpy as np\n"
            "def kernel_forward(x, w, eps=1e-6):\n"
            "    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)) * w\n"
        )
        gpu_res = grade_candidate("rmsnorm_2048", valid_np, backend="triton", float_dtype="fp16", timeout_secs=15.0, run_id="iso-test")
        self.assertEqual(gpu_res["run_id"], "iso-test")
        self.assertEqual(gpu_res["backend"], "triton")
        self.assertEqual(gpu_res["status"], "RUNTIME_ERROR")
        self.assertEqual(gpu_res["gpu_model"], "unavailable")
        self.assertEqual(gpu_res["candidate_source"], valid_np)
        self.assertEqual(len(gpu_res["candidate_sha256"]), 64)
        self.assertEqual(gpu_res["isolation_boundary"], ISOLATION_BOUNDARY)
        json.dumps(gpu_res, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
