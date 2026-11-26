import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kernels.op_registry import OP_REGISTRY
from kernels.torch_references import (
    TORCH_REFERENCES,
    is_integer_numpy_array,
    resolve_input_dtype_plan,
)
from kernels.triton_baselines import TRITON_BASELINES, validate_triton_baselines_syntax
from kernels.validate_cuda import (
    _make_error_record,
    benchmark_compiled_reference,
    check_cuda_triton_environment,
    classify_cuda_exception,
    compute_quantized_numpy_reference,
    main as cuda_cli_main,
    serialize_eval_record,
    tolerances_for_dtype,
    validate_cli_args,
)


class _FakeTensor:
    """Minimal mock tensor simulating bf16 quantized float vs preserved integer tensor."""

    def __init__(self, arr: np.ndarray, is_float: bool):
        self._arr = arr
        self._is_float = is_float

    def is_floating_point(self) -> bool:
        return self._is_float

    def detach(self):
        return self

    def float(self):
        if self._is_float:
            return _FakeTensor(self._arr.astype(np.float32), True)
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


def _simulate_bf16_round(arr: np.ndarray) -> np.ndarray:
    """Truncate fp32 mantissa to 7 bits (exact bfloat16 rounding) in pure NumPy."""
    u32 = arr.astype(np.float32).view(np.uint32)
    rounded = ((u32 + np.uint32(0x7FFF)) & np.uint32(0xFFFF0000)).view(np.float32)
    return rounded


class TestTrack2CudaCliAndTorchRefs(unittest.TestCase):
    def test_torch_references_cover_all_30_triton_baselines(self):
        self.assertEqual(set(TORCH_REFERENCES.keys()), set(TRITON_BASELINES.keys()))
        self.assertEqual(len(TORCH_REFERENCES), 30)
        for op_name in TORCH_REFERENCES:
            self.assertIn(op_name, OP_REGISTRY)
            self.assertTrue(callable(TORCH_REFERENCES[op_name]))

    def test_integer_quantization_inputs_stay_integer_in_dtype_plan(self):
        rng = np.random.default_rng(42)
        inputs = OP_REGISTRY["int8_weight_only_gemm_512"].make_inputs(rng)
        self.assertFalse(is_integer_numpy_array(inputs[0]))
        self.assertTrue(is_integer_numpy_array(inputs[1]))
        self.assertFalse(is_integer_numpy_array(inputs[2]))
        self.assertEqual(resolve_input_dtype_plan(inputs, "fp32"), ["fp32", "int8", "fp32"])
        self.assertEqual(resolve_input_dtype_plan(inputs, "bf16"), ["bf16", "int8", "bf16"])

    def test_triton_baselines_enforce_ieee_dot_and_contiguous(self):
        syntax_map = validate_triton_baselines_syntax()
        self.assertEqual(len(syntax_map), 30)
        for op_name, ok in syntax_map.items():
            self.assertTrue(ok, msg=f"Baseline {op_name} failed AST/IEEE/contiguous check")

    def test_strict_failure_json_serialization_forbids_nan_and_inf(self):
        spec = OP_REGISTRY["rmsnorm_2048"]
        rec = _make_error_record(
            "cuda-val-rmsnorm_2048-fp32",
            spec,
            "COMPILE_ERROR",
            compiled=False,
            err_msg="Triton JIT failed",
            max_abs_err=float("inf"),
        )
        self.assertEqual(rec.max_abs_err, 0.0)
        raw = serialize_eval_record(
            rec,
            seed=42,
            dtype="fp32",
            gpu_model="NVIDIA H100 80GB",
            method="triton_baseline",
        )
        self.assertNotIn("Infinity", raw)
        self.assertNotIn("NaN", raw)
        parsed = json.loads(raw)
        self.assertEqual(parsed["status"], "COMPILE_ERROR")
        self.assertEqual(parsed["max_abs_err"], 0.0)
        self.assertEqual(parsed["seed"], 42)
        self.assertEqual(parsed["dtype"], "fp32")
        self.assertEqual(parsed["method"], "triton_baseline")
        self.assertEqual(parsed["gpu_model"], "NVIDIA H100 80GB")

    def test_cli_invalid_args_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as td:
            existing = Path(td) / "existing.jsonl"
            existing.write_text("already here\n", encoding="utf-8")
            fresh = Path(td) / "fresh.jsonl"

            # Unknown op rejected
            ok, msg = validate_cli_args(["not_a_real_op"], 42, 10, 50, fresh, False)
            self.assertFalse(ok)
            self.assertIn("Unknown", msg)

            # Negative seed / negative warmup / zero iters rejected
            self.assertFalse(validate_cli_args(["rmsnorm_2048"], -1, 10, 50, fresh, False)[0])
            self.assertFalse(validate_cli_args(["rmsnorm_2048"], 42, -1, 50, fresh, False)[0])
            self.assertFalse(validate_cli_args(["rmsnorm_2048"], 42, 10, 0, fresh, False)[0])

            # Existing output rejected without --overwrite
            ok_no_ow, msg_no_ow = validate_cli_args(["rmsnorm_2048"], 42, 10, 50, existing, False)
            self.assertFalse(ok_no_ow)
            self.assertIn("--overwrite", msg_no_ow)

            # Accepted with --overwrite
            self.assertTrue(validate_cli_args(["rmsnorm_2048"], 42, 10, 50, existing, True)[0])

            # CLI returns exit code 2 on invalid args and leaves existing file untouched
            rc = cuda_cli_main(["--ops", "nonexistent_op", "--out", str(fresh)])
            self.assertEqual(rc, 2)
            self.assertFalse(fresh.exists())

            rc_ow = cuda_cli_main(["--ops", "rmsnorm_2048", "--out", str(existing)])
            self.assertEqual(rc_ow, 2)
            self.assertEqual(existing.read_text(encoding="utf-8"), "already here\n")

    def test_bf16_quantized_input_roundtrip_avoids_false_quantization_blame(self):
        spec = OP_REGISTRY["int8_weight_only_gemm_512"]
        rng = np.random.default_rng(7)
        x_fp32, qw_int8, scales_fp32 = spec.make_inputs(rng)

        # Simulate bf16 quantization on float inputs while keeping qw_int8 strictly int8
        x_bf16 = _simulate_bf16_round(x_fp32)
        scales_bf16 = _simulate_bf16_round(scales_fp32)
        mock_gpu_inputs = (
            _FakeTensor(x_bf16, True),
            _FakeTensor(qw_int8, False),
            _FakeTensor(scales_bf16, True),
        )

        # Recomputed NumPy reference matches exact quantized GPU inputs
        recomputed_ref = compute_quantized_numpy_reference(spec, mock_gpu_inputs)
        direct_quant_ref = spec.reference_fn(x_bf16, qw_int8, scales_bf16)
        self.assertTrue(np.array_equal(recomputed_ref, direct_quant_ref))

        # Confirm unquantized fp32 inputs actually differ from quantized bf16 inputs
        unquant_ref = spec.reference_fn(x_fp32, qw_int8, scales_fp32)
        self.assertGreater(float(np.max(np.abs(unquant_ref - recomputed_ref))), 0.0)

    def test_benchmark_compiled_reference_validates_correctness_and_keeps_raw_timing(self):
        eager_ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fake_cuda = SimpleNamespace(synchronize=lambda: None)

        # Case 1: Correct compiled output with very short raw latency (0.000042 ms) is NOT rounded to 0
        fake_torch_ok = SimpleNamespace(
            cuda=fake_cuda,
            compile=lambda fn: (lambda *args: _FakeTensor(eager_ref.copy(), True)),
        )
        ms_ok, note_ok = benchmark_compiled_reference(
            fake_torch_ok,
            lambda x: x,
            (_FakeTensor(eager_ref, True),),
            eager_ref,
            rtol=1e-4,
            atol=1e-4,
            warmup=1,
            iters=3,
            timer_fn=lambda *args: 0.000042,
        )
        self.assertEqual(note_ok, "")
        self.assertAlmostEqual(ms_ok, 0.000042, places=9)
        self.assertGreater(ms_ok, 0.0)

        # Case 2: Numerically wrong compiled output rejected -> compile_ms = 0.0 + explicit error
        fake_torch_bad = SimpleNamespace(
            cuda=fake_cuda,
            compile=lambda fn: (lambda *args: _FakeTensor(np.zeros_like(eager_ref), True)),
        )
        ms_bad, note_bad = benchmark_compiled_reference(
            fake_torch_bad,
            lambda x: x,
            (_FakeTensor(eager_ref, True),),
            eager_ref,
            rtol=1e-4,
            atol=1e-4,
            warmup=1,
            iters=3,
            timer_fn=lambda *args: 0.1,
        )
        self.assertEqual(ms_bad, 0.0)
        self.assertIn("mismatch", note_bad.lower())

        # Case 3: torch.compile raising exception -> compile_ms = 0.0 + explicit error
        def _boom(_fn):
            raise RuntimeError("inductor backend unavailable")

        fake_torch_err = SimpleNamespace(cuda=fake_cuda, compile=_boom)
        ms_err, note_err = benchmark_compiled_reference(
            fake_torch_err,
            lambda x: x,
            (_FakeTensor(eager_ref, True),),
            eager_ref,
            rtol=1e-4,
            atol=1e-4,
            warmup=1,
            iters=3,
        )
        self.assertEqual(ms_err, 0.0)
        self.assertIn("inductor backend unavailable", note_err)

    def test_exception_classification_compile_oom_and_runtime(self):
        class CompilationError(Exception):
            pass

        class OutOfMemoryError(Exception):
            pass

        self.assertEqual(classify_cuda_exception(CompilationError("ptxas failed"), False), ("COMPILE_ERROR", False))
        self.assertEqual(classify_cuda_exception(OutOfMemoryError("CUDA out of memory"), True), ("OOM", True))
        self.assertEqual(classify_cuda_exception(RuntimeError("illegal memory access"), True), ("RUNTIME_ERROR", True))
        self.assertEqual(classify_cuda_exception(RuntimeError("generic error"), False), ("RUNTIME_ERROR", False))

    def test_tolerances_for_fp32_and_bf16(self):
        spec = OP_REGISTRY["rmsnorm_2048"]
        self.assertEqual(tolerances_for_dtype(spec, "fp32"), (spec.rtol, spec.atol))
        bf_rtol, bf_atol = tolerances_for_dtype(spec, "bf16")
        self.assertGreaterEqual(bf_rtol, 1e-2)
        self.assertGreaterEqual(bf_atol, 1e-2)

    def test_cuda_cli_fails_clearly_without_cuda_and_writes_no_fake_traces(self):
        ok, msg = check_cuda_triton_environment()
        if ok:
            self.skipTest(f"Running on CUDA host ({msg}); missing-CUDA failure test skipped.")

        with tempfile.TemporaryDirectory() as td:
            out_jsonl = Path(td) / "should_not_exist.jsonl"
            rc = cuda_cli_main(["--out", str(out_jsonl)])
            self.assertEqual(rc, 2)
            self.assertFalse(out_jsonl.exists())

            proc = subprocess.run(
                [sys.executable, "-m", "kernels.validate_cuda", "--out", str(out_jsonl)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("[ERROR]", proc.stderr)
            self.assertFalse(out_jsonl.exists())


if __name__ == "__main__":
    unittest.main()
