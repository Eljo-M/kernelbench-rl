from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from contracts.schemas import KernelEvalResult
from kernels.op_registry import OP_REGISTRY
from rl_trainer.grpo_reward import compute_kernel_reward

NUMPY_NAMES = set("abs absolute add arange array asarray broadcast_to clip concatenate cos divide einsum empty_like exp expand_dims float16 float32 float64 full_like int32 int64 isfinite log log1p matmul max maximum mean min minimum multiply ndarray ones_like pi power repeat reshape sin split sqrt square squeeze stack subtract sum swapaxes tanh transpose var where zeros_like".split())
ARRAY_NAMES = set("T astype copy dtype flatten item max mean min ndim ravel reshape shape size squeeze sum swapaxes transpose".split())
FORBIDDEN_NAMES = set("open exec eval compile getattr setattr delattr globals locals vars input type object help breakpoint memoryview exit quit".split())


def detect_cheating(source_code: str) -> tuple[bool, str]:
    """Conservative NumPy development policy, not a security sandbox."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    from sandbox_runner.anti_cheat import analyze_candidate_source
    report = analyze_candidate_source(source_code)
    if report.cheat_detected:
        return True, report.reason
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "numpy":
                    return True, f"Forbidden import detected: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module != "numpy" or node.level or any(a.name not in NUMPY_NAMES for a in node.names):
                return True, f"Forbidden from-import detected: {node.module}"
            if node.module and "torch.nn.functional" in node.module:
                return True, "Cheating detected: importing torch.nn.functional"
        elif isinstance(node, ast.Attribute) and node.attr not in NUMPY_NAMES | ARRAY_NAMES:
            return True, f"Cheating detected: calling reference attr '{node.attr}'"
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True, "Attribute mutation is forbidden"
        elif isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in FORBIDDEN_NAMES):
            return True, f"Unsupported name: {node.id}"
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            return True, f"Unsupported syntax: {type(node).__name__}"
    return False, ""


def _time_callable(fn: Callable, spec, seed: int, warmup: int = 3, iters: int = 15) -> float:
    """Vary values outside the timer and verify every timed invocation afterwards."""
    rng = np.random.default_rng(seed)
    times = []
    for i in range(warmup + iters):
        inputs = spec.make_inputs(rng)
        fresh = tuple(x.copy() for x in inputs)
        t0 = time.perf_counter()
        out = fn(*fresh)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if any(not np.array_equal(x, y) for x, y in zip(inputs, fresh)):
            raise ValueError("Candidate mutated input during timing")
        ref = spec.reference_fn(*inputs)
        if not isinstance(out, np.ndarray) or out.shape != ref.shape or not np.all(np.isfinite(out)) or not np.allclose(out, ref, rtol=spec.rtol, atol=spec.atol):
            raise ValueError("Candidate produced invalid output during timing")
        if i >= warmup:
            times.append(elapsed_ms)
    return max(float(np.median(times)), 1e-9)


def _verify_in_process(op_name: str, source_code: str, run_id: str = "dev",
                       seed: int = 42, stress_test: bool = True) -> KernelEvalResult:
    from types import SimpleNamespace
    import builtins
    spec = OP_REGISTRY[op_name]
    numpy_api = SimpleNamespace(**{name: getattr(np, name) for name in NUMPY_NAMES})
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name != "numpy" or level:
            raise ImportError("Only numpy is available")
        return numpy_api
    safe = {name: getattr(builtins, name) for name in "abs bool enumerate float int len list max min range round tuple zip".split()}
    namespace = {"__builtins__": {**safe, "__import__": guarded_import}}
    try:
        exec(compile(source_code, "<candidate_kernel>", "exec"), namespace)
        candidate_fn = namespace["kernel_forward"]
        if not callable(candidate_fn):
            raise TypeError("kernel_forward must be callable")
    except MemoryError:
        return _failure(op_name, run_id, "OOM", "Candidate allocation failed")
    except Exception as exc:
        return _failure(op_name, run_id, "COMPILE_ERROR", f"{type(exc).__name__}: {exc}")

    canonical_inputs = spec.make_inputs(np.random.default_rng(seed))
    cases = [canonical_inputs]
    if stress_test:
        cases.extend(spec.make_inputs(np.random.default_rng(seed + i)) for i in (1, 2))
        stress = getattr(spec, "make_stress_inputs", None)
        if stress:
            cases.extend(stress(np.random.default_rng(seed + 3)))
    worst_err = 0.0
    for inputs in cases:
        ref = spec.reference_fn(*(x.copy() for x in inputs))
        cand_inputs = tuple(x.copy() for x in inputs)
        try:
            out = candidate_fn(*cand_inputs)
            if any(not np.array_equal(x, y) for x, y in zip(inputs, cand_inputs)):
                return _failure(op_name, run_id, "CHEAT", "Candidate mutated input", compiled=True)
            if not isinstance(out, np.ndarray) or out.shape != ref.shape or not np.all(np.isfinite(out)):
                return _failure(op_name, run_id, "NUMERICAL_MISMATCH", "Invalid type, shape or non-finite output", compiled=True)
            err = float(np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64)))) if out.size else 0.0
            worst_err = max(worst_err, err)
            if not np.allclose(out, ref, rtol=spec.rtol, atol=spec.atol):
                return _failure(op_name, run_id, "NUMERICAL_MISMATCH", "Output exceeded numerical tolerances", compiled=True, max_abs_err=worst_err)
        except MemoryError:
            return _failure(op_name, run_id, "OOM", "Candidate allocation failed", compiled=True)
        except Exception as exc:
            return _failure(op_name, run_id, "RUNTIME_ERROR", f"{type(exc).__name__}: {exc}", compiled=True)
    try:
        eager_ms = _time_callable(spec.reference_fn, spec, seed + 1000)
        candidate_ms = _time_callable(candidate_fn, spec, seed + 1000)
    except MemoryError:
        return _failure(op_name, run_id, "OOM", "Timing allocation failed", compiled=True)
    except Exception as exc:
        return _failure(op_name, run_id, "RUNTIME_ERROR", f"Timing failed: {exc}", compiled=True)
    speedup = eager_ms / candidate_ms
    return KernelEvalResult(
        run_id=run_id, op_name=op_name, level=spec.level, backend="numpy_dev", status="PASS",
        compiled=True, correct=True, cheat_detected=False, max_abs_err=worst_err,
        eager_ms=eager_ms, compile_ms=0.0, candidate_ms=candidate_ms,
        speedup_vs_eager=speedup, speedup_vs_compile=0.0,
        reward=compute_kernel_reward(True, True, False, speedup, 0.0),
        achieved_tflops=spec.flops / (candidate_ms * 1e-3) / 1e12,
        arithmetic_intensity=spec.flops / max(spec.bytes_moved, 1),
    )


def _failure(op_name: str, run_id: str, status: str, message: str, *, compiled: bool = False, max_abs_err: float = 0.0) -> KernelEvalResult:
    return KernelEvalResult(
        run_id=run_id, op_name=op_name, level=OP_REGISTRY[op_name].level,
        backend="numpy_dev", status=status, compiled=compiled, correct=False,
        cheat_detected=status == "CHEAT", max_abs_err=max_abs_err,
        eager_ms=0.0, compile_ms=0.0, candidate_ms=0.0,
        speedup_vs_eager=0.0, speedup_vs_compile=0.0,
        reward=compute_kernel_reward(compiled, False, status == "CHEAT", 0.0, 0.0), error_message=message[:2000],
    )


def verify_candidate_source(op_name: str, source_code: str, run_id: str = "dev",
                            seed: int = 42, timeout_secs: float = 30.0, stress_test: bool = True) -> KernelEvalResult:
    """Evaluate in a disposable process with a timeout and a reduced environment.

    This is a CPU development runner. The AST policy is not a security boundary;
    externally isolate adversarial candidates. Windows has no hard memory cap here.
    """
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unknown operator {op_name!r}")
    if not math.isfinite(timeout_secs) or timeout_secs <= 0:
        raise ValueError("timeout_secs must be positive and finite")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if len(source_code.encode("utf-8")) > 100_000:
        return _failure(op_name, run_id, "CHEAT", "Candidate exceeds 100 KB source limit")
    cheated, message = detect_cheating(source_code)
    if cheated:
        return _failure(op_name, run_id, "CHEAT", message)
    if message:
        return _failure(op_name, run_id, "COMPILE_ERROR", message)
    payload = dict(op_name=op_name, source_code=source_code, run_id=run_id, seed=seed, stress_test=stress_test)
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "LD_LIBRARY_PATH",
    }}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="kernelbench-") as directory:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("worker.py").resolve())],
                input=json.dumps(payload), text=True, capture_output=True, cwd=directory,
                env=env, timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            return _failure(op_name, run_id, "TIMEOUT", f"Exceeded {timeout_secs:g}s worker timeout")
    if result.returncode:
        return _failure(op_name, run_id, "RUNTIME_ERROR", f"Worker exited {result.returncode}: {result.stderr[-1000:]}")
    try:
        return KernelEvalResult.from_dict(json.loads(result.stdout,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
    except (ValueError, TypeError) as exc:
        return _failure(op_name, run_id, "RUNTIME_ERROR", f"Invalid worker result: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--run-id", default="cli")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-secs", type=float, default=30)
    args = parser.parse_args()
    source = Path(args.candidate_path).read_text(encoding="utf-8")
    result = verify_candidate_source(args.op_name, source, run_id=args.run_id, seed=args.seed, timeout_secs=args.timeout_secs)
    print(json.dumps(result.to_dict(), allow_nan=False))


if __name__ == "__main__":
    main()
