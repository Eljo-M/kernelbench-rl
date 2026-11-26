"""Timeout-bounded GPU verifier supervisor and unified CPU/GPU grading router.

Launches `sandbox_runner/gpu_worker.py` directly via `python -s -E <script> --repo-root <abs>`
so isolated execution never depends on `PYTHONPATH` or `cwd` (avoiding `-I` import failures),
and strictly validates all required `KernelEvalResult` fields, types, identity fields
(`run_id`, `op_name`, `backend`, `level`), and worker exit codes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from contracts.schemas import KernelEvalResult
from kernels.op_registry import OP_REGISTRY
from kernels.validate_cuda import _make_error_record, serialize_eval_record
from sandbox_runner.gpu_worker import ISOLATION_BOUNDARY
from sandbox_runner.verifier import verify_candidate_source

ALLOWED_WORKER_ENV_KEYS = (
    "PATH",
    "TEMP",
    "TMP",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "TRITON_LIBCUDA_PATH",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_CUDA_ARCH_LIST",
    "HOME",
)

VALID_STATUSES = {
    "PASS",
    "COMPILE_ERROR",
    "NUMERICAL_MISMATCH",
    "TIMEOUT",
    "OOM",
    "RUNTIME_ERROR",
    "CHEAT",
}

REQUIRED_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "run_id": (str,),
    "op_name": (str,),
    "level": (int,),
    "backend": (str,),
    "status": (str,),
    "compiled": (bool,),
    "correct": (bool,),
    "cheat_detected": (bool,),
    "max_abs_err": (int, float),
    "eager_ms": (int, float),
    "compile_ms": (int, float),
    "candidate_ms": (int, float),
    "speedup_vs_eager": (int, float),
    "speedup_vs_compile": (int, float),
    "reward": (int, float),
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reject_non_finite_json(token: str) -> None:
    raise ValueError(f"Non-finite JSON constant rejected: {token}")


def build_sanitized_env(
    repo_root: Path | None = None,
    base_env: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal allow-listed environment for the child GPU worker."""
    src = os.environ if base_env is None else base_env
    clean: dict[str, str] = {}
    for key in ALLOWED_WORKER_ENV_KEYS:
        if key in src and src[key]:
            clean[key] = src[key]
    if os.name == "nt":
        for win_key in ("SystemRoot", "USERPROFILE", "LOCALAPPDATA", "USERNAME", "COMSPEC", "SystemDrive"):
            if win_key in src and src[win_key]:
                clean[win_key] = src[win_key]
    clean["PYTHONNOUSERSITE"] = "1"
    if extra_env:
        for k, v in extra_env.items():
            clean[k] = v
    return clean


def _build_supervisor_fallback(
    *,
    run_id: str,
    op_name: str,
    source_code: str,
    status: str,
    err_msg: str,
    seed: int,
    independent_seed: int,
    dtype: str,
    method: str,
) -> dict[str, Any]:
    spec = OP_REGISTRY[op_name]
    rec = _make_error_record(run_id, spec, status, compiled=False, err_msg=err_msg, max_abs_err=0.0)
    raw = serialize_eval_record(
        rec,
        seed=seed,
        dtype=dtype,
        gpu_model="unavailable",
        method=method,
    )
    payload = json.loads(raw)
    payload["independent_seed"] = int(independent_seed)
    payload["isolation_boundary"] = ISOLATION_BOUNDARY
    payload["candidate_source"] = source_code
    payload["candidate_sha256"] = _sha256_text(source_code)
    return payload


def validate_worker_payload(
    obj: Any,
    *,
    expected_run_id: str,
    expected_op_name: str,
    expected_level: int,
    expected_backend: str = "triton",
) -> bool:
    """Strictly verify all required KernelEvalResult fields, types, finiteness, and identity match."""
    if not isinstance(obj, dict):
        return False
    for field, allowed_types in REQUIRED_FIELD_TYPES.items():
        if field not in obj:
            return False
        val = obj[field]
        # In Python isinstance(True, int) is True; guard bool vs int/float strictly
        if allowed_types == (bool,):
            if not isinstance(val, bool):
                return False
        else:
            if isinstance(val, bool) or not isinstance(val, allowed_types):
                return False
            if isinstance(val, float) and not math.isfinite(val):
                return False
    if obj["status"] not in VALID_STATUSES:
        return False
    if obj["run_id"] != expected_run_id:
        return False
    if obj["op_name"] != expected_op_name:
        return False
    if obj["backend"] != expected_backend:
        return False
    if obj["level"] != expected_level:
        return False
    for field in ("max_abs_err", "eager_ms", "compile_ms", "candidate_ms", "speedup_vs_eager", "speedup_vs_compile"):
        if obj[field] < 0:
            return False
    if obj["status"] == "PASS":
        if not obj["compiled"] or not obj["correct"] or obj["cheat_detected"]:
            return False
        if min(obj["eager_ms"], obj["candidate_ms"], obj["speedup_vs_eager"]) <= 0:
            return False
        if not math.isclose(obj["speedup_vs_eager"], obj["eager_ms"] / obj["candidate_ms"], rel_tol=1e-5):
            return False
        expected_compile = obj["compile_ms"] / obj["candidate_ms"]
        if not math.isclose(obj["speedup_vs_compile"], expected_compile, rel_tol=1e-5, abs_tol=1e-10):
            return False
    elif obj["correct"] or (obj["cheat_detected"] != (obj["status"] == "CHEAT")):
        return False
    for val in obj.values():
        if isinstance(val, float) and not math.isfinite(val):
            return False
    return True


def _parse_strict_worker_json(
    stdout: str,
    *,
    expected_run_id: str,
    expected_op_name: str,
    expected_level: int,
    returncode: int = 0,
) -> dict[str, Any] | None:
    """Parse and strictly validate worker JSON + return code."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for candidate in reversed(lines):
        try:
            obj = json.loads(candidate, parse_constant=_reject_non_finite_json)
        except Exception:
            continue
        if not validate_worker_payload(
            obj,
            expected_run_id=expected_run_id,
            expected_op_name=expected_op_name,
            expected_level=expected_level,
        ):
            continue
        # Only accept returncode 0, or returncode 2 when explicitly reporting unavailable GPU/CUDA
        if returncode == 0:
            return obj
        if returncode == 2 and obj.get("status") == "RUNTIME_ERROR" and obj.get("gpu_model") == "unavailable":
            return obj
        return None
    return None


def verify_candidate_gpu(
    op_name: str,
    source_code: str,
    *,
    float_dtype: str = "bf16",
    seed: int = 42,
    independent_seed: int | None = None,
    warmup: int = 10,
    iters: int = 50,
    timeout_secs: float = 45.0,
    run_id: str = "gpu-eval",
    method: str = "rl_candidate",
    stress_test: bool = True,
    repo_root: Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Grade a candidate Triton kernel inside a timeout-bounded sanitized child process."""
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unknown operator {op_name!r}. Available: {sorted(OP_REGISTRY)}")
    if float_dtype not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported float_dtype {float_dtype!r}. Expected 'fp32', 'bf16', or 'fp16'.")
    if not (isinstance(timeout_secs, (int, float)) and not isinstance(timeout_secs, bool) and math.isfinite(timeout_secs) and timeout_secs > 0):
        raise ValueError(f"timeout_secs must be finite > 0, got {timeout_secs}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (warmup, iters)) or warmup < 0 or iters <= 0:
        raise ValueError(f"Invalid timing params: warmup={warmup}, iters={iters}")

    spec = OP_REGISTRY[op_name]
    indep_seed = (seed + 10_000) if independent_seed is None else independent_seed
    if isinstance(indep_seed, bool) or not isinstance(indep_seed, int) or indep_seed < 0:
        raise ValueError("independent_seed must be an integer >= 0")
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    worker_script = root / "sandbox_runner" / "gpu_worker.py"
    env = build_sanitized_env(repo_root=root)
    py_exe = python_executable or sys.executable

    with tempfile.TemporaryDirectory(prefix="kb_gpu_worker_") as tmpdir:
        cand_path = Path(tmpdir) / "candidate_kernel.py"
        cand_path.write_text(source_code, encoding="utf-8")

        # Use -s -E with explicit absolute script path + --repo-root (avoids -I dropping PYTHONPATH/cwd)
        cmd = [
            py_exe,
            "-s",
            "-E",
            str(worker_script),
            "--repo-root",
            str(root),
            "--op-name",
            op_name,
            "--candidate-path",
            str(cand_path),
            "--dtype",
            float_dtype,
            "--seed",
            str(seed),
            "--independent-seed",
            str(indep_seed),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--run-id",
            run_id,
            "--method",
            method,
        ]
        if not stress_test:
            cmd.append("--no-stress")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_secs,
            )
        except subprocess.TimeoutExpired:
            return _build_supervisor_fallback(
                run_id=run_id,
                op_name=op_name,
                source_code=source_code,
                status="TIMEOUT",
                err_msg=f"GPU worker exceeded hard wall-clock timeout ({timeout_secs}s).",
                seed=seed,
                independent_seed=indep_seed,
                dtype=float_dtype,
                method=method,
            )
        except Exception as exc:
            return _build_supervisor_fallback(
                run_id=run_id,
                op_name=op_name,
                source_code=source_code,
                status="RUNTIME_ERROR",
                err_msg=f"Failed to launch sanitized GPU worker ({type(exc).__name__}): {exc}",
                seed=seed,
                independent_seed=indep_seed,
                dtype=float_dtype,
                method=method,
            )

        parsed = _parse_strict_worker_json(
            proc.stdout,
            expected_run_id=run_id,
            expected_op_name=op_name,
            expected_level=spec.level,
            returncode=proc.returncode,
        )
        if parsed is not None:
            parsed["candidate_source"] = source_code
            parsed["candidate_sha256"] = _sha256_text(source_code)
            parsed["seed"] = seed
            parsed["independent_seed"] = indep_seed
            parsed["dtype"] = float_dtype
            parsed["method"] = method
            parsed["timing_mode"] = "cuda_graph"
            return parsed

        if proc.returncode in (137, -9):
            status = "OOM"
            msg = f"GPU worker terminated with OOM/SIGKILL (exit={proc.returncode})."
        else:
            status = "RUNTIME_ERROR"
            stderr_tail = (proc.stderr or "").strip()[-400:]
            msg = f"GPU worker exited without valid contract JSON (exit={proc.returncode}): {stderr_tail}"

        return _build_supervisor_fallback(
            run_id=run_id,
            op_name=op_name,
            source_code=source_code,
            status=status,
            err_msg=msg,
            seed=seed,
            independent_seed=indep_seed,
            dtype=float_dtype,
            method=method,
        )


def grade_candidate(
    op_name: str,
    source_code: str,
    *,
    backend: str = "numpy_dev",
    float_dtype: str = "bf16",
    seed: int = 42,
    independent_seed: int | None = None,
    warmup: int = 10,
    iters: int = 50,
    timeout_secs: float = 45.0,
    run_id: str = "eval",
    method: str = "rl_candidate",
    stress_test: bool = True,
) -> dict[str, Any]:
    """Unified grading router preserving NumPy CPU mode and explicit Triton GPU mode."""
    indep_seed = (seed + 10_000) if independent_seed is None else independent_seed
    if backend == "triton":
        return verify_candidate_gpu(
            op_name=op_name,
            source_code=source_code,
            float_dtype=float_dtype,
            seed=seed,
            independent_seed=indep_seed,
            warmup=warmup,
            iters=iters,
            timeout_secs=timeout_secs,
            run_id=run_id,
            method=method,
            stress_test=stress_test,
        )
    if backend == "numpy_dev":
        res: KernelEvalResult = verify_candidate_source(
            op_name=op_name,
            source_code=source_code,
            run_id=run_id,
            seed=seed,
            stress_test=stress_test,
        )
        payload = res.to_dict()
        for k, v in list(payload.items()):
            if isinstance(v, float) and not math.isfinite(v):
                payload[k] = 0.0
        payload["backend"] = "numpy_dev"
        payload["seed"] = int(seed)
        payload["independent_seed"] = int(indep_seed)
        payload["dtype"] = "fp32"
        payload["method"] = method
        payload["gpu_model"] = "cpu_numpy_dev"
        payload["isolation_boundary"] = "in_process_cpu_dev_only"
        payload["candidate_source"] = source_code
        payload["candidate_sha256"] = _sha256_text(source_code)
        return payload
    raise ValueError(f"Unsupported backend {backend!r}. Expected 'numpy_dev' or 'triton'.")
