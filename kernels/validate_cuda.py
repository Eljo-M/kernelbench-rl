"""CUDA validation CLI for KernelBench-RL Triton baselines.

Validates every baseline-covered operator in `kernels/triton_baselines.py`
against BOTH its native PyTorch reference (`kernels/torch_references.py`) and
its canonical NumPy reference (`kernels/op_registry.py`) recomputed from the
exact quantized GPU inputs (preserving integer tensors) across canonical and
stress inputs for fp32 and bf16.

Measures actual eager PyTorch, verified `torch.compile`, and Triton timings
with CUDA Events (no rounding short positive timings to zero) and emits strict
RFC-8259 JSONL (`allow_nan=False`) conforming to `KernelEvalResult` plus
grouping metadata (`seed`, `dtype`, `method`, `gpu_model`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import linecache
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from contracts.schemas import KernelEvalResult
from kernels.op_registry import OP_REGISTRY, OperatorSpec
from kernels.torch_references import (
    TORCH_REFERENCES,
    convert_numpy_inputs_to_torch,
    torch_inputs_to_numpy_fp32,
)
from kernels.triton_baselines import TRITON_BASELINES
from rl_trainer.grpo_reward import compute_kernel_reward


def check_cuda_triton_environment() -> tuple[bool, str]:
    """Verify that PyTorch with CUDA and OpenAI Triton are available."""
    try:
        import torch  # type: ignore
    except ImportError:
        return False, "Missing dependency: PyTorch is not installed."
    if not torch.cuda.is_available():
        return False, "No CUDA GPU detected (torch.cuda.is_available() is False). Refusing to emit CPU traces."
    try:
        import triton  # type: ignore  # noqa: F401
    except ImportError:
        return False, "Missing dependency: OpenAI Triton is not installed."
    return True, f"{torch.cuda.get_device_name(0)}"


def tolerances_for_dtype(spec: OperatorSpec, float_dtype: str) -> tuple[float, float]:
    """Use strict operator tolerances for fp32 and realistic bf16 tolerances for bfloat16."""
    if float_dtype == "fp32":
        return spec.rtol, spec.atol
    if float_dtype == "bf16":
        return max(spec.rtol, 2e-2), max(spec.atol, 2e-2)
    raise ValueError(f"Unsupported dtype: {float_dtype}")


def _finite_float(val: float, default: float = 0.0) -> float:
    return float(val) if math.isfinite(val) else default


def classify_cuda_exception(exc: BaseException, compiled_so_far: bool) -> tuple[str, bool]:
    """Classify Triton/CUDA exceptions accurately into COMPILE_ERROR, OOM, or RUNTIME_ERROR.

    Triton compiles lazily on first invocation, so `exec()` alone never establishes
    `compiled=True`.
    """
    cls_name = type(exc).__name__.lower()
    mod_name = (getattr(type(exc), "__module__", "") or "").lower()
    msg = str(exc).lower()

    if "outofmemory" in cls_name or "out of memory" in msg or "cuda error: out of memory" in msg:
        return "OOM", compiled_so_far

    compile_markers = (
        "compilationerror",
        "compiletimeassertion",
        "ptxas",
        "nvrtc",
        "triton.compiler",
        "codegenerror",
        "syntaxerror",
        "nameerror",
        "importerror",
        "jit",
    )
    if (
        isinstance(exc, (SyntaxError, NameError, ImportError))
        or any(m in cls_name for m in compile_markers)
        or any(m in mod_name for m in ("triton.compiler", "triton.runtime.jit", "torch._inductor"))
        or ("compilation" in msg and not compiled_so_far)
    ):
        return "COMPILE_ERROR", False

    return "RUNTIME_ERROR", compiled_so_far


def _make_error_record(
    run_id: str,
    spec: OperatorSpec,
    status: str,
    compiled: bool,
    err_msg: str,
    max_abs_err: float = 0.0,
) -> KernelEvalResult:
    """Create a strict-JSON-safe error record using finite sentinel 0.0 for unavailable errors."""
    safe_err = _finite_float(max_abs_err, default=0.0)
    return KernelEvalResult(
        run_id=run_id,
        op_name=spec.name,
        level=spec.level,
        backend="triton",
        status=status,  # type: ignore[arg-type]
        compiled=compiled,
        correct=False,
        cheat_detected=False,
        max_abs_err=safe_err,
        eager_ms=0.0,
        compile_ms=0.0,
        candidate_ms=0.0,
        speedup_vs_eager=0.0,
        speedup_vs_compile=0.0,
        achieved_tflops=0.0,
        arithmetic_intensity=_finite_float(spec.arithmetic_intensity),
        reward=_finite_float(compute_kernel_reward(compiled, False, False, 0.0, 0.0)),
        error_message=err_msg,
    )


def serialize_eval_record(
    record: KernelEvalResult,
    *,
    seed: int,
    dtype: str,
    gpu_model: str,
    method: str = "triton_baseline",
    timing_mode: str = "events",
) -> str:
    """Serialize KernelEvalResult + grouping metadata as strict RFC-8259 JSON (no NaN/Infinity)."""
    payload = record.to_dict()
    for k, v in list(payload.items()):
        if isinstance(v, float) and not math.isfinite(v):
            payload[k] = 0.0
    payload["seed"] = int(seed)
    payload["dtype"] = dtype
    payload["method"] = method
    payload["gpu_model"] = gpu_model
    payload["timing_mode"] = timing_mode
    source = TRITON_BASELINES.get(record.op_name)
    if source and method == "triton_baseline":
        payload["source"] = source
        payload["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return json.dumps(payload, allow_nan=False)


def _time_cuda_events(torch_mod: Any, fn: Callable, inputs: tuple, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*inputs)
    torch_mod.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = torch_mod.cuda.Event(enable_timing=True)
        end = torch_mod.cuda.Event(enable_timing=True)
        start.record()
        fn(*inputs)
        end.record()
        torch_mod.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    median = float(np.median(np.asarray(times_ms, dtype=np.float64)))
    if not math.isfinite(median) or median <= 0:
        raise ValueError("CUDA event median must be finite and positive")
    return median


def _time_cuda_graph(torch_mod: Any, fn: Callable, inputs: tuple, warmup: int, iters: int) -> float:
    """Amortize host dispatch by timing 16 captured calls per graph replay.

    This reports steady-state device execution, not end-to-end Python latency.
    Every implementation is captured and measured with the same procedure.
    """
    stream = torch_mod.cuda.Stream()
    stream.wait_stream(torch_mod.cuda.current_stream())
    with torch_mod.cuda.stream(stream):
        for _ in range(max(warmup, 3)):
            fn(*inputs)
    torch_mod.cuda.current_stream().wait_stream(stream)
    torch_mod.cuda.synchronize()
    graph = torch_mod.cuda.CUDAGraph()
    outputs = []
    with torch_mod.cuda.graph(graph, stream=stream):
        for _ in range(16):
            outputs.append(fn(*inputs))
    graph.replay()
    torch_mod.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch_mod.cuda.Event(enable_timing=True)
        end = torch_mod.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / 16)
    median = float(np.median(samples))
    if not math.isfinite(median) or median <= 0:
        raise ValueError("CUDA graph median must be finite and positive")
    return median


def benchmark_compiled_reference(
    torch_mod: Any,
    torch_ref_fn: Callable,
    canonical_gpu_inputs: tuple[Any, ...],
    eager_ref_np: np.ndarray,
    rtol: float,
    atol: float,
    warmup: int,
    iters: int,
    timer_fn: Callable[[Any, Callable, tuple, int, int], float] = _time_cuda_events,
) -> tuple[float, str]:
    """Compile `torch_ref_fn`, verify its output matches eager reference, then measure raw median ms.

    Returns `(0.0, error_message)` if compilation fails or produces numerically incorrect output.
    """
    try:
        compiled_fn = torch_mod.compile(torch_ref_fn)
        compiled_out = compiled_fn(*canonical_gpu_inputs)
        torch_mod.cuda.synchronize()
    except Exception as exc:
        return 0.0, f"torch.compile failed ({type(exc).__name__}): {exc}"

    try:
        comp_np = compiled_out.detach().float().cpu().numpy()
    except Exception as exc:
        return 0.0, f"torch.compile output conversion failed ({type(exc).__name__}): {exc}"

    if comp_np.shape != eager_ref_np.shape or not np.all(np.isfinite(comp_np)):
        return 0.0, f"torch.compile output invalid (shape={comp_np.shape}, expected={eager_ref_np.shape})"

    err = float(np.max(np.abs(comp_np - eager_ref_np)))
    if not np.allclose(comp_np, eager_ref_np, rtol=rtol, atol=atol):
        return (
            0.0,
            f"torch.compile numerical mismatch vs eager ref (max_abs_err={err:.3e}, rtol={rtol}, atol={atol})",
        )

    try:
        ms = timer_fn(torch_mod, compiled_fn, canonical_gpu_inputs, warmup, iters)
    except Exception as exc:
        return 0.0, f"torch.compile timing failed ({type(exc).__name__}): {exc}"

    if not math.isfinite(ms) or ms <= 0.0:
        return 0.0, f"torch.compile returned non-positive timing ({ms})"
    return float(ms), ""


def compute_quantized_numpy_reference(
    spec: OperatorSpec,
    gpu_inputs: Sequence[Any],
) -> np.ndarray:
    """Recompute the NumPy reference from the exact quantized GPU inputs converted to fp32/int."""
    quantized_np_inputs = torch_inputs_to_numpy_fp32(gpu_inputs)
    return spec.reference_fn(*quantized_np_inputs)


def validate_cli_args(
    ops: Sequence[str],
    seed: int,
    warmup: int,
    iters: int,
    out_path: Path,
    overwrite: bool,
) -> tuple[bool, str]:
    """Validate CLI arguments before touching CUDA or creating any output files."""
    if not ops:
        return False, "--ops must specify at least one operator."
    unknown = [op for op in ops if op not in OP_REGISTRY or op not in TRITON_BASELINES]
    if unknown:
        return (
            False,
            f"Unknown or non-baseline operator(s): {unknown}. Valid baseline ops: {sorted(TRITON_BASELINES)}",
        )
    if seed < 0:
        return False, f"--seed must be >= 0, got {seed}."
    if warmup < 0:
        return False, f"--warmup must be >= 0, got {warmup}."
    if iters <= 0:
        return False, f"--iters must be > 0, got {iters}."
    if out_path.exists() and not overwrite:
        return False, f"Output file already exists: {out_path}. Pass --overwrite to replace it."
    return True, ""


def validate_single_baseline_cuda(
    op_name: str,
    float_dtype: str = "fp32",
    seed: int = 42,
    warmup: int = 10,
    iters: int = 50,
    run_prefix: str = "cuda-val",
    timing_mode: str = "events",
) -> KernelEvalResult:
    """Validate one Triton baseline on CUDA against PyTorch and quantized-input NumPy references."""
    ok, env_msg = check_cuda_triton_environment()
    if not ok:
        raise RuntimeError(env_msg)

    import torch  # type: ignore

    if timing_mode not in {"events", "cuda_graph"}:
        raise ValueError("Unknown timing mode")
    timer_fn = _time_cuda_graph if timing_mode == "cuda_graph" else _time_cuda_events

    spec = OP_REGISTRY[op_name]
    run_id = f"{run_prefix}-{op_name}-{float_dtype}"
    if op_name not in TRITON_BASELINES:
        return _make_error_record(run_id, spec, "COMPILE_ERROR", False, f"No Triton baseline registered for {op_name}")
    if op_name not in TORCH_REFERENCES:
        return _make_error_record(run_id, spec, "RUNTIME_ERROR", False, f"No PyTorch reference registered for {op_name}")

    # Load candidate module (does NOT establish compiled=True since Triton JITs on first call)
    ns: dict[str, Any] = {}
    try:
        source = TRITON_BASELINES[op_name] + "\n"
        filename = f"<triton_baseline:{op_name}>"
        # Triton inspects the decorated Python function to obtain its source.
        # Keep generated baseline source available to inspect.getsource.
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        ns["__name__"] = f"kernelbench_baseline_{op_name}"
        exec(compile(source, filename, "exec", dont_inherit=True), ns)
        cand_fn = ns["kernel_forward"]
    except Exception as exc:
        return _make_error_record(run_id, spec, "COMPILE_ERROR", False, f"Baseline load error ({type(exc).__name__}): {exc}")

    torch_ref_fn = TORCH_REFERENCES[op_name]
    rtol, atol = tolerances_for_dtype(spec, float_dtype)

    rng = np.random.default_rng(seed)
    np_cases: list[tuple[np.ndarray, ...]] = [spec.make_inputs(rng)]
    if spec.make_stress_inputs is not None:
        np_cases.extend(spec.make_stress_inputs(np.random.default_rng(seed + 1)))

    worst_err = 0.0
    compiled_ok = False
    canonical_gpu_inputs: tuple[Any, ...] | None = None
    canonical_eager_np: np.ndarray | None = None
    canonical_bytes = 0

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if float_dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        for idx, np_in in enumerate(np_cases):
            gpu_in = convert_numpy_inputs_to_torch(np_in, device="cuda", float_dtype=float_dtype)
            # Recompute NumPy reference from exact quantized GPU input values (integer inputs preserved)
            np_ref = compute_quantized_numpy_reference(spec, gpu_in)

            try:
                with torch.no_grad():
                    torch_ref_out = torch_ref_fn(*gpu_in)
                torch.cuda.synchronize()
            except Exception as exc:
                status, _ = classify_cuda_exception(exc, compiled_so_far=compiled_ok)
                return _make_error_record(
                    run_id, spec, status, compiled_ok, f"Case {idx} PyTorch ref error ({type(exc).__name__}): {exc}"
                )

            try:
                with torch.no_grad():
                    cand_out = cand_fn(*gpu_in)
                torch.cuda.synchronize()
                compiled_ok = True
            except Exception as exc:
                status, comp_flag = classify_cuda_exception(exc, compiled_so_far=compiled_ok)
                return _make_error_record(
                    run_id, spec, status, comp_flag, f"Case {idx} candidate error ({type(exc).__name__}): {exc}"
                )

            if not torch.isfinite(cand_out).all():
                return _make_error_record(run_id, spec, "NUMERICAL_MISMATCH", True, f"Case {idx}: non-finite CUDA output")

            cand_np = cand_out.detach().float().cpu().numpy()
            torch_ref_np = torch_ref_out.detach().float().cpu().numpy()
            if idx == 0:
                canonical_gpu_inputs = gpu_in
                canonical_eager_np = torch_ref_np
                canonical_bytes = sum(t.numel() * t.element_size() for t in gpu_in) + cand_out.numel() * cand_out.element_size()

            if cand_np.shape != np_ref.shape or cand_np.shape != torch_ref_np.shape:
                return _make_error_record(
                    run_id,
                    spec,
                    "NUMERICAL_MISMATCH",
                    True,
                    f"Case {idx}: shape mismatch cand={cand_np.shape} torch={torch_ref_np.shape} numpy={np_ref.shape}",
                )

            err_vs_torch = float(np.max(np.abs(cand_np - torch_ref_np)))
            err_vs_numpy = float(np.max(np.abs(cand_np - np_ref)))
            worst_err = max(worst_err, err_vs_torch, err_vs_numpy)

            if not np.allclose(cand_np, torch_ref_np, rtol=rtol, atol=atol):
                return _make_error_record(
                    run_id,
                    spec,
                    "NUMERICAL_MISMATCH",
                    True,
                    f"Case {idx} ({float_dtype}) mismatch vs PyTorch ref: max_abs_err={err_vs_torch:.3e} (rtol={rtol}, atol={atol})",
                    max_abs_err=worst_err,
                )
            if not np.allclose(cand_np, np_ref, rtol=rtol, atol=atol):
                return _make_error_record(
                    run_id,
                    spec,
                    "NUMERICAL_MISMATCH",
                    True,
                    f"Case {idx} ({float_dtype}) mismatch vs quantized-input NumPy ref: max_abs_err={err_vs_numpy:.3e} (rtol={rtol}, atol={atol})",
                    max_abs_err=worst_err,
                )

        assert canonical_gpu_inputs is not None and canonical_eager_np is not None
        with torch.no_grad():
            eager_ms = timer_fn(torch, torch_ref_fn, canonical_gpu_inputs, warmup, iters)
            cand_ms = timer_fn(torch, cand_fn, canonical_gpu_inputs, warmup, iters)
            compile_ms, compile_note = benchmark_compiled_reference(
                torch, torch_ref_fn, canonical_gpu_inputs, canonical_eager_np, rtol, atol, warmup, iters, timer_fn=timer_fn
            )

        speedup_eager = eager_ms / cand_ms if cand_ms > 0.0 else 0.0
        speedup_compile = (compile_ms / cand_ms) if (compile_ms > 0.0 and cand_ms > 0.0) else 0.0
        tflops = (spec.flops / (cand_ms * 1e-3)) / 1e12 if cand_ms > 0.0 else 0.0
        reward = compute_kernel_reward(True, True, False, speedup_eager, speedup_compile)

        return KernelEvalResult(
            run_id=run_id,
            op_name=spec.name,
            level=spec.level,
            backend="triton",
            status="PASS",
            compiled=True,
            correct=True,
            cheat_detected=False,
            max_abs_err=_finite_float(worst_err),
            eager_ms=float(eager_ms),
            compile_ms=float(compile_ms),
            candidate_ms=float(cand_ms),
            speedup_vs_eager=_finite_float(speedup_eager),
            speedup_vs_compile=_finite_float(speedup_compile),
            achieved_tflops=_finite_float(tflops),
            arithmetic_intensity=spec.flops / max(canonical_bytes, 1),
            reward=_finite_float(reward),
            error_message=compile_note,
        )
    except Exception as exc:
        status, compiled_flag = classify_cuda_exception(exc, compiled_so_far=compiled_ok)
        return _make_error_record(run_id, spec, status, compiled_flag,
                                  f"Validation/timing error ({type(exc).__name__}): {exc}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate KernelBench-RL Triton baselines on CUDA GPUs.")
    parser.add_argument(
        "--ops",
        nargs="+",
        default=sorted(TRITON_BASELINES.keys()),
        help="Subset of baseline operator names to validate (defaults to all 11).",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["fp32", "bf16"],
        default=["fp32", "bf16"],
        help="Floating-point dtypes to validate (integer inputs stay integer).",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path for strict KernelEvalResult records.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing --out file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--timing-mode", choices=["events", "cuda_graph"], default="cuda_graph",
                        help="CUDA graphs amortize Python dispatch; events include host dispatch gaps")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    valid_args, arg_err = validate_cli_args(
        ops=args.ops,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
        out_path=out_path,
        overwrite=args.overwrite,
    )
    if not valid_args:
        print(f"[ERROR] {arg_err}", file=sys.stderr)
        return 2

    ok, gpu_or_err = check_cuda_triton_environment()
    if not ok:
        print(f"[ERROR] {gpu_or_err}", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[KernelEvalResult] = []
    any_failed = False

    with out_path.open("w", encoding="utf-8") as fh:
        for op_name in args.ops:
            for dtype in args.dtypes:
                res = validate_single_baseline_cuda(
                    op_name=op_name,
                    float_dtype=dtype,
                    seed=args.seed,
                    warmup=args.warmup,
                    iters=args.iters,
                    timing_mode=args.timing_mode,
                )
                records.append(res)
                line = serialize_eval_record(
                    res,
                    seed=args.seed,
                    dtype=dtype,
                    gpu_model=gpu_or_err,
                    method="triton_baseline",
                    timing_mode=args.timing_mode,
                )
                fh.write(line + "\n")
                fh.flush()
                if res.status != "PASS":
                    any_failed = True
                print(
                    f"[{res.status}] {op_name:<30} dtype={dtype} "
                    f"eager={res.eager_ms:.6f}ms compile={res.compile_ms:.6f}ms "
                    f"triton={res.candidate_ms:.6f}ms max_err={res.max_abs_err:.2e}"
                )

    passed = sum(1 for r in records if r.status == "PASS")
    print(f"\nCUDA validation complete: {passed}/{len(records)} passed. JSONL written to {out_path}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
