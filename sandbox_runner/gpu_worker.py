"""Sanitized child-process GPU kernel grading worker for KernelBench-RL.

Can be invoked directly via `python -s -E path/to/gpu_worker.py --repo-root <abs>`
so isolated execution never depends on `PYTHONPATH` or `cwd`, while loading
candidate kernels from a real on-disk `.py` file via `importlib` so `@triton.jit`
can inspect source cleanly.

Security boundary:
AST filtering + sanitized child process + wall-clock timeout is a development /
trusted-candidate boundary, NOT an OS/container security sandbox.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


def _bootstrap_repo_root(argv: list[str] | None = None) -> None:
    """Insert explicit `--repo-root` onto sys.path[0] before importing local packages."""
    args_list = sys.argv[1:] if argv is None else argv
    for idx, tok in enumerate(args_list):
        if tok == "--repo-root" and idx + 1 < len(args_list):
            root = str(Path(args_list[idx + 1]).resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return
        if tok.startswith("--repo-root="):
            root = str(Path(tok.split("=", 1)[1]).resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    fallback = str(Path(__file__).resolve().parents[1])
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


_bootstrap_repo_root()

import numpy as np  # noqa: E402

from contracts.schemas import KernelEvalResult  # noqa: E402
from kernels.op_registry import OP_REGISTRY  # noqa: E402
from kernels.torch_references import TORCH_REFERENCES, convert_numpy_inputs_to_torch  # noqa: E402
from kernels.validate_cuda import (  # noqa: E402
    _finite_float,
    _make_error_record,
    _time_cuda_graph,
    benchmark_compiled_reference,
    check_cuda_triton_environment,
    classify_cuda_exception,
    compute_quantized_numpy_reference,
    serialize_eval_record,
    tolerances_for_dtype,
)
from rl_trainer.grpo_reward import compute_kernel_reward  # noqa: E402
from sandbox_runner.anti_cheat import analyze_candidate_source  # noqa: E402
from sandbox_runner.gpu_source_policy import check_gpu_source_policy  # noqa: E402

ISOLATION_BOUNDARY = "sanitized_child_process_trusted_only"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_candidate_from_file(candidate_path: Path, module_name: str = "kb_candidate_kernel") -> Callable[..., Any]:
    """Load `kernel_forward` from a real `.py` file so `@triton.jit` inspect.getsource works."""
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(candidate_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {candidate_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "kernel_forward", None)
    if fn is None or not callable(fn):
        raise KeyError("Candidate file must expose a callable `kernel_forward(...)`.")
    return fn


def _emit_strict_json(
    record: KernelEvalResult,
    *,
    source_code: str,
    seed: int,
    independent_seed: int,
    dtype: str,
    gpu_model: str,
    method: str,
) -> str:
    raw = serialize_eval_record(
        record,
        seed=seed,
        dtype=dtype,
        gpu_model=gpu_model,
        method=method,
        timing_mode="cuda_graph",
    )
    payload = json.loads(raw)
    payload["independent_seed"] = int(independent_seed)
    payload["isolation_boundary"] = ISOLATION_BOUNDARY
    payload["candidate_source"] = source_code
    payload["candidate_sha256"] = _sha256_text(source_code)
    payload["source_policy_sha256"] = hashlib.sha256(Path(__file__).with_name("gpu_source_policy.py").read_bytes()).hexdigest()
    return json.dumps(payload, allow_nan=False)


def _check_gpu_input_mutation(orig_tensors: tuple[Any, ...], after_tensors: tuple[Any, ...], torch_mod: Any) -> bool:
    for orig, after in zip(orig_tensors, after_tensors):
        if not bool(torch_mod.equal(orig, after)):
            return True
    return False


def _convert_inputs_with_fp16(np_in: tuple[np.ndarray, ...], float_dtype: str, torch_mod: Any) -> tuple[Any, ...]:
    if float_dtype in {"fp32", "bf16"}:
        return convert_numpy_inputs_to_torch(np_in, device="cuda", float_dtype=float_dtype)
    if float_dtype == "fp16":
        tensors = []
        for arr in np_in:
            t = torch_mod.from_numpy(np.asarray(arr, order="C"))
            if np.issubdtype(arr.dtype, np.floating):
                t = t.to(device="cuda", dtype=torch_mod.float16)
            else:
                t = t.to(device="cuda")
            tensors.append(t.contiguous())
        return tuple(tensors)
    raise ValueError(f"Unsupported float_dtype '{float_dtype}'.")


def _tolerances_with_fp16(spec: Any, float_dtype: str) -> tuple[float, float]:
    if float_dtype in {"fp32", "bf16"}:
        return tolerances_for_dtype(spec, float_dtype)
    if float_dtype == "fp16":
        return max(spec.rtol, 1e-2), max(spec.atol, 1e-2)
    raise ValueError(f"Unsupported float_dtype '{float_dtype}'.")


def grade_candidate_in_worker(
    op_name: str,
    source_code: str,
    *,
    candidate_path: Path | None = None,
    float_dtype: str = "bf16",
    seed: int = 42,
    independent_seed: int | None = None,
    warmup: int = 10,
    iters: int = 50,
    run_id: str = "gpu-worker",
    method: str = "rl_candidate",
    stress_test: bool = True,
) -> tuple[KernelEvalResult, str]:
    """Grade one candidate Triton kernel from a real `.py` file against PyTorch & NumPy refs."""
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unknown operator '{op_name}'. Available: {sorted(OP_REGISTRY)}")
    if float_dtype not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported float_dtype '{float_dtype}'.")
    spec = OP_REGISTRY[op_name]
    indep_seed = (seed + 10_000) if independent_seed is None else independent_seed

    cheat = analyze_candidate_source(source_code)
    try:
        ast.parse(source_code)
    except SyntaxError as exc:
        return _make_error_record(run_id, spec, "COMPILE_ERROR", False, str(exc)), "unavailable"
    policy = check_gpu_source_policy(source_code)
    if not policy.allowed and not cheat.cheat_detected:
        from sandbox_runner.anti_cheat import CheatReport
        cheat = CheatReport(True, "GPU source policy: " + "; ".join(policy.violations[:5]))
    if cheat.cheat_detected:
        rec = KernelEvalResult(
            run_id=run_id,
            op_name=spec.name,
            level=spec.level,
            backend="triton",
            status="CHEAT",
            compiled=False,
            correct=False,
            cheat_detected=True,
            max_abs_err=0.0,
            eager_ms=0.0,
            compile_ms=0.0,
            candidate_ms=0.0,
            speedup_vs_eager=0.0,
            speedup_vs_compile=0.0,
            achieved_tflops=0.0,
            arithmetic_intensity=_finite_float(spec.arithmetic_intensity),
            reward=-1.0,
            error_message=cheat.reason,
        )
        return rec, "unavailable"

    ok, gpu_or_err = check_cuda_triton_environment()
    if not ok:
        rec = _make_error_record(run_id, spec, "RUNTIME_ERROR", False, gpu_or_err, max_abs_err=0.0)
        return rec, "unavailable"

    if op_name not in TORCH_REFERENCES:
        rec = _make_error_record(
            run_id,
            spec,
            "RUNTIME_ERROR",
            False,
            f"No native PyTorch reference registered for '{op_name}' in kernels.torch_references.",
            max_abs_err=0.0,
        )
        return rec, gpu_or_err

    import torch  # type: ignore

    tmp_ctx = tempfile.TemporaryDirectory(prefix="kb_cand_src_") if candidate_path is None else None
    try:
        if candidate_path is None:
            assert tmp_ctx is not None
            real_path = Path(tmp_ctx.name) / f"cand_{op_name}.py"
            real_path.write_text(source_code, encoding="utf-8")
        else:
            real_path = candidate_path

        try:
            cand_fn = load_candidate_from_file(real_path, module_name=f"kb_cand_{op_name}_{abs(hash(run_id)) % 100000}")
        except Exception as exc:
            rec = _make_error_record(
                run_id,
                spec,
                "COMPILE_ERROR",
                False,
                f"Candidate file load error ({type(exc).__name__}): {exc}",
                max_abs_err=0.0,
            )
            return rec, gpu_or_err

        torch_ref_fn = TORCH_REFERENCES[op_name]
        rtol, atol = _tolerances_with_fp16(spec, float_dtype)

        np_cases: list[tuple[str, tuple[np.ndarray, ...]]] = [
            ("canonical", spec.make_inputs(np.random.default_rng(seed))),
        ]
        if stress_test and spec.make_stress_inputs is not None:
            for s_idx, s_in in enumerate(spec.make_stress_inputs(np.random.default_rng(seed + 1))):
                np_cases.append((f"stress_{s_idx}", s_in))
        np_cases.append(("independent_seed", spec.make_inputs(np.random.default_rng(indep_seed))))

        worst_err = 0.0
        compiled_ok = False
        canonical_gpu_inputs: tuple[Any, ...] | None = None
        canonical_eager_np: np.ndarray | None = None

        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        if float_dtype == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False

        try:
            for case_label, np_in in np_cases:
                gpu_in = _convert_inputs_with_fp16(np_in, float_dtype, torch)
                np_ref = compute_quantized_numpy_reference(spec, gpu_in)

                try:
                    with torch.no_grad():
                        torch_ref_out = torch_ref_fn(*gpu_in)
                    torch.cuda.synchronize()
                except Exception as exc:
                    status, _ = classify_cuda_exception(exc, compiled_so_far=compiled_ok)
                    rec = _make_error_record(
                        run_id,
                        spec,
                        status,
                        compiled_ok,
                        f"{case_label} PyTorch ref error ({type(exc).__name__}): {exc}",
                        max_abs_err=0.0,
                    )
                    return rec, gpu_or_err

                cand_in = tuple(t.clone() for t in gpu_in)
                try:
                    with torch.no_grad():
                        cand_out = cand_fn(*cand_in)
                    torch.cuda.synchronize()
                    compiled_ok = True
                except Exception as exc:
                    status, comp_flag = classify_cuda_exception(exc, compiled_so_far=compiled_ok)
                    rec = _make_error_record(
                        run_id,
                        spec,
                        status,
                        comp_flag,
                        f"{case_label} candidate error ({type(exc).__name__}): {exc}",
                        max_abs_err=0.0,
                    )
                    return rec, gpu_or_err

                if _check_gpu_input_mutation(gpu_in, cand_in, torch):
                    rec = KernelEvalResult(
                        run_id=run_id,
                        op_name=spec.name,
                        level=spec.level,
                        backend="triton",
                        status="CHEAT",
                        compiled=True,
                        correct=False,
                        cheat_detected=True,
                        max_abs_err=0.0,
                        eager_ms=0.0,
                        compile_ms=0.0,
                        candidate_ms=0.0,
                        speedup_vs_eager=0.0,
                        speedup_vs_compile=0.0,
                        achieved_tflops=0.0,
                        arithmetic_intensity=_finite_float(spec.arithmetic_intensity),
                        reward=-1.0,
                        error_message=f"{case_label}: candidate mutated input tensor in-place.",
                    )
                    return rec, gpu_or_err

                if not hasattr(cand_out, "detach") or not torch.isfinite(cand_out).all():
                    rec = _make_error_record(
                        run_id,
                        spec,
                        "NUMERICAL_MISMATCH",
                        True,
                        f"{case_label}: candidate output is not a finite tensor.",
                        max_abs_err=0.0,
                    )
                    return rec, gpu_or_err

                cand_np = cand_out.detach().float().cpu().numpy()
                torch_ref_np = torch_ref_out.detach().float().cpu().numpy()
                if case_label == "canonical":
                    canonical_gpu_inputs = gpu_in
                    canonical_eager_np = torch_ref_np

                if cand_np.shape != np_ref.shape or cand_np.shape != torch_ref_np.shape:
                    rec = _make_error_record(
                        run_id,
                        spec,
                        "NUMERICAL_MISMATCH",
                        True,
                        f"{case_label}: shape mismatch cand={cand_np.shape} torch={torch_ref_np.shape} numpy={np_ref.shape}",
                        max_abs_err=0.0,
                    )
                    return rec, gpu_or_err

                err_torch = float(np.max(np.abs(cand_np - torch_ref_np)))
                err_numpy = float(np.max(np.abs(cand_np - np_ref)))
                worst_err = max(worst_err, err_torch, err_numpy)

                if not np.allclose(cand_np, torch_ref_np, rtol=rtol, atol=atol):
                    rec = _make_error_record(
                        run_id,
                        spec,
                        "NUMERICAL_MISMATCH",
                        True,
                        f"{case_label} ({float_dtype}) mismatch vs PyTorch ref: max_abs_err={err_torch:.3e} (rtol={rtol}, atol={atol})",
                        max_abs_err=worst_err,
                    )
                    return rec, gpu_or_err

                if not np.allclose(cand_np, np_ref, rtol=rtol, atol=atol):
                    rec = _make_error_record(
                        run_id,
                        spec,
                        "NUMERICAL_MISMATCH",
                        True,
                        f"{case_label} ({float_dtype}) mismatch vs quantized-input NumPy ref: max_abs_err={err_numpy:.3e} (rtol={rtol}, atol={atol})",
                        max_abs_err=worst_err,
                    )
                    return rec, gpu_or_err

            assert canonical_gpu_inputs is not None and canonical_eager_np is not None
            with torch.no_grad():
                eager_ms = _time_cuda_graph(torch, torch_ref_fn, canonical_gpu_inputs, warmup, iters)
                cand_ms = _time_cuda_graph(torch, cand_fn, canonical_gpu_inputs, warmup, iters)
                compile_ms, compile_note = benchmark_compiled_reference(
                    torch, torch_ref_fn, canonical_gpu_inputs, canonical_eager_np, rtol, atol, warmup, iters,
                    timer_fn=_time_cuda_graph,
                )

            speedup_eager = (eager_ms / cand_ms) if cand_ms > 0.0 else 0.0
            speedup_compile = (compile_ms / cand_ms) if (compile_ms > 0.0 and cand_ms > 0.0) else 0.0
            tflops = (spec.flops / (cand_ms * 1e-3)) / 1e12 if cand_ms > 0.0 else 0.0
            reward = compute_kernel_reward(True, True, False, speedup_eager, speedup_compile)

            rec = KernelEvalResult(
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
                arithmetic_intensity=_finite_float(spec.arithmetic_intensity),
                reward=_finite_float(reward),
                error_message=compile_note,
            )
            return rec, gpu_or_err
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    _bootstrap_repo_root(argv)
    parser = argparse.ArgumentParser(description="Sanitized child-process GPU kernel verifier worker.")
    parser.add_argument("--repo-root", default="", help="Absolute repository root for isolated sys.path bootstrap.")
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--independent-seed", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--run-id", default="gpu-worker")
    parser.add_argument("--method", default="rl_candidate")
    parser.add_argument("--no-stress", action="store_true")
    args = parser.parse_args(argv)

    indep_seed = (args.seed + 10_000) if args.independent_seed is None else args.independent_seed
    cand_path = Path(args.candidate_path).resolve()
    source = cand_path.read_text(encoding="utf-8")
    rec, gpu_model = grade_candidate_in_worker(
        op_name=args.op_name,
        source_code=source,
        candidate_path=cand_path,
        float_dtype=args.dtype,
        seed=args.seed,
        independent_seed=indep_seed,
        warmup=args.warmup,
        iters=args.iters,
        run_id=args.run_id,
        method=args.method,
        stress_test=not args.no_stress,
    )
    print(
        _emit_strict_json(
            rec,
            source_code=source,
            seed=args.seed,
            independent_seed=indep_seed,
            dtype=args.dtype,
            gpu_model=gpu_model,
            method=args.method,
        )
    )
    if gpu_model == "unavailable" and rec.status == "RUNTIME_ERROR":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
