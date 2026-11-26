"""Strict held-out evaluation comparator and report generator for KernelBench-RL.

Compares ``base_pass1``, ``base_best_of_<K>`` (default ``base_best_of_8``), and
``trained_checkpoint`` strictly *within* each run directory while preserving model,
precision, and GPU separation *across* runs.

Each ``--run-dir`` must contain:
- ``config.json`` (ExperimentConfig JSON)
- ``heldout.jsonl`` (raw selection + held-out test records)
- ``heldout_environment.json`` (model_id, revision, dtype, gpu, config_hash,
  source_commit, checkpoint_weights_sha)

Outputs written to ``--out``:
- ``comparison.md``: readable Markdown summary with per-run tables, paired per-op
  statuses, and explicit methodological caveats.
- ``comparison.json``: strict finite JSON (no NaN/Infinity) preserving all per-run
  metrics, failure counts, excluded row counts, and paired per-op statuses.
- ``comparison.png``: static matplotlib chart showing pass count / denominator per
  method in separate panels per run (never pooled across hardware or precision).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_trainer.experiment_config import ExperimentConfig

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID_STATUSES: frozenset[str] = frozenset(
    {
        "PASS",
        "COMPILE_ERROR",
        "NUMERICAL_MISMATCH",
        "TIMEOUT",
        "OOM",
        "RUNTIME_ERROR",
        "CHEAT",
    }
)

FAILURE_STATUSES: tuple[str, ...] = (
    "COMPILE_ERROR",
    "NUMERICAL_MISMATCH",
    "TIMEOUT",
    "OOM",
    "RUNTIME_ERROR",
    "CHEAT",
)

TRAINING_MODE_LABELS: dict[str, str] = {
    "full_param_0.49b": "Full-Parameter 0.49B",
    "qlora_7b_nf4": "7B QLoRA (NF4)",
}


class HeldoutValidationError(ValueError):
    """Raised when a held-out run directory violates integrity or comparability invariants."""


def _reject_non_finite_json(token: str) -> None:
    raise HeldoutValidationError(f"Non-finite JSON constant rejected: {token}")


def _load_strict_json(path: Path) -> Any:
    if not path.is_file():
        raise HeldoutValidationError(f"Required file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite_json)
    except json.JSONDecodeError as exc:
        raise HeldoutValidationError(f"Invalid JSON in {path}: {exc}") from exc


def _load_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise HeldoutValidationError(f"Required file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped, parse_constant=_reject_non_finite_json)
        except json.JSONDecodeError as exc:
            raise HeldoutValidationError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise HeldoutValidationError(f"Expected JSON object at {path}:{line_no}, got {type(obj).__name__}.")
        rows.append(obj)
    return rows


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str], *, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _require_non_empty_str(mapping: Mapping[str, Any], keys: Sequence[str], *, context: str) -> str:
    val = _first_present(mapping, keys)
    if not isinstance(val, str) or not val.strip():
        raise HeldoutValidationError(f"Missing or empty required string field {list(keys)} in {context}.")
    return val.strip()


def _require_int(mapping: Mapping[str, Any], key: str, *, context: str) -> int:
    if key not in mapping:
        raise HeldoutValidationError(f"Missing required integer field '{key}' in {context}.")
    val = mapping[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise HeldoutValidationError(f"Field '{key}' in {context} must be an int, got {type(val).__name__}.")
    return int(val)


def _require_finite_float(mapping: Mapping[str, Any], key: str, *, context: str, default: float = 0.0) -> float:
    val = mapping.get(key, default)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise HeldoutValidationError(f"Field '{key}' in {context} must be numeric, got {type(val).__name__}.")
    fval = float(val)
    if not math.isfinite(fval):
        raise HeldoutValidationError(f"Field '{key}' in {context} must be finite, got {val!r}.")
    return fval


def _training_mode_label(training_mode: str) -> str:
    if training_mode not in TRAINING_MODE_LABELS:
        raise HeldoutValidationError(
            f"Unsupported training_mode '{training_mode}'. Expected one of {sorted(TRAINING_MODE_LABELS)}."
        )
    return TRAINING_MODE_LABELS[training_mode]


def _is_compile_available(row: Mapping[str, Any]) -> bool:
    compile_ms = float(row.get("compile_ms", 0.0))
    if "compile_available" in row:
        flag = row["compile_available"]
        if not isinstance(flag, bool):
            raise HeldoutValidationError(f"compile_available must be bool, got {type(flag).__name__}.")
        return flag and compile_ms > 0.0
    return compile_ms > 0.0


def _geomean_positive(values: Sequence[float]) -> float | None:
    pos = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    if not pos:
        return None
    return round(math.exp(sum(math.log(v) for v in pos) / len(pos)), 4)


@dataclass(frozen=True)
class RunEnvironment:
    run_name: str
    run_dir: str
    model_id: str
    model_name_or_path: str
    revision: str
    training_mode: str
    training_mode_label: str
    backend: str
    dtype: str
    gpu: str
    config_hash: str
    source_commit: str
    checkpoint_weights_sha: str
    heldout_ops: list[str]
    heldout_best_of_k: int
    seed: int
    heldout_seed_offset: int
    independent_seed_offset: int


@dataclass(frozen=True)
class MethodSummary:
    method: str
    search_budget_k: int
    selection_methods: list[str]
    total_ops: int
    compiled_count: int
    pass_count: int
    pass_rate_pct: float
    fast_vs_eager_count: int
    fast_vs_eager_rate_pct: float
    fast_vs_compile_count: int
    fast_vs_compile_denominator: int
    fast_vs_compile_rate_pct: float | None
    compile_available_passing_count: int
    compile_unavailable_passing_count: int
    compile_unavailable_total_count: int
    geomean_speedup_vs_eager_correct_only: float | None
    geomean_speedup_vs_eager_n: int
    geomean_speedup_vs_compile_correct_only: float | None
    geomean_speedup_vs_compile_n: int
    failure_count: int
    failure_counts_by_status: dict[str, int]
    mean_reward: float


def _parse_environment(run_dir: Path) -> tuple[dict[str, Any], RunEnvironment]:
    cfg = _load_strict_json(run_dir / "config.json")
    env_raw = _load_strict_json(run_dir / "heldout_environment.json")
    if not isinstance(cfg, dict):
        raise HeldoutValidationError(f"config.json in {run_dir} must be a JSON object.")
    if not isinstance(env_raw, dict):
        raise HeldoutValidationError(f"heldout_environment.json in {run_dir} must be a JSON object.")

    cfg_ctx = f"{run_dir / 'config.json'}"
    env_ctx = f"{run_dir / 'heldout_environment.json'}"

    cfg_model = _require_non_empty_str(cfg, ("model_name_or_path", "model_id"), context=cfg_ctx)
    cfg_rev = _require_non_empty_str(cfg, ("model_revision", "revision"), context=cfg_ctx)
    cfg_mode = _require_non_empty_str(cfg, ("training_mode",), context=cfg_ctx)
    mode_label = _training_mode_label(cfg_mode)
    cfg_backend = _require_non_empty_str(cfg, ("backend",), context=cfg_ctx)
    cfg_dtype = _require_non_empty_str(cfg, ("dtype",), context=cfg_ctx)
    if cfg_dtype not in {"fp32", "bf16", "fp16"}:
        raise HeldoutValidationError(f"Unsupported dtype '{cfg_dtype}' in {cfg_ctx}.")

    heldout_ops = cfg.get("heldout_ops")
    if not isinstance(heldout_ops, list) or not heldout_ops or not all(isinstance(x, str) and x.strip() for x in heldout_ops):
        raise HeldoutValidationError(f"heldout_ops must be a non-empty list of operator names in {cfg_ctx}.")
    if len(set(heldout_ops)) != len(heldout_ops):
        raise HeldoutValidationError(f"Duplicate entries in heldout_ops in {cfg_ctx}.")
    train_ops = cfg.get("train_ops", [])
    if isinstance(train_ops, list):
        overlap = sorted(set(train_ops) & set(heldout_ops))
        if overlap:
            raise HeldoutValidationError(f"Data leakage in {cfg_ctx}: train_ops and heldout_ops overlap on {overlap}.")

    seed = _require_int(cfg, "seed", context=cfg_ctx)
    heldout_seed_offset = _require_int(cfg, "heldout_seed_offset", context=cfg_ctx) if "heldout_seed_offset" in cfg else 50_000
    independent_seed_offset = (
        _require_int(cfg, "independent_seed_offset", context=cfg_ctx) if "independent_seed_offset" in cfg else 10_000
    )
    best_k = _require_int(cfg, "heldout_best_of_k", context=cfg_ctx) if "heldout_best_of_k" in cfg else 8
    if best_k < 2:
        raise HeldoutValidationError(f"heldout_best_of_k must be >= 2 in {cfg_ctx}, got {best_k}.")

    env_model = _require_non_empty_str(env_raw, ("model_id", "model_name_or_path"), context=env_ctx)
    env_rev = _require_non_empty_str(env_raw, ("revision", "model_revision"), context=env_ctx)
    env_dtype = _require_non_empty_str(env_raw, ("dtype", "float_dtype"), context=env_ctx)
    env_gpu = _require_non_empty_str(env_raw, ("gpu", "gpu_model"), context=env_ctx)
    env_config_hash = _require_non_empty_str(env_raw, ("config_hash",), context=env_ctx)
    env_commit = _require_non_empty_str(env_raw, ("source_commit", "sourcecommit", "git_commit"), context=env_ctx)
    env_ckpt_sha = _require_non_empty_str(
        env_raw,
        ("checkpoint_weights_sha", "checkpoint_weights_sha256", "checkpointweightsSHA", "weights_sha256"),
        context=env_ctx,
    )

    # Local snapshots retain their public model identity in the manifest. Only
    # accept that alias when the recorded local path's final name matches; rows
    # still have to match the exact configured path and pinned revision.
    local_snapshot = cfg_model.startswith(("/", "./", "../", "\\")) or (len(cfg_model) > 2 and cfg_model[1] == ":")
    same_snapshot_name = cfg_model.replace("\\", "/").rstrip("/").split("/")[-1] == env_model.split("/")[-1]
    if env_model != cfg_model and not (local_snapshot and same_snapshot_name):
        raise HeldoutValidationError(f"Model mismatch in {run_dir}: config='{cfg_model}' vs env='{env_model}'.")
    if env_rev != cfg_rev:
        raise HeldoutValidationError(f"Revision mismatch in {run_dir}: config='{cfg_rev}' vs env='{env_rev}'.")
    if env_dtype != cfg_dtype:
        raise HeldoutValidationError(f"Dtype mismatch in {run_dir}: config='{cfg_dtype}' vs env='{env_dtype}'.")
    if "training_mode" in env_raw and env_raw["training_mode"] is not None:
        if str(env_raw["training_mode"]) != cfg_mode:
            raise HeldoutValidationError(
                f"training_mode mismatch in {run_dir}: config='{cfg_mode}' vs env='{env_raw['training_mode']}'."
            )
    if "config_hash" in cfg and cfg["config_hash"] is not None:
        if str(cfg["config_hash"]) != env_config_hash:
            raise HeldoutValidationError(
                f"config_hash mismatch in {run_dir}: config='{cfg['config_hash']}' vs env='{env_config_hash}'."
            )
    try:
        parsed_config = ExperimentConfig(**{k: v for k, v in cfg.items() if k != "config_hash"})
        parsed_config.validate()
    except (TypeError, ValueError) as exc:
        raise HeldoutValidationError(f"Invalid experiment config in {cfg_ctx}: {exc}") from exc
    if parsed_config.config_hash() != env_config_hash:
        raise HeldoutValidationError(f"config_hash mismatch in {cfg_ctx}: recomputed config differs from manifest.")
    if cfg_backend == "triton" and env_gpu.lower() == "unavailable":
        raise HeldoutValidationError(f"heldout_environment.json in {run_dir} must specify a real GPU for backend='triton'.")

    run_name = str(env_raw.get("run_name") or run_dir.name)
    env = RunEnvironment(
        run_name=run_name,
        run_dir=str(run_dir),
        model_id=env_model,
        model_name_or_path=cfg_model,
        revision=env_rev,
        training_mode=cfg_mode,
        training_mode_label=mode_label,
        backend=cfg_backend,
        dtype=env_dtype,
        gpu=env_gpu,
        config_hash=env_config_hash,
        source_commit=env_commit,
        checkpoint_weights_sha=env_ckpt_sha,
        heldout_ops=list(heldout_ops),
        heldout_best_of_k=best_k,
        seed=seed,
        heldout_seed_offset=heldout_seed_offset,
        independent_seed_offset=independent_seed_offset,
    )
    return cfg, env


def _classify_exclusion(row: Mapping[str, Any]) -> str | None:
    """Return exclusion reason if a row must be excluded from final heldout_test metrics."""
    if bool(row.get("fixture", False)) or bool(row.get("is_fixture", False)):
        return "fixture"
    eval_phase = str(row.get("eval_phase", "")).strip()
    sel_method = str(row.get("selection_method", "")).strip()
    if "oracle" in eval_phase.lower() or "oracle" in sel_method.lower():
        return "oracle"
    if eval_phase == "selection_val":
        return "selection_val"
    if eval_phase != "heldout_test":
        return "other_eval_phase"
    return None


def load_and_validate_run(run_dir: Path) -> dict[str, Any]:
    """Load, filter, validate, and summarize one held-out run directory."""
    _, env = _parse_environment(run_dir)
    raw_rows = _load_strict_jsonl(run_dir / "heldout.jsonl")
    if not raw_rows:
        raise HeldoutValidationError(f"heldout.jsonl in {run_dir} contains no records.")

    best_method = f"base_best_of_{env.heldout_best_of_k}"
    canonical_methods = ("base_pass1", best_method, "trained_checkpoint")
    expected_seeds: dict[str, int] = {
        op: env.seed + env.heldout_seed_offset + idx for idx, op in enumerate(env.heldout_ops)
    }

    excluded_counts: dict[str, int] = {
        "selection_val": 0,
        "fixture": 0,
        "oracle": 0,
        "other_eval_phase": 0,
    }
    final_rows: dict[tuple[str, str], dict[str, Any]] = {}
    seen_run_ids: set[str] = set()

    for idx, row in enumerate(raw_rows, start=1):
        excl = _classify_exclusion(row)
        if excl is not None:
            excluded_counts[excl] += 1
            continue

        ctx = f"{run_dir / 'heldout.jsonl'} row #{idx}"
        method = _require_non_empty_str(row, ("method",), context=ctx)
        if method not in canonical_methods:
            raise HeldoutValidationError(
                f"Unexpected method '{method}' in {ctx}; expected one of {list(canonical_methods)}."
            )

        op_name = _require_non_empty_str(row, ("op_name",), context=ctx)
        if op_name not in expected_seeds:
            raise HeldoutValidationError(
                f"Unexpected op_name '{op_name}' in {ctx}; configured heldout_ops={env.heldout_ops}."
            )

        key = (method, op_name)
        if key in final_rows:
            raise HeldoutValidationError(
                f"Duplicate final heldout_test row for method='{method}', op_name='{op_name}' in {run_dir}."
            )

        run_id = _require_non_empty_str(row, ("run_id",), context=ctx)
        if run_id in seen_run_ids:
            raise HeldoutValidationError(f"Duplicate run_id '{run_id}' among final rows in {run_dir}.")
        seen_run_ids.add(run_id)

        if bool(row.get("policy_updated", False)):
            raise HeldoutValidationError(f"Held-out evaluation row must have policy_updated=False in {ctx}.")

        row_hash = _require_non_empty_str(row, ("config_hash",), context=ctx)
        if row_hash != env.config_hash:
            raise HeldoutValidationError(
                f"config_hash mismatch in {ctx}: row='{row_hash}' vs manifest='{env.config_hash}'."
            )

        row_model = _require_non_empty_str(row, ("model_name_or_path", "model_id"), context=ctx)
        if row_model != env.model_name_or_path:
            raise HeldoutValidationError(f"Model mismatch in {ctx}: row='{row_model}' vs config='{env.model_name_or_path}'.")

        row_rev = _require_non_empty_str(row, ("model_revision", "revision"), context=ctx)
        if row_rev != env.revision:
            raise HeldoutValidationError(f"Revision mismatch in {ctx}: row='{row_rev}' vs manifest='{env.revision}'.")

        row_dtype = _require_non_empty_str(row, ("dtype", "float_dtype"), context=ctx)
        if row_dtype != env.dtype:
            raise HeldoutValidationError(f"Dtype mismatch in {ctx}: row='{row_dtype}' vs manifest='{env.dtype}'.")

        if "training_mode" in row and row["training_mode"] is not None:
            if str(row["training_mode"]) != env.training_mode:
                raise HeldoutValidationError(
                    f"training_mode mismatch in {ctx}: row='{row['training_mode']}' vs manifest='{env.training_mode}'."
                )

        if "backend" in row and row["backend"] is not None:
            if str(row["backend"]) != env.backend:
                raise HeldoutValidationError(
                    f"backend mismatch in {ctx}: row='{row['backend']}' vs manifest='{env.backend}'."
                )

        row_seed = _require_int(row, "seed", context=ctx)
        if row_seed != expected_seeds[op_name]:
            raise HeldoutValidationError(
                f"Seed mismatch for op '{op_name}' in {ctx}: got {row_seed}, expected {expected_seeds[op_name]}."
            )
        if "independent_seed" in row and row["independent_seed"] is not None:
            ind_seed = _require_int(row, "independent_seed", context=ctx)
            exp_ind = expected_seeds[op_name] + env.independent_seed_offset
            if ind_seed != exp_ind:
                raise HeldoutValidationError(
                    f"independent_seed mismatch for op '{op_name}' in {ctx}: got {ind_seed}, expected {exp_ind}."
                )

        status = _require_non_empty_str(row, ("status",), context=ctx)
        if status not in VALID_STATUSES:
            raise HeldoutValidationError(f"Invalid status '{status}' in {ctx}.")

        correct = row.get("correct")
        compiled = row.get("compiled")
        if not isinstance(correct, bool) or not isinstance(compiled, bool):
            raise HeldoutValidationError(f"Fields 'compiled' and 'correct' must be booleans in {ctx}.")
        if (status == "PASS") != correct:
            raise HeldoutValidationError(f"Inconsistent status='{status}' and correct={correct} in {ctx}.")

        # Validate finite numeric metrics
        _require_finite_float(row, "speedup_vs_eager", context=ctx)
        _require_finite_float(row, "speedup_vs_compile", context=ctx)
        _require_finite_float(row, "eager_ms", context=ctx)
        _require_finite_float(row, "compile_ms", context=ctx)
        _require_finite_float(row, "candidate_ms", context=ctx)
        _require_finite_float(row, "reward", context=ctx)
        if any(float(row.get(k, 0.0)) < 0 for k in ("eager_ms", "compile_ms", "candidate_ms", "speedup_vs_eager", "speedup_vs_compile")):
            raise HeldoutValidationError(f"Negative timing or speedup in {ctx}.")
        if status == "PASS":
            if not compiled or any(float(row.get(k, 0.0)) <= 0 for k in ("eager_ms", "candidate_ms", "speedup_vs_eager")):
                raise HeldoutValidationError(f"PASS requires compiled=True and positive verified timings in {ctx}.")
            for ref, ratio in (("eager_ms", "speedup_vs_eager"), ("compile_ms", "speedup_vs_compile")):
                if float(row.get(ref, 0)) > 0 and not math.isclose(float(row[ratio]), float(row[ref]) / float(row["candidate_ms"]), rel_tol=1e-4, abs_tol=1e-6):
                    raise HeldoutValidationError(f"Inconsistent {ratio} and latency ratio in {ctx}.")

        # GPU hardware consistency: failed rows may report gpu_model='unavailable',
        # but another real GPU mixed with manifest GPU is always rejected.
        row_gpu = str(_first_present(row, ("gpu_model", "gpu"), default="")).strip()
        if not row_gpu:
            raise HeldoutValidationError(f"Missing gpu_model in {ctx}.")
        if status == "PASS":
            if row_gpu != env.gpu:
                raise HeldoutValidationError(
                    f"GPU mismatch on PASS row in {ctx}: row gpu_model='{row_gpu}' vs manifest gpu='{env.gpu}'."
                )
        else:
            if row_gpu != env.gpu and row_gpu.lower() != "unavailable":
                raise HeldoutValidationError(
                    f"Mixed real GPU hardware in {ctx}: row gpu_model='{row_gpu}' vs manifest gpu='{env.gpu}'."
                )

        final_rows[key] = row

    # Require every configured heldout_op exactly once per canonical method
    missing: list[str] = []
    for method in canonical_methods:
        for op_name in env.heldout_ops:
            if (method, op_name) not in final_rows:
                missing.append(f"{method}/{op_name}")
    if missing:
        raise HeldoutValidationError(
            f"Incomplete heldout_test coverage in {run_dir}: missing {missing}."
        )

    # Summarize each method
    method_summaries: dict[str, MethodSummary] = {}
    n_ops = len(env.heldout_ops)
    for method in canonical_methods:
        m_rows = [final_rows[(method, op)] for op in env.heldout_ops]
        budget_k = env.heldout_best_of_k if method == best_method else 1
        sel_methods = sorted({str(r.get("selection_method", "single_sample")) for r in m_rows})

        compiled_count = sum(1 for r in m_rows if r["compiled"])
        pass_rows = [r for r in m_rows if r["status"] == "PASS"]
        pass_count = len(pass_rows)
        fast_eager_count = sum(1 for r in pass_rows if float(r.get("speedup_vs_eager", 0.0)) > 1.0)

        compile_avail_pass = [r for r in pass_rows if _is_compile_available(r)]
        compile_unavail_pass = [r for r in pass_rows if not _is_compile_available(r)]
        compile_unavail_total = sum(1 for r in m_rows if not _is_compile_available(r))

        # Failures remain in denominator; passing ops with unavailable torch.compile are separated
        fast_compile_denom = n_ops - len(compile_unavail_pass)
        fast_compile_count = sum(1 for r in compile_avail_pass if float(r.get("speedup_vs_compile", 0.0)) > 1.0)
        fast_compile_pct = (
            round(100.0 * fast_compile_count / fast_compile_denom, 2) if fast_compile_denom > 0 else None
        )

        eager_speedups = [
            float(r["speedup_vs_eager"]) for r in pass_rows if float(r.get("speedup_vs_eager", 0.0)) > 0.0
        ]
        compile_speedups = [
            float(r["speedup_vs_compile"])
            for r in compile_avail_pass
            if float(r.get("speedup_vs_compile", 0.0)) > 0.0
        ]

        failure_by_status = {
            st: sum(1 for r in m_rows if r["status"] == st) for st in FAILURE_STATUSES
        }
        failure_count = n_ops - pass_count
        mean_reward = round(sum(float(r.get("reward", 0.0)) for r in m_rows) / n_ops, 4)

        method_summaries[method] = MethodSummary(
            method=method,
            search_budget_k=budget_k,
            selection_methods=sel_methods,
            total_ops=n_ops,
            compiled_count=compiled_count,
            pass_count=pass_count,
            pass_rate_pct=round(100.0 * pass_count / n_ops, 2),
            fast_vs_eager_count=fast_eager_count,
            fast_vs_eager_rate_pct=round(100.0 * fast_eager_count / n_ops, 2),
            fast_vs_compile_count=fast_compile_count,
            fast_vs_compile_denominator=fast_compile_denom,
            fast_vs_compile_rate_pct=fast_compile_pct,
            compile_available_passing_count=len(compile_avail_pass),
            compile_unavailable_passing_count=len(compile_unavail_pass),
            compile_unavailable_total_count=compile_unavail_total,
            geomean_speedup_vs_eager_correct_only=_geomean_positive(eager_speedups),
            geomean_speedup_vs_eager_n=len(eager_speedups),
            geomean_speedup_vs_compile_correct_only=_geomean_positive(compile_speedups),
            geomean_speedup_vs_compile_n=len(compile_speedups),
            failure_count=failure_count,
            failure_counts_by_status=failure_by_status,
            mean_reward=mean_reward,
        )

    # Per-op paired base/trained statuses within this run
    per_op_paired: list[dict[str, Any]] = []
    for op_name in env.heldout_ops:
        b1 = final_rows[("base_pass1", op_name)]
        bk = final_rows[(best_method, op_name)]
        tr = final_rows[("trained_checkpoint", op_name)]

        def _row_view(r: Mapping[str, Any]) -> dict[str, Any]:
            comp_avail = _is_compile_available(r)
            return {
                "run_id": r["run_id"],
                "status": r["status"],
                "compiled": bool(r["compiled"]),
                "correct": bool(r["correct"]),
                "gpu_model": str(_first_present(r, ("gpu_model", "gpu"), default="")),
                "sample_k": int(r.get("sample_k", 0)),
                "budget_k": int(r.get("budget_k", 1)),
                "selection_method": str(r.get("selection_method", "single_sample")),
                "speedup_vs_eager": round(float(r.get("speedup_vs_eager", 0.0)), 4),
                "speedup_vs_compile": round(float(r.get("speedup_vs_compile", 0.0)), 4) if comp_avail else None,
                "compile_available": comp_avail,
                "reward": round(float(r.get("reward", 0.0)), 4),
            }

        b1_pass = b1["status"] == "PASS"
        tr_pass = tr["status"] == "PASS"
        if b1_pass and tr_pass:
            paired_outcome = "both_pass"
        elif tr_pass and not b1_pass:
            paired_outcome = "trained_only_pass"
        elif b1_pass and not tr_pass:
            paired_outcome = "base_pass1_only_pass"
        else:
            paired_outcome = "both_fail"

        per_op_paired.append(
            {
                "op_name": op_name,
                "expected_seed": expected_seeds[op_name],
                "paired_pass1_vs_trained": paired_outcome,
                "base_pass1": _row_view(b1),
                best_method: _row_view(bk),
                "trained_checkpoint": _row_view(tr),
            }
        )

    return {
        "environment": asdict(env),
        "canonical_methods": list(canonical_methods),
        "excluded_row_counts": excluded_counts,
        "methods": {m: asdict(method_summaries[m]) for m in canonical_methods},
        "per_op_paired": per_op_paired,
    }


def _fmt_ratio(num: int, den: int) -> str:
    if den <= 0:
        return f"{num}/0 (N/A)"
    return f"{num}/{den} ({100.0 * num / den:.1f}%)"


def _fmt_geomean(val: float | None, count: int, total: int) -> str:
    if val is None or count <= 0:
        return f"N/A (0/{total} correct)"
    return f"{val:.3f}x (correct-only, {count}/{total})"


def _fmt_failures(counts: Mapping[str, int], total_failures: int) -> str:
    if total_failures == 0:
        return "0"
    parts = [f"{st}={counts[st]}" for st in FAILURE_STATUSES if counts.get(st, 0) > 0]
    return f"{total_failures} ({', '.join(parts)})"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render readable Markdown report with per-run separation and explicit caveats."""
    runs: list[dict[str, Any]] = report["runs"]
    lines: list[str] = [
        "# KernelBench-RL Held-Out Evaluation Comparison",
        "",
        "## Scope and Methodological Notes",
        "",
        f"- **Small held-out set:** Each run evaluates only its configured held-out operators (" + ", ".join(f"N={len(r['environment']['heldout_ops'])}" for r in runs) + "). "
        "No statistical significance claims are made.",
        "- **Different inference search budgets:** `base_pass1` (k=1) and `base_best_of_8` (k=8, "
        "validation-seed selected and regraded on the independent test seed) use different test-time "
        "sampling budgets; only `base_pass1` vs `trained_checkpoint` are matched at k=1.",
        "- **Separate model, precision, and GPU tracks:** Runs are never pooled across different models, "
        "training modes (`Full-Parameter 0.49B` vs `7B QLoRA (NF4)`), dtypes (`fp32`/`bf16`/`fp16`), "
        "or GPUs (e.g., laptop GPU vs Colab T4).",
        "- **Denominator and speedup definitions:** Failures remain in the denominator for pass rate, "
        "`fast_vs_eager`, and `fast_vs_compile`. Passing rows where `torch.compile` was unavailable are "
        "tracked separately rather than counted as fast or silently dropped. Geomean speedups are "
        "explicitly **correct-only**.",
        "- **Strict test-phase filtering:** Only `eval_phase=heldout_test` non-fixture, non-oracle rows "
        "are included; `selection_val`, fixture, and oracle rows are excluded.",
        "",
    ]

    for run in runs:
        env = run["environment"]
        excl = run["excluded_row_counts"]
        n_ops = len(env["heldout_ops"])
        best_method = f"base_best_of_{env['heldout_best_of_k']}"
        lines.extend(
            [
                f"## Run: {env['run_name']} — {env['training_mode_label']} ({env['dtype'].upper()} on {env['gpu']})",
                "",
                f"- **Model:** `{env['model_id']}` (revision `{env['revision']}`)",
                f"- **Training Mode:** {env['training_mode_label']} (`{env['training_mode']}`)",
                f"- **Precision / Backend / GPU:** `{env['dtype']}` / `{env['backend']}` / `{env['gpu']}`",
                f"- **Config Hash:** `{env['config_hash']}` | **Source Commit:** `{env['source_commit']}`",
                f"- **Checkpoint Weights SHA:** `{env['checkpoint_weights_sha']}`",
                f"- **Held-Out Operators (N={n_ops}):** {', '.join(f'`{op}`' for op in env['heldout_ops'])}",
                "- **Excluded Non-Test Rows:** "
                f"selection_val={excl['selection_val']}, fixture={excl['fixture']}, "
                f"oracle={excl['oracle']}, other_eval_phase={excl['other_eval_phase']}",
                "",
                "| Method | Budget | Pass (k/N) | Fast vs Eager | Fast vs Compile (Compile Unavail. Passing) | Geomean vs Eager (Correct-Only) | Geomean vs Compile (Correct-Only) | Failures |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
        )

        for method in run["canonical_methods"]:
            m = run["methods"][method]
            fc_str = _fmt_ratio(m["fast_vs_compile_count"], m["fast_vs_compile_denominator"])
            if m["compile_unavailable_passing_count"] > 0:
                fc_str += f" [unavail: {m['compile_unavailable_passing_count']}]"
            lines.append(
                f"| `{method}` | k={m['search_budget_k']} | "
                f"{_fmt_ratio(m['pass_count'], m['total_ops'])} | "
                f"{_fmt_ratio(m['fast_vs_eager_count'], m['total_ops'])} | "
                f"{fc_str} | "
                f"{_fmt_geomean(m['geomean_speedup_vs_eager_correct_only'], m['geomean_speedup_vs_eager_n'], m['total_ops'])} | "
                f"{_fmt_geomean(m['geomean_speedup_vs_compile_correct_only'], m['geomean_speedup_vs_compile_n'], m['total_ops'])} | "
                f"{_fmt_failures(m['failure_counts_by_status'], m['failure_count'])} |"
            )

        lines.extend(
            [
                "",
                "### Paired Per-Operator Results",
                "",
                f"| Operator | Seed | `base_pass1` (k=1) | `{best_method}` (k={env['heldout_best_of_k']}) | `trained_checkpoint` (k=1) | Paired (`pass1` vs `trained`) |",
                "| :--- | :---: | :--- | :--- | :--- | :--- |",
            ]
        )
        for pair in run["per_op_paired"]:
            def _cell(view: Mapping[str, Any]) -> str:
                if view["status"] == "PASS":
                    comp = (
                        f", {view['speedup_vs_compile']:.2f}x compile"
                        if view["compile_available"] and view["speedup_vs_compile"] is not None
                        else ", compile N/A"
                    )
                    return f"PASS ({view['speedup_vs_eager']:.2f}x eager{comp})"
                return f"{view['status']} (gpu={view['gpu_model']})"

            lines.append(
                f"| `{pair['op_name']}` | {pair['expected_seed']} | "
                f"{_cell(pair['base_pass1'])} | "
                f"{_cell(pair[best_method])} | "
                f"{_cell(pair['trained_checkpoint'])} | "
                f"`{pair['paired_pass1_vs_trained']}` |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_chart(report: Mapping[str, Any], out_png: Path) -> None:
    """Render static PNG showing pass count / denominator per method in separate panels by run."""
    runs: list[dict[str, Any]] = report["runs"]
    num_runs = len(runs)
    fig, axes = plt.subplots(
        1,
        num_runs,
        figsize=(max(6.2 * num_runs, 6.8), 4.8),
        dpi=150,
        squeeze=False,
    )
    colors = ["#475569", "#0284C7", "#16A34A"]

    for col_idx, run in enumerate(runs):
        ax = axes[0][col_idx]
        env = run["environment"]
        n_ops = len(env["heldout_ops"])
        methods = run["canonical_methods"]
        counts = [run["methods"][m]["pass_count"] for m in methods]
        labels = [
            f"{m}\n(k={run['methods'][m]['search_budget_k']})"
            for m in methods
        ]

        bars = ax.bar(range(len(methods)), counts, color=colors[: len(methods)], width=0.55, edgecolor="#1E293B", linewidth=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, max(n_ops, 1) + 0.75)
        ax.set_yticks(range(0, max(n_ops, 1) + 1))
        ax.set_ylabel("Passing Held-Out Ops (count)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{env['run_name']}\n{env['training_mode_label']} | {env['dtype'].upper()} | {env['gpu']} (N={n_ops})",
            fontsize=9.5,
            fontweight="bold",
        )
        for bar, cnt in zip(bars, counts):
            pct = 100.0 * cnt / max(n_ops, 1)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                cnt + 0.1,
                f"{cnt}/{n_ops}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.suptitle(
        "KernelBench-RL held-out pass counts\nSeparate model, precision and GPU runs",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Small held-out set; no significance claims. Failures remain in the denominator.\n"
        "Base pass@1 and trained: k=1. Best-of-K uses a larger search budget.",
        ha="center",
        fontsize=8,
        color="#334155",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.90))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def compare_runs(run_dirs: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    """Validate all run directories and write comparison.md, comparison.json, and comparison.png."""
    if not run_dirs:
        raise HeldoutValidationError("At least one --run-dir must be provided.")

    runs = [load_and_validate_run(Path(rd)) for rd in run_dirs]
    report: dict[str, Any] = {
        "schema_version": "kernelbench_heldout_comparison_v1",
        "pooled_across_runs": False,
        "methodology_notes": {
            "small_test_set": "Each run is evaluated on a small configured held-out set (e.g., N=3); no statistical significance claims are made.",
            "search_budget_note": "base_pass1 (k=1) and base_best_of_k (k>1, validation-seed selected) use different inference-time search budgets; only base_pass1 vs trained_checkpoint share matched k=1 budgets.",
            "hardware_precision_separation": "Runs with distinct models, training modes (Full-Parameter 0.49B vs 7B QLoRA NF4), dtypes, or GPUs are reported separately and never pooled.",
            "denominator_policy": "Failures remain in the denominator for pass rate, fast_vs_eager, and fast_vs_compile; passing ops with unavailable torch.compile are reported separately.",
            "geomean_policy": "Geomean speedups are computed over correct (PASS) rows only and labeled accordingly.",
        },
        "runs": runs,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    md_text = render_markdown(report)
    (out_dir / "comparison.md").write_text(md_text, encoding="utf-8")
    (out_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    render_chart(report, out_dir / "comparison.png")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare held-out evaluation runs (base_pass1, base_best_of_8, trained_checkpoint) within each run."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Path to a held-out run directory containing config.json, heldout.jsonl, and heldout_environment.json (repeatable).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for comparison.md, comparison.json, and comparison.png.",
    )
    args = parser.parse_args(argv)
    compare_runs([Path(p) for p in args.run_dir], Path(args.out))
    print(f"Wrote held-out comparison artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
