"""Read-only diagnostic script for teacher-forced log-prob differences on a single recorded sample.

Measures without performing any model updates or modifying raw traces:
- `use_cache=True` vs `use_cache=False` (under `eval()` with dropout off)
- same-process repeatability (`use_cache=False` repeated back-to-back)
- `eval()` vs `train()` with dropout off (`use_cache=False`)
- stored `old_token_logps` (and `reference_token_logps` when present) vs recomputed logps
- initial zero-update adapter checkpoint checksum (when `--initial-checkpoint` is provided)

Loads strictly from local cached weights/config (no network download) and writes strict finite JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from rl_trainer.checkpointing import load_checkpoint, resolve_checkpoint_dir, compute_trainable_weights_sha256
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import _disable_all_dropout, compute_selected_token_logps
from rl_trainer.recover_initial_batch import _default_load_initial_policy


def _reject_non_finite_json(token: str) -> None:
    raise ValueError(f"Non-finite JSON constant rejected: {token}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_source_commit(cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for mod_name in ("torch", "triton", "transformers", "peft", "bitsandbytes", "numpy"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[mod_name] = "unavailable"
    return versions


def _to_1d_float64(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        arr = values.detach().float().cpu().numpy().astype(np.float64)
    else:
        arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D log-prob array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Non-finite values detected in log-prob array.")
    return arr


def _diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError(f"Log-prob length mismatch: {a.shape} vs {b.shape}.")
    if a.size == 0:
        return {
            "token_count": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "p95_abs_diff": 0.0,
            "max_abs_diff_token_index": -1,
        }
    diffs = np.abs(a - b)
    max_idx = int(np.argmax(diffs))
    return {
        "token_count": int(a.size),
        "max_abs_diff": float(diffs[max_idx]),
        "mean_abs_diff": float(np.mean(diffs)),
        "p95_abs_diff": float(np.percentile(diffs, 95)),
        "max_abs_diff_token_index": max_idx,
    }


def _hash_checkpoint_weights(step_dir: Path, adapter_dir: str, model_weights_file: str) -> tuple[str, list[str]]:
    hasher = hashlib.sha256()
    hashed_files: list[str] = []
    if adapter_dir:
        adir = step_dir / adapter_dir
        if not adir.is_dir():
            raise FileNotFoundError(f"Adapter directory missing in checkpoint: {adir}")
        for fp in sorted(p for p in adir.rglob("*") if p.is_file()):
            rel = fp.relative_to(step_dir).as_posix()
            hasher.update(rel.encode("utf-8"))
            hasher.update(fp.read_bytes())
            hashed_files.append(rel)
    elif model_weights_file:
        fp = step_dir / model_weights_file
        if not fp.is_file():
            raise FileNotFoundError(f"Weights file missing in checkpoint: {fp}")
        hasher.update(fp.name.encode("utf-8"))
        hasher.update(fp.read_bytes())
        hashed_files.append(fp.name)
    else:
        raise ValueError(f"Checkpoint at {step_dir} has neither adapter_dir nor model_weights_file.")
    return hasher.hexdigest(), hashed_files


def _load_local_initial_policy(cfg: ExperimentConfig) -> tuple[Any, Any, Any, Any, Any]:
    """Load initial policy using recover_initial_batch loader with offline env guards (no downloads)."""
    prev_hf = os.environ.get("HF_HUB_OFFLINE")
    prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        return _default_load_initial_policy(cfg)
    finally:
        if prev_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_hf
        if prev_tf is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_tf


def _eval_logps_with_cache_mode(
    policy: Any,
    prompt_ids: list[int],
    comp_ids: list[int],
    device: Any,
    *,
    use_cache: bool,
    logp_eval_fn: Callable[[Any, list[int], list[int], bool], Any] | None = None,
) -> np.ndarray:
    """Evaluate teacher-forced logps under explicit `use_cache` without permanently mutating config."""
    had_cfg = hasattr(policy, "config") and hasattr(policy.config, "use_cache")
    prev_cfg_cache = getattr(policy.config, "use_cache", None) if had_cfg else None
    if had_cfg:
        policy.config.use_cache = bool(use_cache)
    try:
        if logp_eval_fn is not None:
            vals = logp_eval_fn(policy, prompt_ids, comp_ids, bool(use_cache))
        else:
            vals = compute_selected_token_logps(policy, prompt_ids, comp_ids, device, use_cache=bool(use_cache))
        return _to_1d_float64(vals)
    finally:
        if had_cfg:
            policy.config.use_cache = prev_cfg_cache


def _no_grad_ctx() -> Any:
    try:
        import torch  # type: ignore

        return torch.no_grad()
    except ImportError:
        return contextlib.nullcontext()


def _grad_ctx() -> Any:
    try:
        import torch
        return torch.enable_grad()
    except ImportError:
        return contextlib.nullcontext()


def run_logp_diagnostic(
    *,
    config_path: Path,
    input_jsonl: Path,
    out_json: Path,
    initial_checkpoint: Path | None = None,
    sample_index: int = 0,
    policy_loader: Callable[[ExperimentConfig], tuple[Any, Any, Any, Any, Any]] | None = None,
    logp_eval_fn: Callable[[Any, list[int], list[int], bool], Any] | None = None,
) -> dict[str, Any]:
    """Run read-only log-prob diagnostics on a single recorded sample and write strict finite JSON."""
    if out_json.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic output: {out_json}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}.")

    cfg = ExperimentConfig.from_json(config_path)
    cfg_hash = cfg.config_hash()

    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(input_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        obj = json.loads(stripped, parse_constant=_reject_non_finite_json)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {input_jsonl}:{line_no}.")
        rows.append(obj)

    if sample_index >= len(rows):
        raise IndexError(f"sample_index={sample_index} out of range for {len(rows)} rows in {input_jsonl}.")

    row = rows[sample_index]
    if row.get("config_hash") != cfg_hash:
        raise ValueError("Trace config hash differs from the supplied configuration")
    for field in ("model_name_or_path", "model_revision", "training_mode"):
        if field in row and row[field] != getattr(cfg, field):
            raise ValueError(f"Trace {field} differs from the supplied configuration")
    for field in ("prompt_token_ids", "completion_token_ids"):
        ids = row.get(field, [])
        if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError(f"Invalid token IDs in {field}")
    prompt_ids = [int(x) for x in row.get("prompt_token_ids", [])]
    comp_ids = [int(x) for x in row.get("completion_token_ids", [])]
    stored_old = _to_1d_float64(row.get("old_token_logps", []))
    if not prompt_ids or not comp_ids:
        raise ValueError(f"Sample {sample_index} is missing prompt_token_ids or completion_token_ids.")
    if len(stored_old) != len(comp_ids):
        raise ValueError(
            f"Sample {sample_index} old_token_logps length ({len(stored_old)}) != completion_token_ids length ({len(comp_ids)})."
        )
    raw_stored_ref = row.get("reference_token_logps")
    stored_ref = _to_1d_float64(raw_stored_ref) if raw_stored_ref is not None and len(raw_stored_ref) > 0 else None

    loader = _load_local_initial_policy if policy_loader is None else policy_loader
    policy, _trainable, _optimizer, _default_logp_fn, device = loader(cfg)

    checkpoint_info: dict[str, Any] | None = None
    if initial_checkpoint is not None:
        step_dir = resolve_checkpoint_dir(initial_checkpoint)
        ckpt_state = load_checkpoint(
            step_dir,
            expected_config_hash=cfg_hash,
            model=policy if hasattr(policy, "state_dict") else None,
            optimizer=None,
            restore_rng=False,
            map_location=str(device),
        )
        if ckpt_state.optimizer_step_count != 0:
            raise ValueError(
                f"Initial checkpoint must have optimizer_step_count == 0, got {ckpt_state.optimizer_step_count}."
            )
        weights_sha, hashed_files = _hash_checkpoint_weights(
            step_dir, ckpt_state.adapter_dir, ckpt_state.model_weights_file
        )
        checkpoint_info = {
            "checkpoint_dir": str(step_dir),
            "step": int(ckpt_state.step),
            "optimizer_step_count": int(ckpt_state.optimizer_step_count),
            "config_hash": ckpt_state.config_hash,
            "adapter_weights_sha256": weights_sha,
            "hashed_files": hashed_files,
            "trainable_weights_sha256": ckpt_state.trainable_weights_sha256,
        }

    weights_before = compute_trainable_weights_sha256(_trainable) if _trainable else None

    with _no_grad_ctx():
        # 1. eval() + dropout off + use_cache=True
        if hasattr(policy, "eval"):
            policy.eval()
        _disable_all_dropout(policy)
        lp_eval_cache_true = _eval_logps_with_cache_mode(
            policy, prompt_ids, comp_ids, device, use_cache=True, logp_eval_fn=logp_eval_fn
        )

        # 2. eval() + dropout off + use_cache=False (run 1 & run 2 for same-process repeatability)
        lp_eval_cache_false_1 = _eval_logps_with_cache_mode(
            policy, prompt_ids, comp_ids, device, use_cache=False, logp_eval_fn=logp_eval_fn
        )
        lp_eval_cache_false_2 = _eval_logps_with_cache_mode(
            policy, prompt_ids, comp_ids, device, use_cache=False, logp_eval_fn=logp_eval_fn
        )

        # 3. train() + dropout off + use_cache=False
        if hasattr(policy, "train"):
            policy.train()
        _disable_all_dropout(policy)
        # Enable the actual training/checkpointing forward path, then detach;
        # no backward or optimizer step is performed.
        with _grad_ctx():
            lp_train_cache_false = _eval_logps_with_cache_mode(
                policy, prompt_ids, comp_ids, device, use_cache=False, logp_eval_fn=logp_eval_fn
            )

        # Restore eval mode and optionally measure disable_adapter reference logps
        if hasattr(policy, "eval"):
            policy.eval()
        _disable_all_dropout(policy)

        ref_comparisons: dict[str, Any] = {}
        if hasattr(policy, "disable_adapter") and callable(policy.disable_adapter):
            with policy.disable_adapter():
                lp_ref_cache_true = _eval_logps_with_cache_mode(
                    policy, prompt_ids, comp_ids, device, use_cache=True, logp_eval_fn=logp_eval_fn
                )
                lp_ref_cache_false = _eval_logps_with_cache_mode(
                    policy, prompt_ids, comp_ids, device, use_cache=False, logp_eval_fn=logp_eval_fn
                )
            ref_comparisons["ref_use_cache_true_vs_false"] = _diff_stats(lp_ref_cache_true, lp_ref_cache_false)
            ref_comparisons["recomputed_old_vs_ref_use_cache_false"] = _diff_stats(
                lp_eval_cache_false_1, lp_ref_cache_false
            )
            if stored_ref is not None and len(stored_ref) == len(comp_ids):
                ref_comparisons["stored_ref_vs_recomputed_ref_use_cache_true"] = _diff_stats(
                    stored_ref, lp_ref_cache_true
                )
                ref_comparisons["stored_ref_vs_recomputed_ref_use_cache_false"] = _diff_stats(
                    stored_ref, lp_ref_cache_false
                )

    gpu_name = "unavailable"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu_name = str(torch.cuda.get_device_name(0))
    except Exception:
        pass

    weights_after = compute_trainable_weights_sha256(_trainable) if _trainable else None
    if weights_after != weights_before:
        raise RuntimeError("Read-only diagnostic unexpectedly changed trainable weights")
    report: dict[str, Any] = {
        "schema_version": "kernelbench_logp_diagnostic_v1",
        "policy_updated": False,
        "tolerance_changed": False,
        "trainable_weights_sha256_before": weights_before,
        "trainable_weights_sha256_after": weights_after,
        "train_probe_autograd_enabled": True,
        "note": (
            "Read-only numerical diagnostic on a single recorded sample. Reports measured "
            "log-prob differences without modifying raw traces, updating weights, changing "
            "recovery tolerances, or asserting an unmeasured cause."
        ),
        "provenance": {
            "source_commit": _git_source_commit(),
            "config_path": str(config_path),
            "config_hash": cfg_hash,
            "input_jsonl": str(input_jsonl),
            "input_jsonl_sha256": _sha256_file(input_jsonl),
            "model_name_or_path": cfg.model_name_or_path,
            "model_revision": cfg.model_revision,
            "training_mode": cfg.training_mode,
            "dtype": cfg.dtype,
            "gpu": gpu_name,
            "versions": _package_versions(),
        },
        "sample": {
            "sample_index": int(sample_index),
            "run_id": str(row.get("run_id", "")),
            "op_name": str(row.get("op_name", "")),
            "prompt_len": len(prompt_ids),
            "completion_len": len(comp_ids),
            "row_config_hash": str(row.get("config_hash", "")),
            "row_policy_updated": bool(row.get("policy_updated", False)),
        },
        "initial_checkpoint": checkpoint_info,
        "comparisons": {
            "use_cache_true_vs_false_eval": _diff_stats(lp_eval_cache_true, lp_eval_cache_false_1),
            "same_process_repeatability_eval_use_cache_false": _diff_stats(
                lp_eval_cache_false_1, lp_eval_cache_false_2
            ),
            "eval_vs_train_dropout_off_use_cache_false": _diff_stats(
                lp_eval_cache_false_1, lp_train_cache_false
            ),
            "stored_old_vs_recomputed_eval_use_cache_true": _diff_stats(stored_old, lp_eval_cache_true),
            "stored_old_vs_recomputed_eval_use_cache_false": _diff_stats(stored_old, lp_eval_cache_false_1),
            "stored_old_vs_recomputed_train_use_cache_false": _diff_stats(stored_old, lp_train_cache_false),
            **ref_comparisons,
        },
    }

    # Verify strict finiteness before writing
    payload_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(payload_text, encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only log-prob diagnostic (use_cache, repeatability, eval vs train, stored vs recomputed)."
    )
    parser.add_argument("--config", required=True, help="Path to ExperimentConfig JSON.")
    parser.add_argument("--input", required=True, help="Path to recorded train.jsonl (read-only).")
    parser.add_argument(
        "--initial-checkpoint",
        default="",
        help="Optional path to zero-update initial checkpoint directory to load and checksum.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="0-based sample index to inspect (default: 0).")
    parser.add_argument("--out", required=True, help="Path to brand-new output JSON file.")
    args = parser.parse_args(argv)

    report = run_logp_diagnostic(
        config_path=Path(args.config),
        input_jsonl=Path(args.input),
        out_json=Path(args.out),
        initial_checkpoint=Path(args.initial_checkpoint) if args.initial_checkpoint else None,
        sample_index=args.sample_index,
    )
    print(json.dumps(report["comparisons"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
