"""Strict reporting for the frozen supervised-bootstrap + GRPO retry protocol.

Each run supplies config.json, heldout.jsonl, heldout_environment.json,
sft_only.jsonl and sft_only_environment.json. Its parent supplies protocol.json.
No model, GPU, precision, prompt, checkpoint, or development cohort is pooled.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from eval.compare_heldout import (
    HeldoutValidationError, VALID_STATUSES, _first_present, _load_strict_json,
    _load_strict_jsonl, _parse_environment, _require_finite_float, _require_int,
    _require_non_empty_str, load_and_validate_run,
)
from rl_trainer.experiment_config import ExperimentConfig
from kernels.op_registry import OP_REGISTRY

SFT_METHOD = "sft_only_checkpoint"
FRESH_PHASE = "heldout_test"
DEV_PHASE = "development_test"


def _hash(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise HeldoutValidationError(f"Expected a lowercase SHA-256 digest in {context}.")
    return value


def _checkpoint_identity(manifest: Mapping[str, Any], *, context: str, step_zero: bool = False) -> str:
    step = _require_int(manifest, "checkpoint_step", context=context)
    if step < 0 or (step_zero and step != 0):
        raise HeldoutValidationError(f"Invalid checkpoint_step in {context}; SFT-only must be step 0.")
    sha = _hash(_first_present(manifest, ("checkpoint_weights_sha256", "checkpoint_weights_sha",
                                         "weights_sha256")), context=context)
    identity = f"checkpoint:{step}:{sha}"
    if "checkpoint_identity" in manifest and manifest["checkpoint_identity"] != identity:
        raise HeldoutValidationError(f"Checkpoint identity disagrees with step/weights SHA in {context}.")
    if step_zero and manifest.get("checkpoint_identity") != identity:
        raise HeldoutValidationError(f"Missing SFT-only checkpoint_identity in {context}.")
    return identity


def _validate_manifest_pair(cfg: ExperimentConfig, main: Mapping[str, Any], sft: Mapping[str, Any], env) -> tuple[str, str]:
    context = "sft_only_environment.json"
    expected = {
        "config_hash": env.config_hash, "dtype": env.dtype, "gpu": env.gpu,
        "source_commit": env.source_commit,
    }
    for key, value in expected.items():
        if sft.get(key) != value:
            raise HeldoutValidationError(f"SFT-only {key} mismatch with main run.")
    if _first_present(sft, ("model_id", "model_name_or_path")) not in {env.model_id, env.model_name_or_path}:
        raise HeldoutValidationError("SFT-only model mismatch with main run.")
    if _first_present(sft, ("model_revision", "revision")) != env.revision:
        raise HeldoutValidationError("SFT-only revision mismatch with main run.")
    for key in ("training_mode", "backend", "generation_engine"):
        if key in sft and sft[key] != main.get(key, getattr(cfg, key, None)):
            raise HeldoutValidationError(f"SFT-only {key} mismatch with main run.")
    sft_steps = _require_int(sft, "sft_optimizer_steps", context=context)
    main_sft_steps = _require_int(main, "sft_optimizer_steps", context="heldout_environment.json")
    if sft_steps <= 0 or sft_steps != main_sft_steps:
        raise HeldoutValidationError("SFT-only and hybrid must declare the same positive supervised optimizer step count.")
    optimizer_steps = _require_int(main, "optimizer_steps", context="heldout_environment.json")
    if not 0 <= optimizer_steps <= cfg.max_steps:
        raise HeldoutValidationError("Invalid main GRPO optimizer_steps count.")
    if _require_int(main, "checkpoint_step", context="heldout_environment.json") != cfg.max_steps:
        raise HeldoutValidationError("Main checkpoint has not completed the preregistered GRPO group schedule.")
    return (_checkpoint_identity(main, context="heldout_environment.json"),
            _checkpoint_identity(sft, context=context, step_zero=True))


def _validate_record(row: Mapping[str, Any], *, cfg: ExperimentConfig, env, expected_op: str,
                     expected_seed: int, phase: str, method: str, checkpoint_identity: str,
                     budget: int, context: str) -> None:
    if row.get("method") != method or row.get("op_name") != expected_op or row.get("eval_phase") != phase:
        raise HeldoutValidationError(f"Method/operator/evaluation phase mismatch in {context}.")
    if type(row.get("level")) is not int or row["level"] != OP_REGISTRY[expected_op].level:
        raise HeldoutValidationError(f"Operator level mismatch in {context}.")
    development = expected_op in cfg.development_eval_ops
    for key, expected in (("heldout", not development), ("policy_updated", False), ("fixture", False)):
        if type(row.get(key)) is not bool or row[key] != expected:
            raise HeldoutValidationError(f"Contradictory or missing {key} in {context}.")
    for key, expected in (("split", "development" if development else "heldout"),
                          ("evaluation_cohort", "previously_inspected_development" if development else "fresh_heldout")):
        if row.get(key) != expected:
            raise HeldoutValidationError(f"Contradictory or missing {key} in {context}.")
    if row.get("is_fixture", False):
        raise HeldoutValidationError(f"Synthetic fixture record in {context}.")
    for key, expected in (("config_hash", env.config_hash), ("model_name_or_path", env.model_name_or_path),
                          ("model_revision", env.revision), ("dtype", env.dtype),
                          ("training_mode", env.training_mode), ("backend", env.backend),
                          ("checkpoint_identity", checkpoint_identity)):
        if row.get(key) != expected:
            raise HeldoutValidationError(f"{key} mismatch in {context}.")
    if _require_int(row, "seed", context=context) != expected_seed:
        raise HeldoutValidationError(f"Seed mismatch in {context}.")
    if _require_int(row, "independent_seed", context=context) != expected_seed + cfg.independent_seed_offset:
        raise HeldoutValidationError(f"Independent seed mismatch in {context}.")
    if _require_int(row, "budget_k", context=context) != budget:
        raise HeldoutValidationError(f"Search budget mismatch in {context}.")
    sample = _require_int(row, "sample_k", context=context)
    if not 0 <= sample < budget:
        raise HeldoutValidationError(f"sample_k outside search budget in {context}.")
    expected_selection = "validation_seed_regrade" if budget > 1 else "single_sample"
    if row.get("selection_method") != expected_selection:
        raise HeldoutValidationError(f"Selection method mismatch in {context}.")
    if row.get("sampling_config") != cfg.sampling_config_dict():
        raise HeldoutValidationError(f"Sampling configuration mismatch in {context}.")
    if row.get("generation_engine") != "hf_eager":
        raise HeldoutValidationError(f"Unexpected or missing generation_engine in {context}.")
    _require_non_empty_str(row, ("run_id",), context=context)
    status = row.get("status")
    if status not in VALID_STATUSES:
        raise HeldoutValidationError(f"Invalid status in {context}.")
    for key in ("compiled", "correct", "cheat_detected"):
        if type(row.get(key)) is not bool:
            raise HeldoutValidationError(f"{key} must be boolean in {context}.")
    if (status == "PASS") != row["correct"] or (status == "PASS" and (not row["compiled"] or row["cheat_detected"])):
        raise HeldoutValidationError(f"Inconsistent PASS/correctness flags in {context}.")
    gpu = _require_non_empty_str(row, ("gpu_model", "gpu"), context=context)
    if gpu != env.gpu and not (status != "PASS" and gpu.lower() == "unavailable"):
        raise HeldoutValidationError(f"Mixed GPU hardware in {context}.")
    for key in ("max_abs_err", "eager_ms", "compile_ms", "candidate_ms", "speedup_vs_eager", "speedup_vs_compile", "reward"):
        if key not in row:
            raise HeldoutValidationError(f"Missing numeric field {key} in {context}.")
        val = _require_finite_float(row, key, context=context)
        if key != "reward" and val < 0:
            raise HeldoutValidationError(f"Negative {key} in {context}.")
    if "compile_available" in row and type(row["compile_available"]) is not bool:
        raise HeldoutValidationError(f"compile_available must be boolean in {context}.")
    if row.get("compile_available") is False and (row["compile_ms"] > 0 or row["speedup_vs_compile"] > 0):
        raise HeldoutValidationError(f"Unavailable compile baseline has positive timing/speedup in {context}.")
    if status == "PASS":
        if row["candidate_ms"] <= 0 or row["eager_ms"] <= 0:
            raise HeldoutValidationError(f"PASS requires positive measured timings in {context}.")
        for latency, ratio in (("eager_ms", "speedup_vs_eager"), ("compile_ms", "speedup_vs_compile")):
            expected_ratio = row[latency] / row["candidate_ms"] if row[latency] > 0 else 0
            if not math.isclose(row[ratio], expected_ratio, rel_tol=1e-4, abs_tol=1e-6):
                raise HeldoutValidationError(f"Inconsistent {ratio} and timing in {context}.")
    source = row.get("candidate_source")
    if not isinstance(source, str):
        raise HeldoutValidationError(f"Missing candidate_source string in {context}.")
    source_sha = _hash(row.get("candidate_sha256"), context=context)
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != source_sha:
        raise HeldoutValidationError(f"Candidate source checksum mismatch in {context}.")
    _hash(row.get("prompt_sha256"), context=context)


def _summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    n = len(rows)
    passing = [row for row in rows if row["status"] == "PASS"]
    compile_passing = [row for row in passing if row["compile_ms"] > 0 and row.get("compile_available", True)]
    compile_unavailable = len(passing) - len(compile_passing)
    failure_counts = dict(sorted(Counter(row["status"] for row in rows if row["status"] != "PASS").items()))
    geomean = lambda vals: round(math.exp(sum(math.log(v) for v in vals) / len(vals)), 4) if vals else None
    return {"method": method, "budget_k": rows[0]["budget_k"] if rows else None,
            "total_ops": n, "compiled_count": sum(row["compiled"] for row in rows),
            "pass_count": len(passing), "pass_rate_pct": round(100 * len(passing) / n, 2) if n else None,
            "fast_vs_eager_count": sum(row["speedup_vs_eager"] > 1 for row in passing),
            "fast_vs_compile_count": sum(row["speedup_vs_compile"] > 1 for row in compile_passing),
            "fast_vs_compile_denominator": n - compile_unavailable,
            "compile_unavailable_passing_count": compile_unavailable,
            "geomean_speedup_vs_eager_correct_only": geomean([row["speedup_vs_eager"] for row in passing]),
            "geomean_speedup_vs_eager_n": len(passing), "failure_counts_by_status": failure_counts,
            "per_operator": [{key: row[key] for key in ("op_name", "seed", "independent_seed", "status", "compiled", "correct", "gpu_model", "run_id", "candidate_sha256", "checkpoint_identity")} for row in rows]}


def load_retry_run(run_dir: Path, *, protocol_path: Path | None = None) -> dict[str, Any]:
    """Validate the full retry record inventory before producing any metrics."""
    run_dir = Path(run_dir)
    strict_fresh = load_and_validate_run(run_dir)
    cfg_data, env = _parse_environment(run_dir)
    cfg = ExperimentConfig(**{key: value for key, value in cfg_data.items() if key != "config_hash"})
    if cfg.prompt_version != "contract_v2" or cfg.training_reward_mode != "correctness_gated" or cfg.sft_epochs <= 0:
        raise HeldoutValidationError("Retry reporting requires contract_v2, correctness_gated training and supervised bootstrap.")
    protocol_path = protocol_path or run_dir.parent / "protocol.json"
    protocol = _load_strict_json(protocol_path)
    if not isinstance(protocol, dict) or protocol.get("fresh_heldout_ops") != cfg.heldout_ops or protocol.get("development_regressions") != cfg.development_eval_ops:
        raise HeldoutValidationError("Configured fresh/development split differs from the preregistered protocol.")
    _require_non_empty_str(protocol, ("registered_utc",), context="retry protocol")
    registered_hashes = protocol.get("config_hashes", {})
    if not isinstance(registered_hashes, dict) or registered_hashes.get(run_dir.name) != env.config_hash:
        raise HeldoutValidationError(f"Config hash for run '{run_dir.name}' is not preregistered in protocol config_hashes.")
    expected_schedule = {"max_steps": cfg.max_steps, "sft_epochs": cfg.sft_epochs, "group_size": cfg.group_size}
    registered_schedules = protocol.get("run_schedules", {})
    if not isinstance(registered_schedules, dict) or registered_schedules.get(run_dir.name) != expected_schedule:
        raise HeldoutValidationError(f"Training schedule for run '{run_dir.name}' differs from preregistration.")
    expected_budgets = {"base_pass1": 1, f"base_best_of_{cfg.heldout_best_of_k}_fresh_only": cfg.heldout_best_of_k,
                        SFT_METHOD: 1, "trained_checkpoint": 1}
    if protocol.get("inference_budgets") != expected_budgets:
        raise HeldoutValidationError("Inference budgets differ from the preregistered protocol.")
    if not cfg.development_eval_ops:
        raise HeldoutValidationError("Retry protocol must explicitly retain the inspected development cohort.")
    main_env = _load_strict_json(run_dir / "heldout_environment.json")
    sft_env = _load_strict_json(run_dir / "sft_only_environment.json")
    if not isinstance(sft_env, dict):
        raise HeldoutValidationError("sft_only_environment.json must be an object.")
    trained_identity, sft_identity = _validate_manifest_pair(cfg, main_env, sft_env, env)
    main_rows = _load_strict_jsonl(run_dir / "heldout.jsonl")
    sft_rows = _load_strict_jsonl(run_dir / "sft_only.jsonl")
    best = f"base_best_of_{cfg.heldout_best_of_k}"
    expected_seeds = {op: cfg.seed + cfg.heldout_seed_offset + i for i, op in enumerate(cfg.heldout_ops + cfg.development_eval_ops)}
    finals: dict[tuple[str, str], Mapping[str, Any]] = {}
    selection: dict[tuple[str, int], Mapping[str, Any]] = {}
    run_ids: set[str] = set()
    prompt_hashes: dict[str, str] = {}
    for filename, rows in (("heldout.jsonl", main_rows), ("sft_only.jsonl", sft_rows)):
        for index, row in enumerate(rows, 1):
            context = f"{run_dir / filename} row {index}"
            op, method, phase = row.get("op_name"), row.get("method"), row.get("eval_phase")
            if op not in expected_seeds:
                raise HeldoutValidationError(f"Unregistered operator in {context}.")
            dev = op in cfg.development_eval_ops
            if filename == "sft_only.jsonl":
                if method != SFT_METHOD:
                    raise HeldoutValidationError(f"SFT-only file contains a different method in {context}.")
                identity, budget = sft_identity, 1
            else:
                if method not in {"base_pass1", best, "trained_checkpoint"}:
                    raise HeldoutValidationError(f"Unexpected main method in {context}.")
                if dev and method == best:
                    raise HeldoutValidationError("Best-of-K is not preregistered for inspected development operators.")
                identity = trained_identity if method == "trained_checkpoint" else "base:" + env.revision
                budget = cfg.heldout_best_of_k if method == best else 1
            expected_phase = DEV_PHASE if dev else FRESH_PHASE
            seed = expected_seeds[op]
            if phase == "selection_val" and method == best and not dev and filename == "heldout.jsonl":
                expected_phase = "selection_val"
                seed += cfg.selection_seed_offset
            _validate_record(row, cfg=cfg, env=env, expected_op=op, expected_seed=seed,
                             phase=expected_phase, method=method, checkpoint_identity=identity,
                             budget=budget, context=context)
            if row["run_id"] in run_ids:
                raise HeldoutValidationError(f"Duplicate run_id in retry records: {row['run_id']}.")
            run_ids.add(row["run_id"])
            previous_prompt = prompt_hashes.setdefault(op, row["prompt_sha256"])
            if previous_prompt != row["prompt_sha256"]:
                raise HeldoutValidationError(f"Prompt changed across methods for operator {op}.")
            key = (op, row["sample_k"]) if phase == "selection_val" else (method, op)
            target = selection if phase == "selection_val" else finals
            if key in target:
                raise HeldoutValidationError(f"Duplicate retry evaluation identity {key}.")
            target[key] = row
    expected_final_keys = {(method, op) for op in cfg.heldout_ops for method in ("base_pass1", best, "trained_checkpoint", SFT_METHOD)}
    expected_final_keys |= {(method, op) for op in cfg.development_eval_ops for method in ("base_pass1", "trained_checkpoint", SFT_METHOD)}
    if set(finals) != expected_final_keys:
        raise HeldoutValidationError(f"Incomplete retry final coverage; missing {sorted(expected_final_keys - set(finals))}.")
    expected_selection = {(op, k) for op in cfg.heldout_ops for k in range(cfg.heldout_best_of_k)}
    if set(selection) != expected_selection:
        raise HeldoutValidationError("Incomplete validation-seed Best-of-K selection inventory.")
    for op in cfg.heldout_ops:
        selected = finals[(best, op)]
        val = selection[(op, selected["sample_k"])]
        if selected["candidate_sha256"] != val["candidate_sha256"]:
            raise HeldoutValidationError(f"Best-of-K test candidate differs from selected validation candidate for {op}.")
        maximum = max(selection[(op, k)]["reward"] for k in range(cfg.heldout_best_of_k))
        if val["reward"] != maximum:
            raise HeldoutValidationError(f"Best-of-K winner was not validation-reward maximal for {op}.")
    fresh = {method: _summary([finals[(method, op)] for op in cfg.heldout_ops], method) for method in ("base_pass1", best, SFT_METHOD, "trained_checkpoint")}
    development = {method: _summary([finals[(method, op)] for op in cfg.development_eval_ops], method) for method in ("base_pass1", SFT_METHOD, "trained_checkpoint")}
    development[best] = {"available": False, "reason": "Not preregistered for inspected development operators; no Best-of-K generation budget allocated."}
    return {"environment": strict_fresh["environment"], "prompt_version": cfg.prompt_version,
            "protocol_path": str(protocol_path), "protocol_registered_utc": protocol["registered_utc"],
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "training_reward_mode": cfg.training_reward_mode,
            "sft_epochs": cfg.sft_epochs, "sft_optimizer_steps": main_env["sft_optimizer_steps"],
            "grpo_optimizer_steps": main_env["optimizer_steps"],
            "trained_checkpoint_identity": trained_identity, "sft_only_checkpoint_identity": sft_identity,
            "inventory": {"main_rows": len(main_rows), "main_selection_rows": len(selection),
                          "main_fresh_final_rows": sum(method != SFT_METHOD and op in cfg.heldout_ops for method, op in finals),
                          "main_development_final_rows": sum(method != SFT_METHOD and op in cfg.development_eval_ops for method, op in finals),
                          "sft_only_rows": len(sft_rows), "sft_fresh_final_rows": len(cfg.heldout_ops),
                          "sft_development_final_rows": len(cfg.development_eval_ops)},
            "fresh_heldout": {"ops": cfg.heldout_ops, "methods": fresh},
            "inspected_development": {"ops": cfg.development_eval_ops, "methods": development},
            "strict_main_comparison": strict_fresh}


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("`", "'")


def render_retry_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# KernelBench-RL frozen retry comparison", "",
             "Each model, GPU, precision and configuration is reported separately. No pooled metric or significance claim is made.", "",
             "The trained checkpoint combines a supervised bootstrap with GRPO. Base-to-trained changes cannot be attributed to RL alone. The separately evaluated SFT-only checkpoint describes the incremental comparison within this frozen schedule; N=3 does not establish a general causal effect.", "",
             "Previously inspected development operators never enter fresh held-out denominators. Development Best-of-K is unavailable because it was not budgeted. All failures remain in pass-rate denominators; geometric mean speedups are correct-only.", "",
             "Historical contract_v1 results and this contract_v2 retry use different prompts, operator cohorts and training protocols. Historical results are context, not a paired same-protocol before/after comparison.", ""]
    for run in report["runs"]:
        env = run["environment"]
        lines.extend([f"## {_md(env['run_name'])}", "",
                      f"Model: `{_md(env['model_id'])}` · revision `{_md(env['revision'])}` · `{_md(env['dtype'])}` · `{_md(env['gpu'])}`.", "",
                      f"Config `{env['config_hash']}` · source commit `{_md(env['source_commit'])}` · prompt `{run['prompt_version']}`.", "",
                      f"SFT updates: {run['sft_optimizer_steps']}; GRPO optimizer updates: {run['grpo_optimizer_steps']}. Identical checkpoint hashes are permitted when GRPO makes no weight update.", "",
                      f"SFT checkpoint: `{run['sft_only_checkpoint_identity']}`; hybrid checkpoint: `{run['trained_checkpoint_identity']}`.", ""])
        for title, key in (("Fresh held-out operators", "fresh_heldout"), ("Previously inspected development operators", "inspected_development")):
            cohort = run[key]
            lines.extend([f"### {title} (N={len(cohort['ops'])})", "", ", ".join(f"`{_md(op)}`" for op in cohort["ops"]), "",
                          "| Method | Budget | Passed | Fast vs eager | Fast vs compile | Correct-only geomean | Failures |", "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |"])
            for method, stats in cohort["methods"].items():
                if stats.get("available") is False:
                    lines.append(f"| `{method}` | unavailable | unavailable | unavailable | unavailable | unavailable | Not budgeted |")
                    continue
                gm = stats["geomean_speedup_vs_eager_correct_only"]
                gm_text = f"{gm:.4f}x ({stats['geomean_speedup_vs_eager_n']}/{stats['total_ops']})" if gm is not None else "unavailable (0 correct)"
                denom = stats["fast_vs_compile_denominator"]
                comp = f"{stats['fast_vs_compile_count']}/{denom}" if denom else "unavailable"
                fail = ", ".join(f"{_md(k)}={v}" for k, v in stats["failure_counts_by_status"].items()) or "none"
                lines.append(f"| `{method}` | {stats['budget_k']} | {stats['pass_count']}/{stats['total_ops']} | {stats['fast_vs_eager_count']}/{stats['total_ops']} | {comp} | {gm_text} | {fail} |")
            lines.append("")
        lines.extend(["Inventory: " + ", ".join(f"{key}={value}" for key, value in run["inventory"].items()) + ".", ""])
    if report.get("historical_context"):
        lines.extend(["## Historical context (different protocol; unpaired)", ""])
        for run in report["historical_context"]:
            env = run["environment"]
            lines.append(f"- {_md(env['run_name'])}: {_md(env['model_id'])}, {_md(env['dtype'])}, {_md(env['gpu'])}; " + ", ".join(f"{method}={stats['pass_count']}/{stats['total_ops']}" for method, stats in run["methods"].items()) + ".")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compare_retry_runs(run_dirs: Sequence[Path], out_dir: Path, *, protocol_path: Path | None = None,
                       historical_run_dirs: Sequence[Path] = ()) -> dict[str, Any]:
    if not run_dirs:
        raise HeldoutValidationError("At least one retry --run-dir is required.")
    resolved = [Path(path).resolve() for path in run_dirs]
    if len(set(resolved)) != len(resolved):
        raise HeldoutValidationError("Duplicate retry run directories.")
    runs = [load_retry_run(path, protocol_path=protocol_path) for path in run_dirs]
    history = [load_and_validate_run(Path(path)) for path in historical_run_dirs]
    report = {"schema_version": "kernelbench_retry_comparison_v1", "pooled_across_runs": False,
              "rl_only_causal_claim": False, "historical_paired_same_protocol": False,
              "runs": runs, "historical_context": history}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retry_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "retry_comparison.md").write_text(render_retry_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--historical-run-dir", action="append", default=[], type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    compare_retry_runs(args.run_dir, args.out, protocol_path=args.protocol, historical_run_dirs=args.historical_run_dir)
    print(f"Wrote strict retry comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
