"""Single-step on-policy recovery over an un-updated initial-policy rollout batch.

Salvages a multi-group rollout trace (e.g., 8 groups x 2 samples = 16 completions)
where the parent model generated and recorded exact raw tokens + log-probs,
performed ZERO optimizer steps (`policy_updated=False`, `optimizer_step_count=0`),
and grading failed due to an external worker environment issue.

Protocol:
1. Validate every trace row strictly (reject any updated policy, mismatched
   config/model/revision/prompt/group length, non-finite log-probs, duplicate
   run IDs, or source-hash mismatch).
2. Regrade ALL sources using the strict GPU grader BEFORE loading any model.
3. Normalize GRPO advantages independently within each original group.
4. Load the exact same initial policy, verify every stored `old_token_logps`
   (and `reference_token_logps` via `disable_adapter()` when applicable) against
   the unchanged initial policy within `--logp-atol`—aborting on any stale mismatch.
5. Accumulate all completions through `perform_verified_grpo_step` for ONE
   single on-policy optimizer step (never stepping sequentially per old group).
6. Write all outputs into a brand-new empty `--out-dir`: adapter+optimizer+RNG
   checkpoint, regraded JSONL (`method="grpo_recovered_initial_batch"`),
   original-compatible `config.json`, and SHA-256-hashed `recovery_manifest.json`.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from kernels.op_registry import OP_REGISTRY
from rl_trainer.checkpointing import CheckpointState, save_checkpoint, load_checkpoint
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import (
    SampledCandidate,
    _disable_all_dropout,
    _release_cuda_memory,
    compute_selected_token_logps,
    extract_python_source,
    perform_verified_grpo_step,
)
from rl_trainer.grpo_reward import compute_grpo_advantages
from rl_trainer.prompting import sha256_text, build_prompt, format_prompt_for_tokenizer
from rl_trainer.qlora_grpo import compute_shared_base_ref_logps, select_trainable_adapter_params
from sandbox_runner.gpu_verifier import grade_candidate

RECOVERY_METHOD = "grpo_recovered_initial_batch"
RECOVERY_SCHEDULE = "one_optimizer_step_after_initial_groups"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_non_finite_json(token: str) -> None:
    raise ValueError(f"Non-finite JSON constant in trace: {token}")


def load_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input trace JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line, parse_constant=_reject_non_finite_json)
        if not isinstance(obj, dict):
            raise ValueError(f"Line {lineno} in {path} is not a JSON object.")
        rows.append(obj)
    if not rows:
        raise ValueError(f"Input trace JSONL is empty: {path}")
    return rows


def _extract_token_list(row: dict[str, Any], keys: Sequence[str], label: str) -> list[int]:
    for k in keys:
        if k in row and row[k] is not None:
            val = row[k]
            if not isinstance(val, list) or not val or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in val):
                raise ValueError(f"Invalid {label} in key '{k}' for run_id={row.get('run_id')!r}.")
            return list(val)
    raise ValueError(f"Missing required {label} (tried {list(keys)}) for run_id={row.get('run_id')!r}.")


def _extract_logp_list(row: dict[str, Any], keys: Sequence[str], expected_len: int, label: str) -> list[float]:
    for k in keys:
        if k in row and row[k] is not None:
            val = row[k]
            if not isinstance(val, list) or len(val) != expected_len:
                raise ValueError(
                    f"Length mismatch for {label} (key '{k}'): got {len(val) if isinstance(val, list) else type(val).__name__}, "
                    f"expected {expected_len} for run_id={row.get('run_id')!r}."
                )
            out: list[float] = []
            for x in val:
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                    raise ValueError(f"Non-finite value in {label} for run_id={row.get('run_id')!r}.")
                out.append(float(x))
            return out
    raise ValueError(f"Missing required {label} (tried {list(keys)}) for run_id={row.get('run_id')!r}.")


def validate_initial_batch_traces(
    rows: list[dict[str, Any]],
    cfg: ExperimentConfig,
) -> list[list[dict[str, Any]]]:
    """Strictly validate an un-updated initial-policy rollout batch and return ordered groups.

    Rejects:
    - Any row with `policy_updated=True` or `optimizer_step_count > 0`
    - Mismatched `config_hash`, `model_name_or_path`, `model_revision`, `training_mode`, or unknown `op_name`
    - Duplicate `run_id` or `(step, group_index)`
    - Missing/mismatched/non-finite token IDs or log-probs
    - Source SHA-256 mismatch
    - Incomplete group sizes (`!= cfg.group_size`) or mismatched prompts within a group
    """
    if not rows:
        raise ValueError("Cannot recover from an empty rollout batch.")

    expected_hash = cfg.config_hash()
    seen_run_ids: set[str] = set()
    seen_slots: set[tuple[int, int]] = set()
    groups_map: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for idx, row in enumerate(rows):
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"Row {idx} missing valid string run_id.")
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate run_id detected: '{run_id}'.")
        seen_run_ids.add(run_id)

        if row.get("policy_updated") is not False:
            raise ValueError(f"Refusing recovery: row '{run_id}' already has policy_updated=True.")
        opt_steps = row.get("optimizer_step_count")
        if not isinstance(opt_steps, int) or isinstance(opt_steps, bool) or opt_steps != 0:
            raise ValueError(f"Refusing recovery: row '{run_id}' has optimizer_step_count={opt_steps!r} (expected 0).")

        row_hash = row.get("config_hash")
        if row_hash != expected_hash:
            raise ValueError(
                f"Config hash mismatch on '{run_id}': row has '{row_hash}', config expects '{expected_hash}'."
            )
        row_model = row.get("model_name_or_path")
        if row_model != cfg.model_name_or_path:
            raise ValueError(
                f"Model mismatch on '{run_id}': row has '{row_model}', config expects '{cfg.model_name_or_path}'."
            )
        row_rev = row.get("model_revision")
        if row_rev != cfg.model_revision:
            raise ValueError(
                f"Revision mismatch on '{run_id}': row has '{row_rev}', config expects '{cfg.model_revision}'."
            )
        row_mode = row.get("training_mode")
        if row_mode != cfg.training_mode:
            raise ValueError(
                f"Training mode mismatch on '{run_id}': row has '{row_mode}', config expects '{cfg.training_mode}'."
            )

        op_name = row.get("op_name")
        if not isinstance(op_name, str) or op_name not in OP_REGISTRY:
            raise ValueError(f"Unknown or missing op_name '{op_name}' on '{run_id}'.")

        step = row.get("step")
        g_idx = row.get("group_index")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError(f"Invalid step={step!r} on '{run_id}'.")
        if not isinstance(g_idx, int) or isinstance(g_idx, bool) or g_idx < 0:
            raise ValueError(f"Invalid group_index={g_idx!r} on '{run_id}'.")
        slot = (step, g_idx)
        if op_name != cfg.train_ops[(step - 1) % len(cfg.train_ops)]:
            raise ValueError("Operator does not match the configured training cursor")
        if "sampling_config" in row and row["sampling_config"] != cfg.sampling_config_dict():
            raise ValueError("Sampling distribution does not match the original configuration")
        if slot in seen_slots:
            raise ValueError(f"Duplicate (step, group_index) slot {slot} on '{run_id}'.")
        seen_slots.add(slot)

        # Validate raw token IDs & log-probs
        prompt_ids = _extract_token_list(row, ("prompt_token_ids", "prompt_ids"), "prompt_token_ids")
        comp_ids = _extract_token_list(row, ("completion_token_ids", "completion_ids"), "completion_token_ids")
        old_lps = _extract_logp_list(row, ("old_token_logps",), len(comp_ids), "old_token_logps")
        ref_lps = _extract_logp_list(row, ("reference_token_logps", "ref_token_logps"), len(comp_ids), "reference_token_logps")

        # Validate source & SHA-256 hash integrity
        raw_comp = row.get("raw_completion")
        cand_src = row.get("candidate_source")
        if cand_src is None:
            if isinstance(raw_comp, str):
                cand_src = extract_python_source(raw_comp)
            else:
                raise ValueError(f"Row '{run_id}' missing both candidate_source and raw_completion.")
        if not isinstance(cand_src, str):
            raise ValueError(f"Invalid candidate_source type on '{run_id}'.")

        expected_sha = sha256_text(cand_src)
        recorded_sha = row.get("candidate_sha256")
        if recorded_sha != expected_sha:
            raise ValueError(
                f"candidate_sha256 mismatch on '{run_id}': recorded '{recorded_sha}' != actual '{expected_sha}'."
            )
        if isinstance(raw_comp, str) and extract_python_source(raw_comp).strip() != cand_src.strip():
            raise ValueError("Extracted source differs from the recorded raw completion")

        norm = dict(row)
        norm["_prompt_ids"] = prompt_ids
        norm["_completion_ids"] = comp_ids
        norm["_old_token_logps"] = old_lps
        norm["_ref_token_logps"] = ref_lps
        norm["_candidate_source"] = cand_src
        norm["_candidate_sha256"] = expected_sha
        norm["_raw_completion"] = raw_comp if isinstance(raw_comp, str) else cand_src
        groups_map[step].append(norm)

    ordered_groups: list[list[dict[str, Any]]] = []
    for step in sorted(groups_map.keys()):
        grp = sorted(groups_map[step], key=lambda r: int(r["group_index"]))
        if [r["group_index"] for r in grp] != list(range(cfg.group_size)):
            raise ValueError("Group indices must cover the original sampling slots exactly")
        if len(grp) != cfg.group_size:
            raise ValueError(
                f"Group step={step} has {len(grp)} samples, expected group_size={cfg.group_size}."
            )
        ref_op = grp[0]["op_name"]
        ref_prompt_ids = grp[0]["_prompt_ids"]
        for member in grp[1:]:
            if member["op_name"] != ref_op:
                raise ValueError(f"Mixed op_name inside group step={step}: '{member['op_name']}' vs '{ref_op}'.")
            if member["_prompt_ids"] != ref_prompt_ids:
                raise ValueError(f"Mismatched prompt_token_ids inside group step={step} for '{member['run_id']}'.")
        ordered_groups.append(grp)

    if sorted(groups_map) != list(range(1, len(groups_map) + 1)):
        raise ValueError("Recovered groups must be a complete initial prefix")
    return ordered_groups


def verify_stored_logps_against_policy(
    *,
    policy: Any,
    samples: list[SampledCandidate],
    logp_fn: Callable[[Any, list[int], list[int]], Any],
    logp_verify_atol: float = 1e-3,
    verify_ref_with_disable_adapter: bool = True,
) -> tuple[float, float]:
    if not math.isfinite(logp_verify_atol) or logp_verify_atol <= 0:
        raise ValueError("logp_verify_atol must be finite and positive")
    try:
        import torch
        context = torch.no_grad()
    except ImportError:
        context = contextlib.nullcontext()
    with context:
        return _verify_stored_logps(policy=policy, samples=samples, logp_fn=logp_fn,
            logp_verify_atol=logp_verify_atol,
            verify_ref_with_disable_adapter=verify_ref_with_disable_adapter)


def _verify_stored_logps(*, policy, samples, logp_fn, logp_verify_atol,
                         verify_ref_with_disable_adapter):
    """Verify stored `old_token_logps` (and `ref_token_logps`) match the freshly loaded initial policy.

    Raises `ValueError` if any completion exceeds `logp_verify_atol`, preventing
    stale-policy or mismatched-init updates.
    """
    max_old_err = 0.0
    max_ref_err = 0.0

    for idx, smp in enumerate(samples):
        assert smp.old_token_logps is not None and smp.ref_token_logps is not None
        cur_old = logp_fn(policy, smp.prompt_ids, smp.completion_ids)
        cur_old_np = cur_old.detach().float().cpu().numpy() if hasattr(cur_old, "detach") else np.asarray(cur_old, dtype=np.float64)
        stored_old_np = np.asarray(smp.old_token_logps, dtype=np.float64)

        if cur_old_np.shape != stored_old_np.shape or not np.all(np.isfinite(cur_old_np)):
            raise ValueError(f"Sample {idx}: recomputed old_token_logps shape/finiteness mismatch.")
        old_err = float(np.max(np.abs(cur_old_np - stored_old_np)))
        max_old_err = max(max_old_err, old_err)
        if old_err > logp_verify_atol:
            raise ValueError(
                f"Stale/mismatched policy detected on sample {idx}: "
                f"max|recomputed_old_logp - stored_old_logp|={old_err:.4e} > atol={logp_verify_atol:.4e}."
            )

        if verify_ref_with_disable_adapter and hasattr(policy, "disable_adapter"):
            cur_ref = compute_shared_base_ref_logps(policy, logp_fn, smp.prompt_ids, smp.completion_ids)
            cur_ref_np = cur_ref.detach().float().cpu().numpy() if hasattr(cur_ref, "detach") else np.asarray(cur_ref, dtype=np.float64)
            stored_ref_np = np.asarray(smp.ref_token_logps, dtype=np.float64)
            if cur_ref_np.shape != stored_ref_np.shape or not np.all(np.isfinite(cur_ref_np)):
                raise ValueError(f"Sample {idx}: recomputed reference_token_logps shape/finiteness mismatch.")
            ref_err = float(np.max(np.abs(cur_ref_np - stored_ref_np)))
            max_ref_err = max(max_ref_err, ref_err)
            if ref_err > logp_verify_atol:
                raise ValueError(
                    f"Frozen base reference mismatch on sample {idx}: "
                    f"max|recomputed_ref_logp - stored_ref_logp|={ref_err:.4e} > atol={logp_verify_atol:.4e}."
                )
        print(json.dumps({"event": "recovery_logp_verified", "sample": idx,
                          "max_old_err": max_old_err, "max_ref_err": max_ref_err}), flush=True)

    return max_old_err, max_ref_err


def _ensure_empty_out_dir(out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"--out-dir exists and is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            raise FileExistsError(f"Refusing to write into non-empty --out-dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def recover_initial_batch(
    *,
    input_jsonl: Path,
    config_path: Path,
    out_dir: Path,
    grader_fn: Callable[..., dict[str, Any]] = grade_candidate,
    model_loader_fn: Callable[[ExperimentConfig], tuple[Any, list[Any], Any, Callable[[Any, list[int], list[int]], Any], Any]] | None = None,
    logp_verify_atol: float = 1e-3,
    is_fixture: bool = False,
    initial_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Execute full initial-batch recovery: validate -> regrade (no model) -> load & verify -> 1 step -> save."""
    _ensure_empty_out_dir(out_dir)

    cfg = ExperimentConfig.from_json(config_path)
    cfg.validate(strict_on_policy_sampling=True)
    raw_rows = load_strict_jsonl(input_jsonl)
    groups = validate_initial_batch_traces(raw_rows, cfg)
    initial_state = None
    if initial_checkpoint is not None:
        initial_state = load_checkpoint(initial_checkpoint, expected_config_hash=cfg.config_hash(), restore_rng=False)
        if initial_state.optimizer_step_count != 0:
            raise ValueError("Recovery requires an initial checkpoint with zero optimizer steps")
    if model_loader_fn is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, revision=cfg.model_revision,
                                                  local_files_only=True, trust_remote_code=False)
        for group in groups:
            formatted, _, prompt_sha = format_prompt_for_tokenizer(build_prompt(group[0]["op_name"], cfg.backend), tokenizer)
            if tokenizer(formatted)["input_ids"] != group[0]["_prompt_ids"]:
                raise ValueError("Stored prompt tokens do not match the operator's original chat prompt")
            for row in group:
                if row.get("prompt_sha256") != prompt_sha:
                    raise ValueError("Stored formatted prompt checksum does not match")
                if tokenizer.decode(row["_completion_ids"], skip_special_tokens=True) != row["_raw_completion"]:
                    raise ValueError("Raw completion differs from its sampled token IDs")
    input_jsonl_sha = _sha256_file(input_jsonl)
    input_config_sha = _sha256_file(config_path)

    # Phase 1: Regrade ALL sources with NO model loaded in VRAM, then normalize within each original group
    regraded_records: list[dict[str, Any]] = []
    all_samples: list[SampledCandidate] = []
    all_advantages: list[float] = []

    for grp in groups:
        grp_records: list[dict[str, Any]] = []
        for row in grp:
            op_name = row["op_name"]
            row_seed = int(row.get("seed", cfg.seed + int(row["step"])))
            indep_seed = int(row.get("independent_seed", row_seed + cfg.independent_seed_offset))
            rec = grader_fn(
                op_name,
                row["_candidate_source"],
                backend=cfg.backend,
                float_dtype=cfg.dtype,
                seed=row_seed,
                independent_seed=indep_seed,
                warmup=cfg.warmup_iters,
                iters=cfg.bench_iters,
                timeout_secs=cfg.worker_timeout_secs,
                run_id=f"recover-{row['run_id']}",
                method=RECOVERY_METHOD,
            )
            rec["original_run_id"] = row["run_id"]
            rec["step"] = int(row["step"])
            rec["group_index"] = int(row["group_index"])
            rec["candidate_source"] = row["_candidate_source"]
            rec["candidate_sha256"] = row["_candidate_sha256"]
            rec["prompt_token_ids"] = row["_prompt_ids"]
            rec["completion_token_ids"] = row["_completion_ids"]
            rec["old_token_logps"] = row["_old_token_logps"]
            rec["reference_token_logps"] = row["_ref_token_logps"]
            rec["model_name_or_path"] = cfg.model_name_or_path
            rec["model_revision"] = cfg.model_revision
            rec["training_mode"] = cfg.training_mode
            rec["sampling_config"] = cfg.sampling_config_dict()
            rec["config_hash"] = cfg.config_hash()
            rec["recovery_schedule"] = RECOVERY_SCHEDULE
            rec["source_trace_sha256"] = input_jsonl_sha
            rec["split"] = "train"
            rec["fixture"] = bool(is_fixture)
            rec["raw_completion"] = row["_raw_completion"]
            rec["prompt_sha256"] = row.get("prompt_sha256", "")
            rec["chat_template_applied"] = row.get("chat_template_applied", False)
            with (out_dir / "regraded_before_update.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, allow_nan=False) + "\n")
            print(json.dumps({"event": "recovery_grade", "op_name": op_name,
                              "status": rec["status"], "reward": rec["reward"]}), flush=True)
            grp_records.append(rec)

            all_samples.append(
                SampledCandidate(
                    prompt_ids=row["_prompt_ids"],
                    completion_ids=row["_completion_ids"],
                    raw_text=row["_raw_completion"],
                    extracted_source=row["_candidate_source"],
                    old_token_logps=row["_old_token_logps"],
                    ref_token_logps=row["_ref_token_logps"],
                    chat_template_applied=bool(row.get("chat_template_applied", False)),
                    prompt_sha256=str(row.get("prompt_sha256", "")),
                )
            )

        grp_advs = compute_grpo_advantages([float(r.get("reward", 0.0)) for r in grp_records])
        for r, adv in zip(grp_records, grp_advs):
            r["grpo_advantage"] = float(adv)
        regraded_records.extend(grp_records)
        all_advantages.extend(grp_advs)

    # Phase 2: Load exact initial policy and verify stored old_token_logps BEFORE any optimizer step
    if model_loader_fn is None:
        policy, trainable_params, optimizer, logp_fn, device = _default_load_initial_policy(cfg)
    else:
        policy, trainable_params, optimizer, logp_fn, device = model_loader_fn(cfg)
    if initial_checkpoint is not None:
        load_checkpoint(initial_checkpoint, expected_config_hash=cfg.config_hash(),
                        model=policy, optimizer=optimizer, restore_rng=True)

    policy.eval()
    _disable_all_dropout(policy)
    if hasattr(policy, "config") and hasattr(policy.config, "use_cache"):
        policy.config.use_cache = False

    max_old_err, max_ref_err = verify_stored_logps_against_policy(
        policy=policy,
        samples=all_samples,
        logp_fn=logp_fn,
        logp_verify_atol=logp_verify_atol,
        verify_ref_with_disable_adapter=(cfg.training_mode == "qlora_7b_nf4"),
    )

    # Phase 3: ONE accumulated on-policy GRPO step across all completions (never per-group!)
    policy_updated = False
    update_metrics: dict[str, Any] = {
        "policy_updated": False,
        "policy_loss": 0.0,
        "mean_kl": 0.0,
        "weight_l2_delta": 0.0,
        "trainable_weights_sha256": "",
    }
    if any(abs(a) > 1e-8 for a in all_advantages):
        policy.train()
        _disable_all_dropout(policy)
        if hasattr(policy, "config") and hasattr(policy.config, "use_cache"):
            policy.config.use_cache = False
        update_metrics = perform_verified_grpo_step(
            policy=policy,
            trainable_params=trainable_params,
            optimizer=optimizer,
            samples=all_samples,
            advantages=all_advantages,
            logp_fn=logp_fn,
            clip_eps=cfg.clip_eps,
            kl_coef=cfg.kl_coef,
            max_grad_norm=cfg.max_grad_norm,
            device=device,
        )
        policy_updated = bool(update_metrics.get("policy_updated", False))

    opt_steps = 1 if policy_updated else 0
    num_groups = len(groups)
    total_rollouts = len(regraded_records)
    mean_reward = sum(float(r.get("reward", 0.0)) for r in regraded_records) / max(total_rollouts, 1)

    # Phase 4: Save honest outputs into brand-new --out-dir
    out_jsonl = out_dir / "recovered_rollouts.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in regraded_records:
            rec["policy_updated"] = policy_updated
            rec["optimizer_step_count"] = opt_steps
            rec["max_old_logp_verify_err"] = float(max_old_err)
            rec["max_ref_logp_verify_err"] = float(max_ref_err)
            rec["weight_l2_delta"] = float(update_metrics.get("weight_l2_delta", 0.0))
            rec["trainable_weights_sha256"] = str(update_metrics.get("trainable_weights_sha256", ""))
            rec["policy_loss"] = float(update_metrics.get("policy_loss", 0.0))
            rec["mean_kl"] = float(update_metrics.get("mean_kl", 0.0))
            fh.write(json.dumps(rec, allow_nan=False) + "\n")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_state = CheckpointState(
        step=num_groups,
        op_cursor=num_groups,
        seed=cfg.seed,
        config_hash=cfg.config_hash(),
        best_mean_reward=float(mean_reward),
        total_rollouts=total_rollouts,
        optimizer_step_count=opt_steps,
        trainable_weights_sha256=str(update_metrics.get("trainable_weights_sha256", "")),
        training_mode=cfg.training_mode,
        is_fixture=is_fixture,
    )
    save_checkpoint(
        ckpt_dir,
        ckpt_state,
        model=policy,
        optimizer=optimizer,
        adapter_only=(cfg.training_mode == "qlora_7b_nf4"),
        save_rng=True,
    )

    # Write original-compatible config.json for held-out evaluation
    compat_cfg_path = out_dir / "config.json"
    cfg.checkpoint_dir = str(ckpt_dir)
    cfg.output_jsonl = str(out_jsonl)
    cfg.to_json(compat_cfg_path)

    # Write separately SHA-256-hashed recovery manifest documenting the 1-step-over-N-groups schedule
    recovered_jsonl_sha = _sha256_file(out_jsonl)
    manifest: dict[str, Any] = {
        "method": RECOVERY_METHOD,
        "recovery_schedule": RECOVERY_SCHEDULE,
        "schedule_note": (
            f"Single accumulated on-policy clipped GRPO update across {num_groups} initial groups "
            f"({total_rollouts} completions, group_size={cfg.group_size}) after pre-load regrading; "
            "config_hash matches original ExperimentConfig for eval compatibility and does not by itself encode this recovery schedule."
        ),
        "input_jsonl": str(input_jsonl),
        "input_jsonl_sha256": input_jsonl_sha,
        "input_config": str(config_path),
        "input_config_sha256": input_config_sha,
        "original_config_hash": cfg.config_hash(),
        "recovered_rollouts_sha256": recovered_jsonl_sha,
        "model_name_or_path": cfg.model_name_or_path,
        "model_revision": cfg.model_revision,
        "training_mode": cfg.training_mode,
        "dtype": cfg.dtype,
        "num_groups": num_groups,
        "total_rollouts": total_rollouts,
        "group_size": cfg.group_size,
        "logp_verify_atol": float(logp_verify_atol),
        "max_old_logp_verify_err": float(max_old_err),
        "max_ref_logp_verify_err": float(max_ref_err),
        "policy_updated": policy_updated,
        "optimizer_step_count": opt_steps,
        "policy_loss": float(update_metrics.get("policy_loss", 0.0)),
        "mean_kl": float(update_metrics.get("mean_kl", 0.0)),
        "weight_l2_delta": float(update_metrics.get("weight_l2_delta", 0.0)),
        "trainable_weights_sha256": str(update_metrics.get("trainable_weights_sha256", "")),
        "fixture": bool(is_fixture),
        "recovery_source_sha256": _sha256_file(Path(__file__)),
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "initial_weights_sha256": initial_state.trainable_weights_sha256 if initial_state else None,
    }
    canonical_manifest_bytes = json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical_manifest_bytes).hexdigest()
    (out_dir / "recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _default_load_initial_policy(cfg: ExperimentConfig):
    """Load the initial 7B NF4+PEFT policy seeded identically to collection."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore

    if not torch.cuda.is_available() and cfg.backend == "triton":
        raise RuntimeError("CUDA is required to load the initial GPU policy for backend='triton'.")

    torch.manual_seed(cfg.seed)
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    compute_dtype = dtype_map[cfg.dtype] if torch.cuda.is_available() else torch.float32

    if cfg.training_mode == "qlora_7b_nf4":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import BitsAndBytesConfig  # type: ignore

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            revision=cfg.model_revision,
            quantization_config=bnb_cfg,
            device_map={"": 0}, local_files_only=True, trust_remote_code=False,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=cfg.gradient_checkpointing)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(base, lora_cfg)
        _disable_all_dropout(policy)
        trainable = select_trainable_adapter_params(policy)
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
        device = next(policy.parameters()).device
    else:
        raise ValueError("The default recovery loader supports 7B QLoRA; provide an explicit loader for other modes")

    def _logp_fn(model_obj: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
        return compute_selected_token_logps(model_obj, prompt_ids, comp_ids, device)

    _release_cuda_memory(torch)
    return policy, trainable, optimizer, _logp_fn, device


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover an un-updated initial-policy GRPO batch as one accumulated step.")
    parser.add_argument("--input", required=True, help="Original train.jsonl path (read-only; never modified).")
    parser.add_argument("--config", required=True, help="Original ExperimentConfig config.json path.")
    parser.add_argument("--out-dir", required=True, help="Brand-new empty output directory.")
    parser.add_argument("--initial-checkpoint", type=Path, help="Optional zero-update checkpoint preserving the exact original adapter initialization.")
    parser.add_argument("--logp-atol", type=float, default=1e-3, help="Max allowed |recomputed - stored| old_token_logp diff.")
    args = parser.parse_args(argv)

    manifest = recover_initial_batch(
        input_jsonl=Path(args.input),
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        logp_verify_atol=args.logp_atol,
        is_fixture=False,
        initial_checkpoint=args.initial_checkpoint,
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
