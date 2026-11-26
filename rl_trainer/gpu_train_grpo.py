"""On-policy clipped GRPO training bridge with `grpo_loss_torch` delegation and CPU FP32 master AdamW."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from rl_trainer.checkpointing import (
    CheckpointState,
    compute_trainable_weights_sha256,
    compute_weight_l2_delta,
    load_checkpoint,
    save_checkpoint,
    snapshot_trainable_params,
    verify_compatible_run_paths,
)
from rl_trainer.cpu_master_adamw import CpuMasterAdamW
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.grpo_reward import compute_grpo_advantages
from rl_trainer.prompting import build_prompt, format_prompt_for_tokenizer, sha256_text
from sandbox_runner.gpu_verifier import grade_candidate

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

try:
    from rl_trainer.objective import grpo_loss_torch as _REPO_GRPO_LOSS_TORCH  # type: ignore
except ImportError:
    _REPO_GRPO_LOSS_TORCH = None


@dataclass
class SampledCandidate:
    prompt_ids: list[int]
    completion_ids: list[int]
    raw_text: str
    extracted_source: str
    old_token_logps: list[float] | None = None
    ref_token_logps: list[float] | None = None
    chat_template_applied: bool = False
    prompt_sha256: str = ""


def extract_python_source(raw_completion: str) -> str:
    match = _CODE_BLOCK_RE.search(raw_completion)
    if match:
        return match.group(1).strip()
    return raw_completion.strip()


def compute_selected_token_logps(
    model_obj: Any,
    prompt_ids: list[int],
    comp_ids: list[int],
    device: Any,
    *,
    chunk_size: int = 32,
) -> Any:
    """Compute exact selected-token log-probs using `logits_to_keep` + chunked logsumexp."""
    import torch  # type: ignore

    if not comp_ids:
        return torch.empty((0,), dtype=torch.float32, device=device)

    full = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
    p_len = len(prompt_ids)
    c_len = len(comp_ids)

    supports_keep = False
    try:
        sig = inspect.signature(model_obj.forward)
        supports_keep = "logits_to_keep" in sig.parameters
        if not supports_keep and hasattr(model_obj, "get_base_model"):
            supports_keep = "logits_to_keep" in inspect.signature(model_obj.get_base_model().forward).parameters
    except (TypeError, ValueError):
        supports_keep = False

    if supports_keep:
        out = model_obj(full, logits_to_keep=c_len + 1)
        comp_logits = out.logits[0, :-1, :]
    else:
        out = model_obj(full)
        comp_logits = out.logits[0, p_len - 1 : p_len - 1 + c_len, :]

    target_ids = torch.tensor(comp_ids, dtype=torch.long, device=device)
    chunks = []
    step_sz = max(1, int(chunk_size))
    for start in range(0, c_len, step_sz):
        end = min(start + step_sz, c_len)
        sub_logits = comp_logits[start:end, :].float()
        sub_targets = target_ids[start:end].unsqueeze(-1)
        chosen = sub_logits.gather(-1, sub_targets).squeeze(-1)
        lse = torch.logsumexp(sub_logits, dim=-1)
        chunks.append(chosen - lse)
    return torch.cat(chunks, dim=0)


def compute_selected_token_logps_numpy(logits_2d: Any, target_ids: Sequence[int], chunk_size: int = 32) -> Any:
    import numpy as np

    logits = np.asarray(logits_2d, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=np.int64)
    c_len = len(targets)
    out = np.empty((c_len,), dtype=np.float64)
    for start in range(0, c_len, max(1, chunk_size)):
        end = min(start + chunk_size, c_len)
        sub = logits[start:end, :]
        m = np.max(sub, axis=-1, keepdims=True)
        lse = m.squeeze(-1) + np.log(np.sum(np.exp(sub - m), axis=-1))
        chosen = sub[np.arange(end - start), targets[start:end]]
        out[start:end] = chosen - lse
    return out


def compute_clipped_grpo_loss_torch(
    new_logps: Any,
    old_logps: Any,
    ref_logps: Any,
    advantage: float,
    clip_eps: float,
    kl_coef: float,
    *,
    objective_fn: Callable[..., Any] | None = None,
) -> tuple[Any, Any]:
    """Delegate directly to `rl_trainer.objective.grpo_loss_torch([G, T], mask, clip=, beta=)`.

    Never catches arbitrary exceptions from `grpo_loss_torch`; any mismatch or
    bug in the objective surfaces immediately.
    """
    import torch  # type: ignore

    fn = _REPO_GRPO_LOSS_TORCH if objective_fn is None else objective_fn
    if fn is not None:
        new_2d = new_logps.unsqueeze(0) if new_logps.ndim == 1 else new_logps
        old_2d = old_logps.unsqueeze(0) if old_logps.ndim == 1 else old_logps
        ref_2d = ref_logps.unsqueeze(0) if ref_logps.ndim == 1 else ref_logps
        mask_2d = torch.ones_like(new_2d, dtype=torch.float32)
        adv_1d = torch.tensor([float(advantage)], device=new_2d.device, dtype=torch.float32)

        sig = inspect.signature(fn)
        kw: dict[str, Any] = {}
        if "mask" in sig.parameters:
            kw["mask"] = mask_2d
        if "clip" in sig.parameters:
            kw["clip"] = clip_eps
        if "beta" in sig.parameters:
            kw["beta"] = kl_coef

        res = fn(new_2d, old_2d, ref_2d, adv_1d, **kw)
        if isinstance(res, tuple):
            loss = res[0]
            kl = res[1] if len(res) > 1 else torch.tensor(0.0, device=new_2d.device)
            return loss, kl
        if isinstance(res, dict):
            return res["loss"], res.get("kl", torch.tensor(0.0, device=new_2d.device))
        # Scalar loss returned directly; compute exact KL for telemetry
        ref_diff = ref_2d - new_2d
        kl = (torch.exp(torch.clamp(ref_diff, min=-20.0, max=20.0)) - ref_diff - 1.0).mean()
        return res, kl

    # Standalone fallback used ONLY when rl_trainer.objective is absent (ImportError)
    log_ratio = new_logps - old_logps
    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    adv_t = torch.tensor(float(advantage), device=new_logps.device, dtype=torch.float32)
    pg_loss = -torch.minimum(ratio * adv_t, clipped_ratio * adv_t).mean()
    ref_diff = ref_logps - new_logps
    kl = (torch.exp(torch.clamp(ref_diff, min=-20.0, max=20.0)) - ref_diff - 1.0).mean()
    return pg_loss + kl_coef * kl, kl


def compute_clipped_grpo_loss_numpy(
    new_logps: Sequence[float],
    old_logps: Sequence[float],
    ref_logps: Sequence[float],
    advantage: float,
    clip_eps: float,
    kl_coef: float,
) -> tuple[float, float]:
    import numpy as np

    new_a = np.asarray(new_logps, dtype=np.float64)
    old_a = np.asarray(old_logps, dtype=np.float64)
    ref_a = np.asarray(ref_logps, dtype=np.float64)
    ratio = np.exp(np.clip(new_a - old_a, -20.0, 20.0))
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = -float(np.mean(np.minimum(ratio * advantage, clipped * advantage)))
    diff = ref_a - new_a
    kl = float(np.mean(np.exp(np.clip(diff, -20.0, 20.0)) - diff - 1.0))
    return pg_loss + kl_coef * kl, kl


def all_grads_finite(params: Iterable[Any]) -> bool:
    import torch  # type: ignore

    saw_grad = False
    for p in params:
        if not getattr(p, "requires_grad", False) or p.grad is None:
            continue
        saw_grad = True
        if not bool(torch.isfinite(p.grad).all()):
            return False
    return saw_grad


def perform_verified_grpo_step(
    *,
    policy: Any,
    trainable_params: list[Any],
    optimizer: Any,
    samples: list[SampledCandidate],
    advantages: list[float],
    logp_fn: Callable[[Any, list[int], list[int]], Any],
    clip_eps: float,
    kl_coef: float,
    max_grad_norm: float,
    device: Any,
) -> dict[str, Any]:
    """Execute one clipped GRPO backward + finite-grad-verified optimizer step (supports CpuMasterAdamW)."""
    import torch  # type: ignore

    is_cpu_master = isinstance(optimizer, CpuMasterAdamW)
    optimizer.zero_grad(set_to_none=True)
    before_snaps = [] if is_cpu_master else snapshot_trainable_params(trainable_params)

    valid_terms = 0
    total_loss = 0.0
    total_kl = 0.0
    skipped_non_finite_loss = False

    for smp, adv in zip(samples, advantages):
        if not smp.completion_ids or abs(adv) < 1e-8:
            continue
        assert smp.old_token_logps is not None and smp.ref_token_logps is not None
        new_lp = logp_fn(policy, smp.prompt_ids, smp.completion_ids)
        old_lp = torch.tensor(smp.old_token_logps, dtype=torch.float32, device=device)
        ref_lp = torch.tensor(smp.ref_token_logps, dtype=torch.float32, device=device)
        loss, kl = compute_clipped_grpo_loss_torch(new_lp, old_lp, ref_lp, adv, clip_eps, kl_coef)
        if not (bool(torch.isfinite(loss).all()) and bool(torch.isfinite(kl).all())):
            skipped_non_finite_loss = True
            continue
        (loss / max(len(samples), 1)).backward()
        total_loss += float(loss.detach().cpu().item())
        total_kl += float(kl.detach().cpu().item())
        valid_terms += 1

    if valid_terms == 0:
        optimizer.zero_grad(set_to_none=True)
        sha_now = optimizer.master_weights_sha256() if is_cpu_master else compute_trainable_weights_sha256(trainable_params)
        return {
            "policy_updated": False,
            "skipped_non_finite_grad": False,
            "skipped_non_finite_loss": skipped_non_finite_loss,
            "policy_loss": 0.0,
            "mean_kl": 0.0,
            "weight_l2_delta": 0.0,
            "trainable_weights_sha256": sha_now,
        }

    if is_cpu_master:
        updated, delta = optimizer.copy_grads_and_step(max_grad_norm)
        sha_after = optimizer.master_weights_sha256()
        if not updated:
            return {
                "policy_updated": False,
                "skipped_non_finite_grad": True,
                "skipped_non_finite_loss": skipped_non_finite_loss,
                "policy_loss": 0.0,
                "mean_kl": 0.0,
                "weight_l2_delta": 0.0,
                "trainable_weights_sha256": sha_after,
            }
        return {
            "policy_updated": True,
            "skipped_non_finite_grad": False,
            "skipped_non_finite_loss": skipped_non_finite_loss,
            "policy_loss": total_loss / valid_terms,
            "mean_kl": total_kl / valid_terms,
            "weight_l2_delta": delta,
            "trainable_weights_sha256": sha_after,
        }

    if not all_grads_finite(trainable_params):
        optimizer.zero_grad(set_to_none=True)
        return {
            "policy_updated": False,
            "skipped_non_finite_grad": True,
            "skipped_non_finite_loss": skipped_non_finite_loss,
            "policy_loss": 0.0,
            "mean_kl": 0.0,
            "weight_l2_delta": 0.0,
            "trainable_weights_sha256": compute_trainable_weights_sha256(trainable_params),
        }

    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    delta = compute_weight_l2_delta(before_snaps, trainable_params)
    sha_after = compute_trainable_weights_sha256(trainable_params)
    return {
        "policy_updated": True,
        "skipped_non_finite_grad": False,
        "skipped_non_finite_loss": skipped_non_finite_loss,
        "policy_loss": total_loss / valid_terms,
        "mean_kl": total_kl / valid_terms,
        "weight_l2_delta": delta,
        "trainable_weights_sha256": sha_after,
    }


def _disable_all_dropout(model: Any) -> None:
    for mod in model.modules():
        if "dropout" in type(mod).__name__.lower():
            mod.eval()
            if hasattr(mod, "p"):
                mod.p = 0.0


def _release_cuda_memory(torch_mod: Any) -> None:
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


def rollout_and_grade_on_policy(
    op_name: str,
    cfg: ExperimentConfig,
    *,
    step: int,
    sample_fn: Callable[[str, int, int], list[SampledCandidate]],
    grader_fn: Callable[..., dict[str, Any]] = grade_candidate,
    before_grade_hook: Callable[[], None] | None = None,
    after_grade_hook: Callable[[], None] | None = None,
    method: str = "grpo_train",
    is_fixture: bool = False,
) -> tuple[list[dict[str, Any]], list[SampledCandidate], list[float]]:
    prompt = build_prompt(op_name, backend=cfg.backend)
    samples = sample_fn(prompt, cfg.group_size, cfg.seed + step)
    if len(samples) != cfg.group_size:
        raise ValueError(f"Sampler returned {len(samples)} rollouts, expected group_size={cfg.group_size}")

    if before_grade_hook is not None:
        before_grade_hook()
    try:
        records: list[dict[str, Any]] = []
        for g_idx, smp in enumerate(samples):
            rec = grader_fn(
                op_name,
                smp.extracted_source,
                backend=cfg.backend,
                float_dtype=cfg.dtype,
                seed=cfg.seed + step,
                independent_seed=cfg.seed + cfg.independent_seed_offset + step,
                warmup=cfg.warmup_iters,
                iters=cfg.bench_iters,
                timeout_secs=cfg.worker_timeout_secs,
                run_id=f"step{step:05d}-g{g_idx}",
                method=method,
            )
            rec["candidate_source"] = smp.extracted_source
            rec["candidate_sha256"] = sha256_text(smp.extracted_source)
            rec["completion_token_count"] = len(smp.completion_ids)
            rec["completion_token_ids"] = smp.completion_ids
            rec["prompt_token_ids"] = smp.prompt_ids
            rec["raw_completion"] = smp.raw_text
            rec["old_token_logps"] = smp.old_token_logps
            rec["reference_token_logps"] = smp.ref_token_logps
            rec["chat_template_applied"] = bool(smp.chat_template_applied)
            rec["prompt_sha256"] = smp.prompt_sha256 or sha256_text(prompt)
            rec["step"] = int(step)
            rec["group_index"] = int(g_idx)
            rec["split"] = "train"
            rec["model_name_or_path"] = cfg.model_name_or_path
            rec["model_revision"] = cfg.model_revision
            rec["training_mode"] = cfg.training_mode
            rec["cpu_master_adamw"] = bool(cfg.cpu_master_adamw)
            rec["sampling_config"] = cfg.sampling_config_dict()
            rec["config_hash"] = cfg.config_hash()
            rec["fixture"] = bool(is_fixture)
            records.append(rec)
    finally:
        if after_grade_hook is not None:
            after_grade_hook()

    rewards = [float(r.get("reward", 0.0)) for r in records]
    advantages = compute_grpo_advantages(rewards)
    for rec, adv in zip(records, advantages):
        rec["grpo_advantage"] = float(adv)
    return records, samples, advantages


def run_gpu_training_loop(
    cfg: ExperimentConfig,
    *,
    sample_fn: Callable[[str, int, int], list[SampledCandidate]],
    update_fn: Callable[[list[SampledCandidate], list[float]], dict[str, Any]] | None = None,
    grader_fn: Callable[..., dict[str, Any]] = grade_candidate,
    before_grade_hook: Callable[[], None] | None = None,
    after_grade_hook: Callable[[], None] | None = None,
    model: Any = None,
    optimizer: Any = None,
    adapter_only: bool = False,
    overwrite: bool = False,
    is_fixture: bool = False,
) -> CheckpointState:
    cfg.validate(strict_on_policy_sampling=True)
    ckpt_dir = Path(cfg.checkpoint_dir)
    out_jsonl = Path(cfg.output_jsonl)
    cfg_hash = cfg.config_hash()

    verify_compatible_run_paths(
        ckpt_dir,
        out_jsonl,
        cfg_hash,
        is_resuming=bool(cfg.resume_from),
        overwrite=overwrite,
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_jsonl.exists() and not cfg.resume_from:
        out_jsonl.unlink()

    if cfg.resume_from:
        state = load_checkpoint(
            cfg.resume_from,
            expected_config_hash=cfg_hash,
            model=model,
            optimizer=optimizer,
            restore_rng=True,
        )
        start_step = state.step + 1
        op_cursor = state.op_cursor
        best_mean_reward = state.best_mean_reward
        total_rollouts = state.total_rollouts
        optimizer_step_count = state.optimizer_step_count
        last_weights_sha = state.trainable_weights_sha256
    else:
        start_step = 1
        op_cursor = 0
        best_mean_reward = float("-inf")
        total_rollouts = 0
        optimizer_step_count = 0
        last_weights_sha = ""
        state = CheckpointState(
            step=0,
            op_cursor=0,
            seed=cfg.seed,
            config_hash=cfg_hash,
            best_mean_reward=0.0,
            total_rollouts=0,
            optimizer_step_count=0,
            trainable_weights_sha256="",
            training_mode=cfg.training_mode,
            is_fixture=is_fixture,
        )

    for step in range(start_step, cfg.max_steps + 1):
        op_name = cfg.train_ops[op_cursor % len(cfg.train_ops)]
        print(json.dumps({"event": "step_start", "step": step, "op_name": op_name}), flush=True)
        op_cursor += 1
        records, samples, advantages = rollout_and_grade_on_policy(
            op_name,
            cfg,
            step=step,
            sample_fn=sample_fn,
            grader_fn=grader_fn,
            before_grade_hook=before_grade_hook,
            after_grade_hook=after_grade_hook,
            method="grpo_train" if not is_fixture else "grpo_fixture",
            is_fixture=is_fixture,
        )

        policy_updated = False
        update_metrics: dict[str, Any] = {}
        if update_fn is not None and any(abs(a) > 1e-8 for a in advantages):
            update_metrics = update_fn(samples, advantages)
            policy_updated = bool(update_metrics.get("policy_updated", False))
            if policy_updated:
                optimizer_step_count += 1
            if update_metrics.get("trainable_weights_sha256"):
                last_weights_sha = str(update_metrics["trainable_weights_sha256"])

        mean_reward = sum(float(r["reward"]) for r in records) / max(len(records), 1)
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
        total_rollouts += len(records)
        print(json.dumps({"event": "step_complete", "step": step, "statuses": [r["status"] for r in records],
                          "rewards": [r["reward"] for r in records], "optimizer_step_count": optimizer_step_count,
                          **update_metrics}, allow_nan=False), flush=True)

        with out_jsonl.open("a", encoding="utf-8") as fh:
            for rec in records:
                rec["policy_updated"] = policy_updated
                rec["optimizer_step_count"] = int(optimizer_step_count)
                for mk, mv in update_metrics.items():
                    if mk in {"policy_updated", "optimizer_step_count"}:
                        continue
                    if isinstance(mv, float) and math.isfinite(mv):
                        rec[mk] = mv
                    elif isinstance(mv, (int, bool, str)):
                        rec[mk] = mv
                fh.write(json.dumps(rec, allow_nan=False) + "\n")

        state = CheckpointState(
            step=step,
            op_cursor=op_cursor,
            seed=cfg.seed,
            config_hash=cfg_hash,
            best_mean_reward=0.0 if not math.isfinite(best_mean_reward) else best_mean_reward,
            total_rollouts=total_rollouts,
            optimizer_step_count=optimizer_step_count,
            trainable_weights_sha256=last_weights_sha,
            training_mode=cfg.training_mode,
            is_fixture=is_fixture,
        )
        if step % cfg.checkpoint_every == 0 or step == cfg.max_steps:
            save_checkpoint(
                ckpt_dir,
                state,
                model=model,
                optimizer=optimizer,
                adapter_only=adapter_only,
                save_rng=True,
            )

    return state


def run_single_process_full_param_grpo(cfg: ExperimentConfig, *, overwrite: bool = False) -> CheckpointState:
    if cfg.training_mode != "full_param_0.49b":
        raise ValueError("Use rl_trainer.qlora_grpo.run_qlora_7b_grpo for training_mode='qlora_7b_nf4'.")
    cfg.validate(strict_on_policy_sampling=True)

    try:
        import copy
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Single-process real-model GRPO requires `torch` and `transformers`.") from exc

    device = "cuda" if (cfg.backend == "triton" and torch.cuda.is_available()) else "cpu"
    if cfg.backend == "triton" and device != "cuda":
        raise RuntimeError("backend='triton' requires CUDA; refusing to run GPU training on CPU.")

    torch.manual_seed(cfg.seed)
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    model_dtype = dtype_map[cfg.dtype] if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, revision=cfg.model_revision, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, revision=cfg.model_revision, torch_dtype=model_dtype,
        local_files_only=True, trust_remote_code=False,
    ).to(device)
    if cfg.gradient_checkpointing and hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
    _disable_all_dropout(policy)

    ref_device = "cpu" if cfg.offload_ref_to_cpu else device
    ref_model = copy.deepcopy(policy).to(ref_device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    if cfg.cpu_master_adamw:
        optimizer: Any = CpuMasterAdamW(trainable_params, lr=cfg.learning_rate)
    elif cfg.optimizer_name == "adafactor":
        optimizer = torch.optim.Adafactor(trainable_params, lr=cfg.learning_rate, foreach=False)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    print(json.dumps({"event": "model_ready", "training_mode": cfg.training_mode,
                      "parameters": sum(p.numel() for p in policy.parameters()),
                      "trainable_parameters": sum(p.numel() for p in trainable_params),
                      "dtype": cfg.dtype, "optimizer": cfg.optimizer_name,
                      "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}), flush=True)

    def _move_optimizer_state(target: str) -> None:
        if isinstance(optimizer, CpuMasterAdamW):
            return  # Already lives permanently on CPU RAM
        for st in optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(target)

    def _before_grade() -> None:
        if device == "cuda" and cfg.offload_policy_during_grading:
            policy.to("cpu")
            _move_optimizer_state("cpu")
            _release_cuda_memory(torch)

    def _after_grade() -> None:
        if device == "cuda" and cfg.offload_policy_during_grading:
            policy.to(device)

    def _sample(prompt: str, n: int, gen_seed: int) -> list[SampledCandidate]:
        formatted, applied_chat, p_sha = format_prompt_for_tokenizer(prompt, tokenizer)
        enc = tokenizer(formatted, return_tensors="pt")
        prompt_ids = enc["input_ids"][0].tolist()
        input_ids = enc["input_ids"].to(device)
        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        policy.eval()
        _disable_all_dropout(policy)
        policy.config.use_cache = True
        samples: list[SampledCandidate] = []
        with torch.no_grad():
            for idx in range(n):
                torch.manual_seed(gen_seed + idx)
                seq = policy.generate(input_ids=input_ids, generation_config=gen_cfg)
                comp_ids = seq[0][len(prompt_ids) :].tolist()
                raw_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                old_lp = compute_selected_token_logps(policy, prompt_ids, comp_ids, device).cpu().tolist() if comp_ids else []
                ref_lp = compute_selected_token_logps(ref_model, prompt_ids, comp_ids, ref_device).cpu().tolist() if comp_ids else []
                samples.append(
                    SampledCandidate(
                        prompt_ids=prompt_ids,
                        completion_ids=comp_ids,
                        raw_text=raw_text,
                        extracted_source=extract_python_source(raw_text),
                        old_token_logps=old_lp,
                        ref_token_logps=ref_lp,
                        chat_template_applied=applied_chat,
                        prompt_sha256=p_sha,
                    )
                )
        _release_cuda_memory(torch)
        return samples

    def _update(samples: list[SampledCandidate], advantages: list[float]) -> dict[str, Any]:
        policy.train()
        _disable_all_dropout(policy)
        policy.config.use_cache = False
        if device == "cuda" and cfg.offload_policy_during_grading:
            _move_optimizer_state(device)
        metrics = perform_verified_grpo_step(
            policy=policy,
            trainable_params=trainable_params,
            optimizer=optimizer,
            samples=samples,
            advantages=advantages,
            logp_fn=lambda m, p_ids, c_ids: compute_selected_token_logps(m, p_ids, c_ids, device),
            clip_eps=cfg.clip_eps,
            kl_coef=cfg.kl_coef,
            max_grad_norm=cfg.max_grad_norm,
            device=device,
        )
        _release_cuda_memory(torch)
        return metrics

    return run_gpu_training_loop(
        cfg,
        sample_fn=_sample,
        update_fn=_update,
        before_grade_hook=_before_grade,
        after_grade_hook=_after_grade,
        model=policy,
        optimizer=optimizer,
        adapter_only=False,
        overwrite=overwrite,
        is_fixture=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KernelBench-RL On-Policy Clipped GRPO GPU Bridge")
    parser.add_argument("--config", required=True, help="Path to ExperimentConfig JSON file.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_json(Path(args.config))
    if cfg.training_mode == "qlora_7b_nf4":
        from rl_trainer.qlora_grpo import run_qlora_7b_grpo

        state = run_qlora_7b_grpo(cfg, overwrite=args.overwrite)
    else:
        state = run_single_process_full_param_grpo(cfg, overwrite=args.overwrite)
    print(
        f"Completed step {state.step} ({cfg.training_mode}); "
        f"optimizer_steps={state.optimizer_step_count}; rollouts={state.total_rollouts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
