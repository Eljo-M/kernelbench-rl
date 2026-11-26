"""Leak-free supervised warmup bootstrap over verified handwritten Triton baselines.

Trains strictly on `cfg.train_ops` (never `cfg.heldout_ops`) using contract-derived
prompts formatted with the tokenizer's chat template and completion-only NLL loss
computed via `rl_trainer.gpu_train_grpo.compute_selected_token_logps(..., use_cache=False)`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from kernels.triton_baselines import TRITON_BASELINES
from rl_trainer.gpu_train_grpo import _disable_all_dropout, compute_selected_token_logps
from rl_trainer.cpu_master_adamw import CpuMasterAdamW
from rl_trainer.prompting import build_prompt, format_prompt_for_tokenizer, sha256_text
from sandbox_runner.gpu_source_policy import check_gpu_source_policy


@dataclass(frozen=True)
class SupervisedExample:
    op_name: str
    prompt_text: str
    formatted_prompt: str
    target_completion: str
    prompt_ids: list[int]
    completion_ids: list[int]
    prompt_sha256: str
    source_sha256: str
    chat_template_applied: bool


def load_verified_triton_sources() -> dict[str, str]:
    """Derive the verified handwritten Triton source dictionary from current kernels modules."""
    sources: dict[str, str] = dict(TRITON_BASELINES)
    try:
        from kernels.triton_extended import TRITON_EXTENDED

        for op_name, src in TRITON_EXTENDED.items():
            if op_name not in sources:
                sources[op_name] = src
    except ImportError:
        pass
    return sources


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
    elif callable(tokenizer):
        out = tokenizer(text, add_special_tokens=False)
        ids = out["input_ids"] if isinstance(out, Mapping) else out
    else:
        raise TypeError("Tokenizer must implement `.encode(text)` or be callable.")

    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(tok) for tok in ids]


def _trainable_params_checksum(trainable_params: Sequence[Any]) -> str:
    hasher = hashlib.sha256()
    for p in trainable_params:
        data = p.detach().float().cpu().contiguous()
        if hasattr(data, "numpy"):
            hasher.update(data.numpy().tobytes())
        else:
            hasher.update(bytes(str(data), "utf-8"))
    return hasher.hexdigest()


def _snapshot_trainable_params(trainable_params: Sequence[Any]) -> list[Any]:
    return [p.detach().float().cpu().clone() for p in trainable_params]


def _compute_l2_delta(before_snaps: Sequence[Any], after_params: Sequence[Any]) -> float:
    import torch  # type: ignore

    sq_sum = 0.0
    for before, after in zip(before_snaps, after_params):
        diff = after.detach().float().cpu() - before
        sq_sum += float(torch.sum(diff * diff).item())
    return float(math.sqrt(max(sq_sum, 0.0)))


def build_supervised_dataset(
    cfg: Any,
    tokenizer: Any,
    *,
    source_map: Mapping[str, str] | None = None,
    max_tokens: int | None = None,
) -> list[SupervisedExample]:
    """Build and validate a train-only supervised dataset from verified Triton baseline sources.

    Enforces:
    - Non-empty, deduplicated `cfg.train_ops` strictly disjoint from `cfg.heldout_ops`
    - Never accesses any `cfg.heldout_ops` key in `source_map`
    - Every training source passes `sandbox_runner.gpu_source_policy.check_gpu_source_policy`
    - Exact target format: fenced ```python ... ``` block + EOS token
    - Explicit rejection of overlength samples (never truncates source code)
    """
    train_ops = list(getattr(cfg, "train_ops", []) or [])
    heldout_ops = list(getattr(cfg, "heldout_ops", []) or []) + list(getattr(cfg, "development_eval_ops", []) or [])
    if not train_ops:
        raise ValueError("cfg.train_ops must be non-empty for supervised warmup.")
    if len(set(train_ops)) != len(train_ops):
        raise ValueError("cfg.train_ops contains duplicate operator names.")

    overlap = sorted(set(train_ops) & set(heldout_ops))
    if overlap:
        raise ValueError(f"Data leakage: cfg.train_ops and cfg.heldout_ops overlap on {overlap}.")

    sources = load_verified_triton_sources() if source_map is None else source_map
    backend = str(getattr(cfg, "backend", "triton"))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("A completion EOS token is required for supervised warmup")

    examples: list[SupervisedExample] = []
    for op_name in train_ops:
        if op_name not in sources:
            raise KeyError(f"Verified baseline source not found for training operator '{op_name}'.")
        raw_source = sources[op_name]
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise ValueError(f"Empty baseline source for training operator '{op_name}'.")

        policy_res = check_gpu_source_policy(raw_source)
        if not policy_res.allowed:
            raise ValueError(
                f"Training baseline source for '{op_name}' failed strict GPU source policy: {policy_res.violations}"
            )

        prompt_text = build_prompt(op_name, backend=backend,
            prompt_version=getattr(cfg, "prompt_version", "contract_v1"),
            float_dtype=getattr(cfg, "dtype", "fp32"))
        formatted_prompt, chat_applied, _ = format_prompt_for_tokenizer(prompt_text, tokenizer)
        target_completion = f"```python\n{raw_source.strip()}\n```"

        prompt_ids = _encode_ids(tokenizer, formatted_prompt)
        comp_ids = _encode_ids(tokenizer, target_completion)
        if not prompt_ids:
            raise ValueError(f"Empty tokenized prompt for training operator '{op_name}'.")
        if not comp_ids:
            raise ValueError(f"Empty tokenized completion for training operator '{op_name}'.")

        if eos_id is not None and comp_ids[-1] != int(eos_id):
            comp_ids.append(int(eos_id))

        total_tokens = len(prompt_ids) + len(comp_ids)
        if max_tokens is not None and max_tokens > 0 and total_tokens > int(max_tokens):
            raise ValueError(
                f"Overlength supervised example for '{op_name}': total_tokens={total_tokens} "
                f"(prompt={len(prompt_ids)}, completion={len(comp_ids)}) > sft_max_tokens={int(max_tokens)}. "
                "Refusing to truncate source."
            )

        examples.append(
            SupervisedExample(
                op_name=op_name,
                prompt_text=prompt_text,
                formatted_prompt=formatted_prompt,
                target_completion=target_completion,
                prompt_ids=prompt_ids,
                completion_ids=comp_ids,
                prompt_sha256=sha256_text(formatted_prompt),
                source_sha256=sha256_text(raw_source.strip()),
                chat_template_applied=bool(chat_applied),
            )
        )

    return examples


def run_supervised_warmup(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    optimizer: Any,
    *,
    device: Any,
    trainable_params: Sequence[Any],
    logp_fn: Callable[[Any, list[int], list[int]], Any] | None = None,
    source_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run deterministic train-only supervised warmup with completion-only NLL loss.

    - Returns a clean no-op metrics dict when `cfg.sft_epochs` is absent or <= 0.
    - Does not mutate `cfg` or `model.config.use_cache`.
    - Respects the supplied optimizer and restores its original param-group LRs afterward.
    - Performs bounded per-example backward, checks finite loss and gradients, and clips
      gradients before every optimizer step.
    """
    raw_epochs = getattr(cfg, "sft_epochs", 0)
    sft_epochs = 0 if raw_epochs is None else int(raw_epochs)
    if sft_epochs < 0:
        raise ValueError(f"sft_epochs must be >= 0, got {sft_epochs}.")

    trainable_list = list(trainable_params)
    before_sha = _trainable_params_checksum(trainable_list) if trainable_list else ""

    if sft_epochs == 0:
        return {
            "ran": False,
            "sft_epochs": 0,
            "optimizer_step_count": 0,
            "total_examples": 0,
            "total_completion_tokens": 0,
            "mean_loss": None,
            "weight_sha256_before": before_sha,
            "weight_sha256_after": before_sha,
            "weight_l2_delta": 0.0,
            "records": [],
        }

    import torch  # type: ignore

    if not trainable_list:
        raise ValueError("trainable_params must be non-empty when sft_epochs > 0.")

    raw_max_tokens = getattr(cfg, "sft_max_tokens", None)
    sft_max_tokens = None if raw_max_tokens is None else int(raw_max_tokens)
    raw_sft_lr = getattr(cfg, "sft_learning_rate", None)
    sft_lr = None if raw_sft_lr is None else float(raw_sft_lr)
    if sft_lr is not None and (not math.isfinite(sft_lr) or sft_lr <= 0.0):
        raise ValueError(f"sft_learning_rate must be finite > 0, got {sft_lr}.")

    max_grad_norm = float(getattr(cfg, "max_grad_norm", 1.0))
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise ValueError(f"max_grad_norm must be finite > 0, got {max_grad_norm}.")

    sft_out_raw = getattr(cfg, "sft_output_jsonl", "") or ""
    sft_output_jsonl = Path(sft_out_raw) if str(sft_out_raw).strip() else None
    if sft_output_jsonl is not None and sft_output_jsonl.exists():
        raise FileExistsError(f"Preserving existing supervised log: {sft_output_jsonl}")

    dataset = build_supervised_dataset(cfg, tokenizer, source_map=source_map, max_tokens=sft_max_tokens)
    generation_budget = getattr(cfg, "max_new_tokens", None)
    if generation_budget is not None and any(len(ex.completion_ids) > generation_budget for ex in dataset):
        raise ValueError("A supervised completion exceeds the configured generation token budget")

    active_logp_fn: Callable[[Any, list[int], list[int]], Any]
    if logp_fn is not None:
        active_logp_fn = logp_fn
    else:
        active_logp_fn = lambda m, p_ids, c_ids: compute_selected_token_logps(
            m, p_ids, c_ids, device, use_cache=False
        )

    before_snaps = _snapshot_trainable_params(trainable_list)
    had_cfg_cache = hasattr(model, "config") and hasattr(model.config, "use_cache")
    orig_cfg_cache = getattr(model.config, "use_cache", None) if had_cfg_cache else None
    orig_training = bool(getattr(model, "training", False))
    master_optimizer = isinstance(optimizer, CpuMasterAdamW)
    param_groups = optimizer.optimizer.param_groups if master_optimizer else optimizer.param_groups
    orig_lrs = [pg.get("lr") for pg in param_groups]

    if sft_lr is not None:
        for pg in param_groups:
            pg["lr"] = sft_lr

    records: list[dict[str, Any]] = []
    optimizer_step_count = 0
    total_completion_tokens = 0
    loss_sum = 0.0
    seed = int(getattr(cfg, "seed", 42))
    stream = None
    try:
        if sft_output_jsonl is not None:
            sft_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            stream = sft_output_jsonl.open("x", encoding="utf-8")
        if hasattr(model, "train"):
            model.train()
        _disable_all_dropout(model)

        for epoch in range(1, sft_epochs + 1):
            order = list(range(len(dataset)))
            rng = random.Random(seed + 1000 * epoch)
            rng.shuffle(order)

            for ex_idx in order:
                ex = dataset[ex_idx]
                optimizer.zero_grad(set_to_none=True)

                selected_logps = active_logp_fn(model, ex.prompt_ids, ex.completion_ids)
                if selected_logps.ndim != 1 or int(selected_logps.shape[0]) != len(ex.completion_ids):
                    raise RuntimeError(
                        f"Expected 1D completion logps of length {len(ex.completion_ids)}, "
                        f"got shape {tuple(selected_logps.shape)}."
                    )
                if not bool(torch.isfinite(selected_logps).all().item()):
                    raise RuntimeError(f"Non-finite completion logps encountered for '{ex.op_name}' in epoch {epoch}.")

                loss = -selected_logps.mean()
                loss_val = float(loss.detach().item())
                if not math.isfinite(loss_val):
                    raise RuntimeError(f"Non-finite SFT loss ({loss_val}) for '{ex.op_name}' in epoch {epoch}.")

                loss.backward()

                if not any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in trainable_list):
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError("No nonzero training gradient; refusing to count an optimizer update")

                for p in trainable_list:
                    if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
                        optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(
                            f"Non-finite gradient detected before clipping for '{ex.op_name}' in epoch {epoch}."
                        )

                grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable_list, max_grad_norm)
                grad_norm = float(grad_norm_t.item() if hasattr(grad_norm_t, "item") else grad_norm_t)
                if not math.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(f"Non-finite gradient norm ({grad_norm}) for '{ex.op_name}' in epoch {epoch}.")

                for p in trainable_list:
                    if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
                        optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(
                            f"Non-finite gradient detected after clipping for '{ex.op_name}' in epoch {epoch}."
                        )

                if master_optimizer:
                    updated, _ = optimizer.copy_grads_and_step(max_grad_norm)
                    if not updated:
                        raise RuntimeError("CPU master optimizer rejected the supervised update")
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1
                total_completion_tokens += len(ex.completion_ids)
                loss_sum += loss_val

                active_lr = float(param_groups[0].get("lr", 0.0)) if param_groups else 0.0
                records.append(
                    {
                        "epoch": int(epoch),
                        "step": int(optimizer_step_count),
                        "op_name": ex.op_name,
                        "prompt_tokens": len(ex.prompt_ids),
                        "completion_tokens": len(ex.completion_ids),
                        "loss": round(loss_val, 6),
                        "grad_norm": round(grad_norm, 6),
                        "learning_rate": active_lr,
                        "prompt_sha256": ex.prompt_sha256,
                        "source_sha256": ex.source_sha256,
                        "chat_template_applied": ex.chat_template_applied,
                    }
                )
                if stream is not None:
                    stream.write(json.dumps(records[-1], allow_nan=False) + "\n")
                    stream.flush()
                print(json.dumps({"event": "supervised_step", **records[-1]}, allow_nan=False), flush=True)
    finally:
        if stream is not None:
            stream.close()
        for pg, orig_lr in zip(param_groups, orig_lrs):
            if orig_lr is None:
                pg.pop("lr", None)
            else:
                pg["lr"] = orig_lr
        if had_cfg_cache:
            model.config.use_cache = orig_cfg_cache
        if hasattr(model, "train"):
            model.train(orig_training)
            _disable_all_dropout(model)

    after_sha = _trainable_params_checksum(trainable_list)
    l2_delta = _compute_l2_delta(before_snaps, trainable_list)
    mean_loss = round(loss_sum / max(len(records), 1), 6) if records else None

    return {
        "ran": True,
        "sft_epochs": sft_epochs,
        "optimizer_step_count": optimizer_step_count,
        "total_examples": len(records),
        "total_completion_tokens": total_completion_tokens,
        "mean_loss": mean_loss,
        "weight_sha256_before": before_sha,
        "weight_sha256_after": after_sha,
        "weight_l2_delta": round(l2_delta, 8),
        "records": records,
    }
