"""7B QLoRA/PEFT on-policy clipped GRPO trainer for free Tesla T4 (14.56 GB VRAM).

Uses:
- Contract-derived prompts + consistent chat-template formatting (`rl_trainer/prompting.py`).
- Chunked `logits_to_keep` selected-token log-probs (`compute_selected_token_logps`) to
  avoid materializing `[seq_len, 152k]` fp32 `log_softmax` tensors on 14.56 GB T4.
- Shared frozen-base reference via `policy.disable_adapter()` (zero second 7B model).
- Finite-gradient verification + `optimizer_step_count` + `weight_l2_delta` +
  `trainable_weights_sha256` proof (`perform_verified_grpo_step`).
- In-place PEFT adapter resume (`rl_trainer/checkpointing.py`) preserving trainable
  parameter object identity and unmodified frozen base reference.
"""

from __future__ import annotations

import json
from typing import Any

from rl_trainer.checkpointing import CheckpointState
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import (
    SampledCandidate,
    _disable_all_dropout,
    _release_cuda_memory,
    compute_selected_token_logps,
    extract_python_source,
    perform_verified_grpo_step,
    run_gpu_training_loop,
)
from rl_trainer.prompting import format_prompt_for_tokenizer


def select_trainable_adapter_params(model: Any) -> list[Any]:
    """Return strictly the trainable LoRA adapter parameters (excluding frozen 4-bit base)."""
    params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    if not params:
        raise RuntimeError("No trainable LoRA adapter parameters found on PEFT model.")
    return params


def compute_shared_base_ref_logps(
    peft_model: Any,
    logp_fn: Any,
    prompt_ids: list[int],
    comp_ids: list[int],
) -> Any:
    """Compute frozen base-model reference log-probs using `disable_adapter()` with zero extra model copy."""
    if not hasattr(peft_model, "disable_adapter"):
        raise RuntimeError("Expected a PEFT model supporting `disable_adapter()` for shared frozen-base reference.")
    with peft_model.disable_adapter():
        return logp_fn(peft_model, prompt_ids, comp_ids)


def run_qlora_7b_grpo(cfg: ExperimentConfig, *, overwrite: bool = False) -> CheckpointState:
    """Run genuine 7B QLoRA on-policy clipped GRPO on a 14–16 GB GPU (e.g., free Colab Tesla T4)."""
    if cfg.training_mode != "qlora_7b_nf4":
        raise ValueError(f"run_qlora_7b_grpo requires training_mode='qlora_7b_nf4', got '{cfg.training_mode}'.")
    cfg.validate(strict_on_policy_sampling=True)

    try:
        import torch  # type: ignore
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            GenerationConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "7B QLoRA mode requires `torch`, `transformers`, `peft`, and `bitsandbytes`."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("7B QLoRA NF4 training requires a CUDA GPU (e.g., Tesla T4).")

    torch.manual_seed(cfg.seed)
    compute_dtype = torch.float16 if cfg.dtype == "fp16" else (torch.bfloat16 if cfg.dtype == "bf16" else torch.float32)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, revision=cfg.model_revision, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        revision=cfg.model_revision,
        quantization_config=bnb_config,
        device_map={"": 0}, local_files_only=True, trust_remote_code=False,
    )
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base_model, lora_cfg)
    _disable_all_dropout(policy)

    trainable_params = select_trainable_adapter_params(policy)
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    device = next(policy.parameters()).device
    print(json.dumps({"event": "model_ready", "training_mode": cfg.training_mode,
        "trainable_parameters": sum(p.numel() for p in trainable_params),
        "gpu": torch.cuda.get_device_name(0), "compute_dtype": str(compute_dtype)}), flush=True)

    def _logp_fn(model_obj: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
        return compute_selected_token_logps(model_obj, prompt_ids, comp_ids, device)

    def _before_grade() -> None:
        _release_cuda_memory(torch)

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
                old_lp = _logp_fn(policy, prompt_ids, comp_ids).cpu().tolist() if comp_ids else []
                ref_lp = (
                    compute_shared_base_ref_logps(policy, _logp_fn, prompt_ids, comp_ids).cpu().tolist()
                    if comp_ids
                    else []
                )
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
        metrics = perform_verified_grpo_step(
            policy=policy,
            trainable_params=trainable_params,
            optimizer=optimizer,
            samples=samples,
            advantages=advantages,
            logp_fn=_logp_fn,
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
        model=policy,
        optimizer=optimizer,
        adapter_only=True,
        overwrite=overwrite,
        is_fixture=False,
    )
