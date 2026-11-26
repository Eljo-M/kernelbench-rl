"""Held-out evaluation harness with sequential load/unload and leak-free Best-of-K selection.

Memory & statistical rigor:
1. Never loads base and trained models simultaneously: sequentially loads model A,
   generates & caches all completions, unloads model A (`del` + `gc.collect()` +
   `torch.cuda.empty_cache()`), repeats for model B, and only then grades all
   cached completions with NO model resident in VRAM.
2. Leak-free Best-of-K (`selection_mode="validation_seed_regrade"` default):
   grades all K candidates on a separate validation seed (`eval_seed + selection_seed_offset`),
   picks the winner on the validation seed, and regrades ONLY that winner on the
   independent held-out test seed (`eval_seed`). If `selection_mode="oracle_same_seed"`
   is explicitly requested, every record and summary is labeled `"oracle_same_seed"`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from rl_trainer.checkpointing import load_checkpoint, resolve_checkpoint_dir
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import extract_python_source
from rl_trainer.prompting import build_prompt, format_prompt_for_tokenizer, sha256_text
from sandbox_runner.gpu_verifier import grade_candidate

SelectionMode = Literal["validation_seed_regrade", "oracle_same_seed"]


def generate_cached_completions(
    cfg: ExperimentConfig,
    *,
    method_budgets: Sequence[tuple[str, int]],
    generate_fn: Callable[[str, int, int], list[str]],
    chat_template_applied: bool = False,
    cache_path: Path | None = None,
    model_identity: str = "",
    prompt_formatter: Callable[[str], str] | None = None,
) -> dict[str, list[tuple[str, int, str, list[str]]]]:
    """Generate and cache all completions for one model before unloading it from VRAM."""
    saved = {}
    if cache_path is not None and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("config_hash") != cfg.config_hash() or row.get("model_identity") != model_identity:
                raise ValueError(f"Incompatible generation cache: {cache_path}")
            key = (row["method"], row["op_name"], row["sample_k"])
            if key in saved:
                raise ValueError(f"Duplicate generation cache entry: {key}")
            saved[key] = row
    cached: dict[str, list[tuple[str, int, str, list[str]]]] = {}
    for method_name, num_samples in method_budgets:
        entries: list[tuple[str, int, str, list[str]]] = []
        for idx, op_name in enumerate(cfg.heldout_ops + cfg.development_eval_ops):
            if op_name in cfg.development_eval_ops and num_samples > 1:
                continue  # Inspected regressions get pass@1 only; reserve search budget for fresh tests.
            prompt = build_prompt(op_name, backend=cfg.backend, prompt_version=cfg.prompt_version, float_dtype=cfg.dtype)
            eval_seed = cfg.seed + cfg.heldout_seed_offset + idx
            prompt_sha = sha256_text(prompt_formatter(prompt) if prompt_formatter else prompt)
            if cache_path is None:
                completions = generate_fn(prompt, num_samples, eval_seed)
            else:
                completions = []
                for sample_k in range(num_samples):
                    key = (method_name, op_name, sample_k)
                    row = saved.get(key)
                    if row is not None:
                        if row["prompt_sha256"] != prompt_sha or row["generation_seed"] != eval_seed + sample_k:
                            raise ValueError(f"Prompt/seed changed for cached completion {key}")
                        completion = row["raw_completion"]
                    else:
                        generated = generate_fn(prompt, 1, eval_seed + sample_k)
                        if len(generated) != 1 or not isinstance(generated[0], str):
                            raise ValueError("Sequential sampler must return one text completion")
                        completion = generated[0]
                        row = dict(method=method_name, op_name=op_name, sample_k=sample_k,
                            config_hash=cfg.config_hash(), model_identity=model_identity,
                            prompt_sha256=prompt_sha, generation_seed=eval_seed + sample_k,
                            raw_completion=completion)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with cache_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row, allow_nan=False) + "\n")
                    completions.append(completion)
                    print(json.dumps({"event": "heldout_generation", "method": method_name,
                        "op_name": op_name, "sample_k": sample_k, "cached": key in saved}), flush=True)
            if len(completions) != num_samples:
                raise ValueError(f"{method_name} returned {len(completions)} samples for {op_name}, expected {num_samples}")
            entries.append((op_name, eval_seed, prompt_sha, [extract_python_source(c) for c in completions]))
        cached[method_name] = entries
    return cached


def grade_cached_heldout_completions(
    cfg: ExperimentConfig,
    cached_by_method: dict[str, list[tuple[str, int, str, list[str]]]],
    *,
    grader_fn: Callable[..., dict[str, Any]] = grade_candidate,
    selection_mode: SelectionMode = "validation_seed_regrade",
    out_jsonl: Path | None = None,
    overwrite: bool = False,
    chat_template_applied: bool = False,
    is_fixture: bool = False,
    checkpoint_identity: str = "",
    generation_engine: str = "external_callback",
) -> dict[str, dict[str, Any]]:
    """Grade cached completions with NO model loaded, using leak-free validation-seed Best-of-K by default."""
    cfg.validate(strict_on_policy_sampling=False)
    if selection_mode not in {"validation_seed_regrade", "oracle_same_seed"}:
        raise ValueError(f"Unsupported selection_mode '{selection_mode}'.")
    if out_jsonl is not None and out_jsonl.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {out_jsonl}. Pass --overwrite to replace.")

    all_records: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    def _annotate(
        rec: dict[str, Any],
        *,
        src: str,
        p_sha: str,
        k_idx: int,
        num_samples: int,
        eval_phase: str,
        sel_method: str,
    ) -> dict[str, Any]:
        rec["candidate_source"] = src
        rec["candidate_sha256"] = sha256_text(src)
        rec["chat_template_applied"] = bool(chat_template_applied)
        rec["prompt_sha256"] = p_sha
        rec["policy_updated"] = False
        development = rec["op_name"] in cfg.development_eval_ops
        rec["split"] = "development" if development else "heldout"
        rec["heldout"] = not development
        rec["eval_phase"] = eval_phase.replace("heldout_test", "development_test").replace("selection_val", "development_selection") if development else eval_phase
        rec["evaluation_cohort"] = "previously_inspected_development" if development else "fresh_heldout"
        rec["selection_method"] = sel_method
        rec["sample_k"] = int(k_idx)
        rec["budget_k"] = int(num_samples)
        rec["model_name_or_path"] = cfg.model_name_or_path
        rec["model_revision"] = cfg.model_revision
        rec["training_mode"] = cfg.training_mode
        rec["sampling_config"] = cfg.sampling_config_dict()
        rec["config_hash"] = cfg.config_hash()
        rec["fixture"] = bool(is_fixture)
        rec["generation_engine"] = generation_engine
        rec["checkpoint_identity"] = checkpoint_identity if rec.get("method") in {"trained_checkpoint", "sft_only_checkpoint"} else "base:" + cfg.model_revision
        print(json.dumps({"event": "heldout_grade", "method": rec.get("method"),
                          "op_name": rec["op_name"], "phase": rec["eval_phase"], "status": rec["status"]}), flush=True)
        return rec

    for method_name, op_entries in cached_by_method.items():
        final_heldout_recs: list[dict[str, Any]] = []
        used_sel_method = "single_sample"

        for op_name, eval_seed, p_sha, sources in op_entries:
            num_samples = len(sources)
            if num_samples == 1:
                rec = grader_fn(
                    op_name,
                    sources[0],
                    backend=cfg.backend,
                    float_dtype=cfg.dtype,
                    seed=eval_seed,
                    independent_seed=eval_seed + cfg.independent_seed_offset,
                    warmup=cfg.warmup_iters,
                    iters=cfg.bench_iters,
                    timeout_secs=cfg.worker_timeout_secs,
                    run_id=f"{method_name}-{op_name}-k0",
                    method=method_name,
                )
                _annotate(
                    rec,
                    src=sources[0],
                    p_sha=p_sha,
                    k_idx=0,
                    num_samples=1,
                    eval_phase="heldout_test",
                    sel_method="single_sample",
                )
                all_records.append(rec)
                final_heldout_recs.append(rec)
            elif selection_mode == "validation_seed_regrade":
                used_sel_method = "validation_seed_regrade"
                val_seed = eval_seed + cfg.selection_seed_offset
                val_candidates: list[tuple[int, str, dict[str, Any]]] = []
                for k_idx, src in enumerate(sources):
                    vrec = grader_fn(
                        op_name,
                        src,
                        backend=cfg.backend,
                        float_dtype=cfg.dtype,
                        seed=val_seed,
                        independent_seed=val_seed + cfg.independent_seed_offset,
                        warmup=cfg.warmup_iters,
                        iters=cfg.bench_iters,
                        timeout_secs=cfg.worker_timeout_secs,
                        run_id=f"{method_name}-{op_name}-val-k{k_idx}",
                        method=method_name,
                    )
                    _annotate(
                        vrec,
                        src=src,
                        p_sha=p_sha,
                        k_idx=k_idx,
                        num_samples=num_samples,
                        eval_phase="selection_val",
                        sel_method=used_sel_method,
                    )
                    all_records.append(vrec)
                    val_candidates.append((k_idx, src, vrec))

                # Select winner strictly on validation seed, then regrade ONLY winner on independent test seed
                best_k, best_src, _ = max(val_candidates, key=lambda item: float(item[2].get("reward", 0.0)))
                test_rec = grader_fn(
                    op_name,
                    best_src,
                    backend=cfg.backend,
                    float_dtype=cfg.dtype,
                    seed=eval_seed,
                    independent_seed=eval_seed + cfg.independent_seed_offset,
                    warmup=cfg.warmup_iters,
                    iters=cfg.bench_iters,
                    timeout_secs=cfg.worker_timeout_secs,
                    run_id=f"{method_name}-{op_name}-test-selected-k{best_k}",
                    method=method_name,
                )
                _annotate(
                    test_rec,
                    src=best_src,
                    p_sha=p_sha,
                    k_idx=best_k,
                    num_samples=num_samples,
                    eval_phase="heldout_test",
                    sel_method=used_sel_method,
                )
                all_records.append(test_rec)
                final_heldout_recs.append(test_rec)
            else:
                used_sel_method = "oracle_same_seed"
                cand_recs: list[dict[str, Any]] = []
                for k_idx, src in enumerate(sources):
                    rec = grader_fn(
                        op_name,
                        src,
                        backend=cfg.backend,
                        float_dtype=cfg.dtype,
                        seed=eval_seed,
                        independent_seed=eval_seed + cfg.independent_seed_offset,
                        warmup=cfg.warmup_iters,
                        iters=cfg.bench_iters,
                        timeout_secs=cfg.worker_timeout_secs,
                        run_id=f"{method_name}-{op_name}-oracle-k{k_idx}",
                        method=method_name,
                    )
                    _annotate(
                        rec,
                        src=src,
                        p_sha=p_sha,
                        k_idx=k_idx,
                        num_samples=num_samples,
                        eval_phase="heldout_test_oracle",
                        sel_method=used_sel_method,
                    )
                    all_records.append(rec)
                    cand_recs.append(rec)
                final_heldout_recs.append(max(cand_recs, key=lambda r: float(r.get("reward", 0.0))))

        development_recs = [r for r in final_heldout_recs if not r["heldout"]]
        final_heldout_recs = [r for r in final_heldout_recs if r["heldout"]]
        n = len(final_heldout_recs)
        passed = sum(1 for r in final_heldout_recs if r.get("status") == "PASS")
        fast_eager = sum(1 for r in final_heldout_recs if r.get("status") == "PASS" and float(r.get("speedup_vs_eager", 0.0)) > 1.0)
        fast_compile = sum(
            1 for r in final_heldout_recs if r.get("status") == "PASS" and float(r.get("speedup_vs_compile", 0.0)) > 1.0
        )
        speedups = [
            float(r["speedup_vs_eager"])
            for r in final_heldout_recs
            if r.get("status") == "PASS" and float(r.get("speedup_vs_eager", 0.0)) > 0.0
        ]
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else 0.0
        summaries[method_name] = {
            "ops": float(n),
            "development_passed": sum(r.get("status") == "PASS" for r in development_recs),
            "development_total": len(development_recs),
            "selection_method": used_sel_method,
            "pass_rate": round(100.0 * passed / max(n, 1), 2),
            "fast_1x_rate": round(100.0 * fast_eager / max(n, 1), 2),
            "fast_compile_rate": round(100.0 * fast_compile / max(n, 1), 2),
            "geomean_speedup_vs_eager": round(geomean, 4),
            "mean_reward": round(sum(float(r.get("reward", 0.0)) for r in final_heldout_recs) / max(n, 1), 4),
        }

    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for rec in all_records:
                fh.write(json.dumps(rec, allow_nan=False) + "\n")

    return summaries


def evaluate_heldout_methods(
    cfg: ExperimentConfig,
    *,
    base_generate_fn: Callable[[str, int, int], list[str]],
    trained_generate_fn: Callable[[str, int, int], list[str]] | None = None,
    grader_fn: Callable[..., dict[str, Any]] = grade_candidate,
    best_of_k: int | None = None,
    selection_mode: SelectionMode = "validation_seed_regrade",
    out_jsonl: Path | None = None,
    overwrite: bool = False,
    chat_template_applied: bool = False,
    is_fixture: bool = False,
) -> dict[str, dict[str, Any]]:
    """Backward-compatible wrapper: generates & caches each model's outputs sequentially, then grades."""
    k_budget = cfg.heldout_best_of_k if best_of_k is None else best_of_k
    cached = generate_cached_completions(
        cfg,
        method_budgets=[("base_pass1", 1), (f"base_best_of_{k_budget}", k_budget)],
        generate_fn=base_generate_fn,
        chat_template_applied=chat_template_applied,
    )
    if trained_generate_fn is not None:
        cached.update(
            generate_cached_completions(
                cfg,
                method_budgets=[("trained_checkpoint", 1)],
                generate_fn=trained_generate_fn,
                chat_template_applied=chat_template_applied,
            )
        )
    return grade_cached_heldout_completions(
        cfg,
        cached,
        grader_fn=grader_fn,
        selection_mode=selection_mode,
        out_jsonl=out_jsonl,
        overwrite=overwrite,
        chat_template_applied=chat_template_applied,
        is_fixture=is_fixture,
    )


def _generate_and_unload_hf_model(
    cfg: ExperimentConfig,
    method_budgets: Sequence[tuple[str, int]],
    *,
    checkpoint_dir: str = "",
    cache_path: Path | None = None,
) -> tuple[dict[str, list[tuple[str, int, str, list[str]]]], bool]:
    """Load ONE model, generate & cache all requested completions, then completely unload VRAM."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # type: ignore

    device = "cuda" if (cfg.backend == "triton" and torch.cuda.is_available()) else "cpu"
    if cfg.backend == "triton" and device != "cuda":
        raise RuntimeError("backend='triton' requires CUDA for held-out evaluation.")
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    model_dtype = dtype_map[cfg.dtype] if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, revision=cfg.model_revision,
                                            local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _, has_chat_template, _ = format_prompt_for_tokenizer("ping", tokenizer)
    model_identity = "base:" + cfg.model_revision
    if checkpoint_dir:
        state = load_checkpoint(checkpoint_dir, expected_config_hash=cfg.config_hash(), restore_rng=False)
        model_identity = f"checkpoint:{state.step}:{state.trainable_weights_sha256}"
    formatter = lambda prompt: format_prompt_for_tokenizer(prompt, tokenizer)[0]
    class CacheMiss(Exception):
        pass
    def missing(*args):
        raise CacheMiss()
    if cache_path is not None and cache_path.exists():
        try:
            cached = generate_cached_completions(cfg, method_budgets=method_budgets,
                generate_fn=missing, cache_path=cache_path, model_identity=model_identity,
                prompt_formatter=formatter, chat_template_applied=has_chat_template)
            return cached, has_chat_template
        except CacheMiss:
            pass

    model: Any = None
    base: Any = None
    try:
        if cfg.training_mode == "qlora_7b_nf4":
            from peft import PeftModel, prepare_model_for_kbit_training  # type: ignore
            from transformers import BitsAndBytesConfig  # type: ignore

            from rl_trainer.model_loading import load_quantized_base
            base = load_quantized_base(cfg, gradient_checkpointing=False)
            if checkpoint_dir:
                step_dir = resolve_checkpoint_dir(checkpoint_dir)
                state = load_checkpoint(step_dir, expected_config_hash=cfg.config_hash(), restore_rng=False)
                if state.adapter_dir:
                    model = PeftModel.from_pretrained(base, str(step_dir / state.adapter_dir))
                else:
                    model = base
                    load_checkpoint(step_dir, expected_config_hash=cfg.config_hash(), model=model, restore_rng=False)
            else:
                model = base
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path, revision=cfg.model_revision, torch_dtype=model_dtype,
                local_files_only=True, trust_remote_code=False,
            ).to(device)
            if checkpoint_dir:
                load_checkpoint(
                    checkpoint_dir,
                    expected_config_hash=cfg.config_hash(),
                    model=model,
                    restore_rng=False,
                    map_location=device,
                )
        model.eval()
        run_dev = next(model.parameters()).device

        def _gen(prompt: str, n: int, gen_seed: int) -> list[str]:
            formatted, _, _ = format_prompt_for_tokenizer(prompt, tokenizer)
            enc = tokenizer(formatted, return_tensors="pt").to(run_dev)
            p_len = enc["input_ids"].shape[1]
            gen_cfg = GenerationConfig(
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs: list[str] = []
            with torch.no_grad():
                for idx in range(n):
                    torch.manual_seed(gen_seed + idx)
                    seq = model.generate(input_ids=enc["input_ids"], generation_config=gen_cfg)
                    outputs.append(tokenizer.decode(seq[0][p_len:], skip_special_tokens=True))
            return outputs

        cached = generate_cached_completions(
            cfg,
            method_budgets=method_budgets,
            generate_fn=_gen,
            chat_template_applied=has_chat_template,
            cache_path=cache_path, model_identity=model_identity, prompt_formatter=formatter,
        )
        return cached, has_chat_template
    finally:
        # Unload model completely before loading next model or running GPU grading worker
        del model
        if cfg.training_mode == "qlora_7b_nf4":
            del base
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential load/unload held-out evaluation with leak-free Best-of-K.")
    parser.add_argument("--config", required=True, help="Path to ExperimentConfig JSON.")
    parser.add_argument("--checkpoint-dir", default="", help="Optional trained checkpoint directory (full or QLoRA adapter).")
    parser.add_argument("--best-of-k", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["validation_seed_regrade", "oracle_same_seed"],
        default="validation_seed_regrade",
    )
    parser.add_argument("--out", required=True, help="Output strict JSONL path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.out).exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.out}")

    cfg = ExperimentConfig.from_json(Path(args.config))
    checkpoint_identity = ""
    if args.checkpoint_dir:
        state = load_checkpoint(args.checkpoint_dir, expected_config_hash=cfg.config_hash(), restore_rng=False)
        checkpoint_identity = f"checkpoint:{state.step}:{state.trainable_weights_sha256}"
    k_budget = cfg.heldout_best_of_k if args.best_of_k is None else args.best_of_k

    # Phase 1: Load base -> generate & cache -> unload base
    cached, applied_chat = _generate_and_unload_hf_model(
        cfg,
        [("base_pass1", 1), (f"base_best_of_{k_budget}", k_budget)],
        cache_path=Path(args.out).with_suffix(".base.generations.jsonl"),
    )
    # Phase 2: Load trained -> generate & cache -> unload trained (never concurrent with base!)
    if args.checkpoint_dir:
        trained_cached, _ = _generate_and_unload_hf_model(
            cfg,
            [("trained_checkpoint", 1)],
            checkpoint_dir=args.checkpoint_dir,
            cache_path=Path(args.out).with_suffix(".trained.generations.jsonl"),
        )
        cached.update(trained_cached)

    # Phase 3: Grade all cached completions with NO model loaded in VRAM
    summaries = grade_cached_heldout_completions(
        cfg,
        cached,
        selection_mode=args.selection_mode,
        out_jsonl=Path(args.out),
        overwrite=args.overwrite,
        chat_template_applied=applied_chat,
        is_fixture=False,
        checkpoint_identity=checkpoint_identity,
        generation_engine="hf_eager",
    )
    print(json.dumps(summaries, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
