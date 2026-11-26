"""Run a recorded local/free-GPU study with explicit, identity-checked stage resume."""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import uuid

from rl_trainer.checkpointing import load_checkpoint
from rl_trainer.experiment_config import ExperimentConfig

IDENTITY_KEYS = ("config_hash", "model_id", "model_revision", "source_commit", "training_schedule",
                 "checkpoint_root", "train_output")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive(path, *, move=False):
    archive = path.with_name(path.name + ".attempt-" + uuid.uuid4().hex)
    if move:
        path.rename(archive)
    else:
        with archive.open("xb") as stream:
            stream.write(path.read_bytes())
    return archive.name


def _require_identity(previous, expected, context):
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ValueError(f"{context}: identity mismatch for {key}; preserve the attempt and use its original configuration/source")


def _read_stage(path, expected):
    if not path.exists():
        return None
    previous = json.loads(path.read_text(encoding="utf-8"))
    _require_identity(previous, expected, str(path))
    return previous


def _start_stage(path, expected, previous):
    history = list(previous.get("prior_attempts", [])) if previous else []
    if previous:
        history.append(_archive(path))
    stage = dict(expected, started_utc=now(), prior_attempts=history)
    write(path, stage)
    return stage


def _complete_output_is_valid(path, stage):
    if not path.is_file() or not stage.get("output_sha256"):
        raise ValueError(f"Completed stage lacks its recorded output: {path}")
    if file_sha256(path) != stage["output_sha256"]:
        raise ValueError(f"Completed stage output changed: {path}")


def _checkpoint_identity(state):
    if not state.trainable_weights_sha256:
        raise ValueError("A durable study checkpoint requires a nonempty trainable-weight checksum")
    return dict(checkpoint_step=state.step, checkpoint_weights_sha256=state.trainable_weights_sha256,
        checkpoint_identity=f"checkpoint:{state.step}:{state.trainable_weights_sha256}",
        optimizer_steps=state.optimizer_step_count, sft_optimizer_steps=state.sft_optimizer_steps)


def _load_durable_resume(cfg):
    root = Path(cfg.checkpoint_dir)
    if (root/"latest.json").exists() or (root/"checkpoint_state.json").exists():
        return root, load_checkpoint(root, expected_config_hash=cfg.config_hash(), restore_rng=False)
    sft_root = root/"sft_only"
    if (sft_root/"latest.json").exists() or (sft_root/"checkpoint_state.json").exists():
        state = load_checkpoint(sft_root, expected_config_hash=cfg.config_hash(), restore_rng=False)
        if state.step != 0 or state.sft_optimizer_steps <= 0:
            raise ValueError("SFT recovery requires a completed warmup checkpoint at step zero")
        return sft_root, state
    raise ValueError("No durable checkpoint is available. Preserve this interrupted attempt and start a new directory; interrupted SFT cannot be silently repeated")


def _runtime_dependencies():
    import torch
    import triton
    import transformers
    from rl_trainer.gpu_train_grpo import run_single_process_full_param_grpo
    from rl_trainer.qlora_grpo import run_qlora_7b_grpo
    from rl_trainer.evaluate_heldout import (
        main as evaluate_main, _generate_and_unload_hf_model, grade_cached_heldout_completions,
    )
    return SimpleNamespace(torch=torch, triton=triton, transformers=transformers,
        full_train=run_single_process_full_param_grpo, qlora_train=run_qlora_7b_grpo,
        evaluate=evaluate_main, generate=_generate_and_unload_hf_model,
        grade=grade_cached_heldout_completions)


def run_study(config_path: Path, *, threads: int = 4, resume_study: bool = False):
    runtime = _runtime_dependencies()
    torch = runtime.torch
    cfg = ExperimentConfig.from_json(config_path)
    out = Path(cfg.output_jsonl).parent
    environment_path = out/"environment.json"
    if environment_path.exists() and not resume_study:
        raise FileExistsError("Study already exists; use --resume-study to verify and resume this exact attempt")
    if resume_study and not environment_path.exists():
        raise FileNotFoundError("--resume-study requires an existing environment.json; no fresh run was started")
    if not resume_study and cfg.resume_from:
        raise ValueError("A study with resume_from requires --resume-study and its original environment manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA GPU is required; CPU fixtures are not experiment results")
    torch.set_num_threads(threads)
    model_id = {"full_param_0.49b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "qlora_7b_nf4": "Qwen/Qwen2.5-Coder-7B-Instruct"}[cfg.training_mode]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    changed_code = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", "*.py"], text=True).strip()
    untracked_code = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"], text=True).strip()
    if changed_code or untracked_code:
        raise RuntimeError("Commit reviewed source before starting a reproducible study")
    common = dict(model_id=model_id, model_revision=cfg.model_revision, config_hash=cfg.config_hash(),
        source_commit=commit, source_code_dirty=False, training_mode=cfg.training_mode, dtype=cfg.dtype,
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, triton=runtime.triton.__version__,
        transformers=runtime.transformers.__version__, cuda=torch.version.cuda, train_ops=cfg.train_ops,
        heldout_ops=cfg.heldout_ops, development_eval_ops=cfg.development_eval_ops,
        generation_engine="hf_eager", sampling=cfg.sampling_config_dict(), best_of_k=cfg.heldout_best_of_k,
        qlora_cpu_offload=cfg.qlora_cpu_offload,
        checkpoint_root=str(Path(cfg.checkpoint_dir).resolve()), train_output=str(Path(cfg.output_jsonl).resolve()),
        training_schedule=f"{cfg.sft_epochs} training-only supervised epochs then {cfg.max_steps} on-policy GRPO groups")
    identity = {key: common[key] for key in IDENTITY_KEYS}
    env = _read_stage(environment_path, identity)
    train = runtime.qlora_train if cfg.training_mode == "qlora_7b_nf4" else runtime.full_train
    if env and env.get("completed_utc") and not env.get("error"):
        state = load_checkpoint(cfg.checkpoint_dir, expected_config_hash=cfg.config_hash(), restore_rng=False)
        expected = _checkpoint_identity(state)
        _require_identity(env, dict(steps=state.step, optimizer_steps=state.optimizer_step_count,
            sft_optimizer_steps=state.sft_optimizer_steps, trainable_weights_sha256=state.trainable_weights_sha256),
            "completed training checkpoint")
        if state.step != cfg.max_steps:
            raise ValueError("Completed checkpoint does not match the frozen GRPO schedule")
        print("TRAINING_STAGE_REUSED", json.dumps(expected), flush=True)
    else:
        if env:
            durable, _ = _load_durable_resume(cfg)
            cfg.resume_from = str(durable)
        out.mkdir(parents=True, exist_ok=True)
        previous = env
        env = _start_stage(environment_path, common, previous)
        env.update(training_performed=False, resumed_from=cfg.resume_from)
        start = time.monotonic()
        try:
            state = train(cfg)
            persisted = load_checkpoint(cfg.checkpoint_dir, expected_config_hash=cfg.config_hash(), restore_rng=False)
            _require_identity(_checkpoint_identity(state), _checkpoint_identity(persisted), "returned checkpoint")
            if state.step != cfg.max_steps:
                raise ValueError("Trainer did not complete the frozen GRPO schedule")
            env.update(completed_utc=now(), steps=state.step, optimizer_steps=state.optimizer_step_count,
                sft_optimizer_steps=state.sft_optimizer_steps,
                training_performed=state.optimizer_step_count + state.sft_optimizer_steps > 0,
                trainable_weights_sha256=state.trainable_weights_sha256)
        except Exception as exc:
            env.update(error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            env.update(seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write(environment_path, env)
            print("TRAINING_STAGE", json.dumps(env, allow_nan=False), flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = _checkpoint_identity(state)
    stage_identity = dict(identity, **checkpoint)
    proof_path = out/"resume_verified.json"
    proof = _read_stage(proof_path, stage_identity)
    if proof and proof.get("completed_utc") and not proof.get("error"):
        _require_identity(proof, dict(restored_model_checksum_verified=True, restored_optimizer_and_rng=True,
                                     additional_training_steps=0), "completed resume proof")
    else:
        proof = _start_stage(proof_path, dict(common, **checkpoint), proof)
        cfg.resume_from = cfg.checkpoint_dir
        try:
            resumed = train(cfg)
            _require_identity(_checkpoint_identity(resumed), checkpoint, "resume proof")
            proof.update(completed_utc=now(), restored_model_checksum_verified=True,
                restored_optimizer_and_rng=True, additional_training_steps=0)
        except Exception as exc:
            proof.update(error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            write(proof_path, proof)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    evaluation_path = out/"heldout_environment.json"
    evaluation = _read_stage(evaluation_path, stage_identity)
    output_path = out/"heldout.jsonl"
    if evaluation and evaluation.get("completed_utc") and not evaluation.get("error"):
        _complete_output_is_valid(output_path, evaluation)
    else:
        evaluation = _start_stage(evaluation_path, dict(common, **checkpoint), evaluation)
        if output_path.exists():
            evaluation["previous_output"] = _archive(output_path, move=True)
        start = time.monotonic()
        try:
            runtime.evaluate(["--config", str(config_path), "--checkpoint-dir", cfg.checkpoint_dir,
                              "--out", str(output_path)])
            evaluation.update(completed_utc=now(), output_sha256=file_sha256(output_path))
        except Exception as exc:
            evaluation.update(error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            evaluation.update(seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated())
            write(evaluation_path, evaluation)
            print("EVALUATION_STAGE", json.dumps(evaluation, allow_nan=False), flush=True)

    if cfg.sft_epochs:
        sft_dir = Path(cfg.checkpoint_dir)/"sft_only"
        sft_state = load_checkpoint(sft_dir, expected_config_hash=cfg.config_hash(), restore_rng=False)
        if sft_state.step != 0 or sft_state.optimizer_step_count != 0 or sft_state.sft_optimizer_steps <= 0:
            raise ValueError("SFT-only control requires the completed warmup checkpoint before any GRPO update")
        sft_checkpoint = _checkpoint_identity(sft_state)
        ablation_path = out/"sft_only_environment.json"
        ablation = _read_stage(ablation_path, dict(identity, **sft_checkpoint))
        output_path = out/"sft_only.jsonl"
        if ablation and ablation.get("completed_utc") and not ablation.get("error"):
            _complete_output_is_valid(output_path, ablation)
        else:
            ablation = _start_stage(ablation_path, dict(common, **sft_checkpoint), ablation)
            if output_path.exists():
                ablation["previous_output"] = _archive(output_path, move=True)
            start = time.monotonic()
            try:
                cached, chat = runtime.generate(cfg, [("sft_only_checkpoint", 1)],
                    checkpoint_dir=str(sft_dir), cache_path=out/"sft_only.generations.jsonl")
                summary = runtime.grade(cfg, cached, out_jsonl=output_path, chat_template_applied=chat,
                    checkpoint_identity=sft_checkpoint["checkpoint_identity"], generation_engine="hf_eager")
                ablation.update(completed_utc=now(), summary=summary, output_sha256=file_sha256(output_path))
            except Exception as exc:
                ablation.update(error=type(exc).__name__ + ": " + str(exc))
                raise
            finally:
                ablation["seconds"] = time.monotonic()-start
                write(ablation_path, ablation)
                print("SFT_CONTROL_STAGE", json.dumps(ablation, allow_nan=False), flush=True)
    print("STUDY_COMPLETE", str(out), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume-study", action="store_true", help="Resume only this exact recorded source/configuration and durable stages")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    run_study(args.config, threads=args.threads, resume_study=args.resume_study)


if __name__ == "__main__":
    main()
