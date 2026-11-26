# Execution-feedback training

The original `train_grpo` entrypoint supports `numpy_dev` development experiments.
The separate `gpu_train_grpo` bridge adds real CUDA candidate grading,
full-parameter small-model training and optional 7B QLoRA with resume. FSDP and
multi-node training are not implemented. CPU development timings are not GPU
speedup evidence.

`python -m rl_trainer.train_grpo --dry-run` grades four fixed RMSNorm fixtures,
writes strict JSONL with source hashes, seeds and group advantages, and explicitly
records that no weights were updated. The `.config.json` sibling records the run
configuration. This mode needs only the ordinary development dependencies.

Search against an already-running vLLM server:

```bash
python -m rl_trainer.train_grpo --mode search --base-url http://localhost:8000/v1 \
  --model YOUR_SERVED_MODEL --group-size 8 --repair-turns 2 \
  --out eval/results/search.jsonl
```

The endpoint uses vLLM's [Completions protocol](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/).
`VLLM_API_KEY`, if set, is passed in an authorization header and never logged.
Generation and grading have bounded concurrency. All group members are regraded
after each repair turn; consumers should group by `group_id` and `repair_turn`
instead of counting repaired rows as independent trials. Search writes no learned
checkpoints and never assumes remote serving weights have changed.

Optional single-process policy training requires `torch` and `transformers` and
predownloaded local model weights; loading uses `local_files_only=True` and does
not execute remote model code:

```bash
python -m rl_trainer.train_grpo --mode train --model /models/local-small-coder \
  --device cuda --group-size 8 --steps 10 --out eval/results/train.jsonl \
  --checkpoint-dir checkpoints/experiment-01
```

Each group is sampled from the current policy with untruncated temperature-one
sampling. Completion tokens include EOS and exclude padding/prompt tokens. Old
policy and frozen reference token probabilities are captured before the update.
The update uses a clipped token ratio, per-completion group advantage, sampled KL
penalty, AdamW, and gradient clipping. Dropout stays disabled for consistent
rollout and update probabilities. Logs include entropy, loss and gradient norm.
Each successful step saves the policy, tokenizer, optimizer, RNG state and run
configuration. Checkpoints are saved for inspection; automatic resume is not yet
implemented. Repair is search-only because training repair trajectories require
separate conditional probability accounting.

A policy plus frozen reference plus optimizer can need substantially more memory
than model inference. Use a small model to validate the optional path before a
large run. The training path has not been validated on a distributed GPU setup,
and the presence of a checkpoint alone is not evidence of better kernel quality.
Compare held-out base pass@1, best-of-group search and trained-policy runs on the
same operators, seeds and verifier settings before reporting gains.

`launch_cloud_gpu.sh` defaults to CPU fixtures. It runs on an existing node and
does not provision resources, install dependencies, or download weights.

## CUDA bridge and free-resource experiments

`gpu_train_grpo` preserves the original entrypoint and adds configuration-driven
CUDA grading with raw completion tokens (including EOS), old/reference logps,
clipped GRPO, finite-gradient checks, weight-delta/checksum evidence and resume:

```bash
python -m rl_trainer.gpu_train_grpo --config experiments/laptop_fullparam/config.json
python -m rl_trainer.evaluate_heldout --config experiments/laptop_fullparam/config.json \
  --checkpoint-dir checkpoints/qwen05b_fp32_adafactor --out experiments/laptop_fullparam/heldout.jsonl
```

Download the public model weights first and set `model_name_or_path` to that local
directory. Record its immutable Hugging Face revision in `model_revision`.
`ExperimentConfig` validates the JSON configuration. The laptop experiment uses
the 0.49B model in FP32 with Adafactor to fit the available memory while updating
the full model. Optional BF16/FP16 CPU-master AdamW retains FP32 master weights
and optimizer moments in RAM. These are distinct optimizer configurations.

`ExperimentConfig.preset_t4_7b_qlora()` configures a 7B NF4 base with rank-8 LoRA,
FP16 compute, adapter-only AdamW and a shared frozen-base reference. It saves
adapter weights, not a full-parameter 7B checkpoint. The free T4 environment uses
torch 2.6/cu124 and Triton 3.2 for sm75 compatibility. LoRA dependencies are in the
`qlora` extra; no resource provisioning or billing is performed by these tools.

Resume by setting `resume_from` to the existing checkpoint directory in the same
configuration and increasing `max_steps`. Optimizer, RNG, step count and model or
adapter weights are restored. Incompatible configuration hashes are rejected.

Held-out generation loads each model sequentially and frees it before grading.
Each completion is saved immediately to an identity-checked generation cache;
restarting the same command reuses completed samples and generates only missing
ones. Configuration, checkpoint identity and the actual formatted prompt hash
must match. Checkpoints become visible only after all weights and optimizer files
have been written. Resume archives any logged steps beyond the restored checkpoint
before replaying them, preserving the original log for audit.
Best-of-K selects on a separate validation seed, then regrades the winner on the
held-out test seed. Selection records carry `eval_phase: selection_val`; only
`heldout_test` rows belong in final comparison metrics. The train/held-out split
contains RMSNorm shape generalization and two unseen operation types. Three
held-out workloads are a small experiment, not a broad generalization claim.
