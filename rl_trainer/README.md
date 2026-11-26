# Execution-feedback training

The current target is `numpy_dev`. These CPU development measurements are useful
for checking rollout plumbing and reward optimization; they are not GPU kernel
speedup evidence. Full CUDA/Triton grading, FSDP, LoRA and multi-node training are
not implemented in this trainer.

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
