# Execution-feedback training

<!-- BEGIN KERNELBENCH_RL_V02 -->

## v0.2: completed corrected studies

Publication record: **2026-11-26**. These separately identified v2/v3 protocols use the custom 30-operator registry; they do not claim coverage of the complete official KernelBench benchmark. Original v1 results below remain historical evidence.

[Project website](https://eljo-m.github.io/kernelbench-rl/) · [Paper landing page](https://eljo-m.github.io/kernelbench-rl/paper.html) · [Searchable PDF](https://eljo-m.github.io/kernelbench-rl/assets/kernelbench-rl-paper.pdf) · [Study report](https://eljo-m.github.io/kernelbench-rl/report.html) · [Release index](https://github.com/Eljo-M/kernelbench-rl/releases)

The v2 schedule uses the dtype-aware `contract_v2` prompt, 32 training-only supervised updates, then eight groups of two completions with correctness-gated GRPO. Every non-PASS candidate receives zero training reward; valid PASS speed rewards are preserved rather than replaced by binary values. Equal rewards yield zero relative advantage; even correct groups can have equal or differing speed rewards. The SFT-only checkpoint is saved before GRPO. The original pre-SFT base anchors reference regularization.

The separate prospective **v3 7B rank 16** study starts from the pinned original base with fresh adapters, not transferred v2 weights or exact resume. Rank 16/alpha 32 targets q/k/v/o plus gate/up/down projections. It records 88 SFT updates over 11 training operators and 16 GRPO groups; actual GRPO updates are reported below. All three former v2 fresh targets are now training-seen and excluded from v3 fresh/development denominators. New prospective fresh operators are `cosine_similarity_1024`, `fused_bias_gelu_4096`, `matmul_bias_silu_512`. Their generated programs were not previously inspected in the recorded repository exposure audit; public fixed references and related families were already exposed. Rank, targets, schedule, seed and fresh tasks differ from v2, so no causal SFT/GRPO, model-capacity or pooled comparison follows.

| Model track | SFT updates | GRPO groups | GRPO candidates | GRPO updates | Training statuses |
| --- | --- | --- | --- | --- | --- |
| v2 0.49B FP32 | 32 | 8 | 16 | 0 | `COMPILE_ERROR`=14, `RUNTIME_ERROR`=2 |
| v2 7B rank 8 | 32 | 8 | 16 | 1 | `CHEAT`=1, `COMPILE_ERROR`=5, `NUMERICAL_MISMATCH`=3, `PASS`=5, `RUNTIME_ERROR`=2 |
| v3 7B rank 16 | 88 | 16 | 32 | 5 | `PASS`=31, `RUNTIME_ERROR`=1 |

V2 fresh held-out passes (N=3 per method; different fresh operators across protocols):

| Model track | Base pass@1 | Base best-of-8 | SFT-only pass@1 | Hybrid pass@1 |
| --- | --- | --- | --- | --- |
| v2 0.49B FP32 | 0/3 | 0/3 | 0/3 | 0/3 |
| v2 7B rank 8 | 0/3 | 0/3 | 2/3 | 2/3 |

V3 fresh held-out passes (N=3 per method; different fresh operators across protocols):

| Model track | Base pass@1 | Base best-of-8 | SFT-only pass@1 | Hybrid pass@1 |
| --- | --- | --- | --- | --- |
| v3 7B rank 16 | 0/3 | 0/3 | 2/3 | 2/3 |

Pass@1 methods use one completion; base best-of-eight uses eight validation-selection samples per fresh operator, followed by independent regrading of its winner. Selection records are excluded from final denominators. Failures remain in pass-rate and fast-versus-eager denominators; correct-only speedups and unavailable compiled comparisons are labeled separately.

- v2 0.49B FP32: 39 main rows = 24 selection + 9 fresh final + 6 development final; 6 separate SFT-only rows = 3 fresh + 3 development.
- v2 7B rank 8: 39 main rows = 24 selection + 9 fresh final + 6 development final; 6 separate SFT-only rows = 3 fresh + 3 development.
- v3 7B rank 16: 39 main rows = 24 selection + 9 fresh final + 6 development final; 6 separate SFT-only rows = 3 fresh + 3 development.

V2 fresh targets were `l2_normalize_2048`, `fused_bias_silu_4096` and `matmul_residual_add_512`; they are training-seen in the separate v3 follow-up when present. Each prospective trio tests programs within shared training families, not unseen-family out-of-distribution behavior. Previously inspected RMSNorm-4096, INT8 weight-only GEMM and attention remain development regressions. Protocols, model/precision tracks and different fresh cohorts are not pooled. Historical `contract_v1` and later `contract_v2` outcomes are unpaired comparisons. The hybrid includes SFT and establishes no RL-only causal effect; N=3 and one training seed per study do not establish significance or broad generalization.

The laptop 7B studies update LoRA adapters over a frozen NF4 base with an FP16 GPU decoder and fixed FP32 CPU embedding/output computations. They are separate from both the historical free-T4 placement and full-parameter 0.49B FP32 training. FSDP, multi-node training and full-parameter 7B remain outside implemented and validated scope.

Zero GRPO updates were recorded for v2 0.49B FP32. Each corresponding final trainable-tensor digest equals its SFT-only digest. Sampled-output or timing differences between those arms are not RL parameter gains; skipped-update zero diagnostics are sentinels.

Fixed baselines are curated implementations prepared with coding-assistant support, distinct from programs sampled by the evaluated policies; historical files' 'handwritten' wording does not assert exclusively human authorship. The historical v1 **180/180** campaign and geometric means remain unchanged: FP32 **2.494x eager / 1.089x compiled**, BF16 **5.417x / 1.248x**, with 30 eager and 29 available compiled operators. The separate updated-worker correctness regression passed **60/60** operator/precision records at seed 142 and independent seed 10142. It checks all 30 operators in FP32/BF16 and does not re-estimate the historical speedup campaign.

| Model track | Training seconds | Parent peak allocated CUDA GiB |
| --- | --- | --- |
| v2 0.49B FP32 | 607.42 | 4.741 |
| v2 7B rank 8 | 2079.19 | 4.692 |
| v3 7B rank 16 | 4958.95 | 5.295 |

The CUDA peak is the training parent process's PyTorch allocator peak; it excludes child-worker and other applications' device allocations. A checksum-verified checkpoint reload restores optimizer/RNG state without extra training steps; this is not an independent next-step trajectory replication or a separate optimizer/RNG digest comparison.

From the repository root, the v0.2 entrypoint is `python -m rl_trainer.run_study --config experiments/retry_v2/laptop/config.json` (use `experiments/retry_v2/laptop_7b/config.json` for the separate 7B track). The reproduction guide gives the complete setup, fixed model revisions and checkpoint restoration layout. The legacy `train_grpo` development/search path remains a separate entrypoint.

Checkpoint tensors and optimizer/RNG files are excluded from Git. For versioned archives and SHA-256 manifests, see [GitHub Releases](https://github.com/Eljo-M/kernelbench-rl/releases).

The publication snapshot marks `experiments/**`, `docs/**` and `paper/**` as `-text` in `.gitattributes`, preserving exact saved bytes, existing line endings and embedded output/proof hashes in committed blobs and clean clones. Earlier Git snapshots used text normalization and may contain LF-normalized blobs where Windows saved files contained CRLF. Earlier release ZIPs are unchanged; their recorded hashes remain their own identities. This packaging correction does not rerun experiments or retroactively change study commits.

Evidence: [strict comparison](../experiments/retry_v2/comparison/retry_comparison.md) · [strict JSON](../experiments/retry_v2/comparison/retry_comparison.json) · [frozen protocol](../experiments/retry_v2/protocol.json) · [reproduction guide](../REPRODUCING_RETRY.md) · [manuscript source](../paper/manuscript.json) · [worker regression](../experiments/retry_v2/verifier_regression) · [offline final report](../docs/report.html).

Frozen study source: `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`. The later publication snapshot contains documentation, saved-byte packaging and result assets; its distinct source identity is recorded in release provenance.

V3 evidence: [separate strict comparison](../experiments/retry_v3/comparison/retry_comparison.md) · [separate strict JSON](../experiments/retry_v3/comparison/retry_comparison.json) · [prospective protocol](../experiments/retry_v3/protocol.json) · [registered source provenance](../experiments/retry_v3/source_provenance.json). V3 study source prefix `2c174abe2684` is 12 characters only; full source/configuration/tensor identities are retained in the linked manifests and manuscript metadata.

<!-- END KERNELBENCH_RL_V02 -->

The original `train_grpo` entrypoint supports `numpy_dev` development experiments.
The separate `gpu_train_grpo` bridge adds real CUDA candidate grading,
full-parameter small-model training and optional 7B QLoRA with resume. FSDP and
multi-node training are not implemented. CPU development timings are not GPU
speedup evidence.

`python -m rl_trainer.train_grpo --dry-run` grades four fixed RMSNorm fixtures,
writes strict JSONL with source hashes, seeds and group advantages, and explicitly
records that no weights were updated. The `.config.json` sibling records the run
configuration. This mode needs only the ordinary development dependencies.

Search against an already-running vLLM server (the optional `serving` extra
installs vLLM on Linux; GPU grading itself does not require a serving engine):

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

## Historical v1: CUDA bridge and free-resource experiments

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
`heldout_test` rows belong in final comparison metrics. The historical v1 train/held-out split
contains RMSNorm shape generalization and two unseen operation types. Three
held-out workloads are a small experiment, not a broad generalization claim.

The [free-T4 notebook](../notebooks/free_t4.ipynb) defaults to a real worker
preflight. Full model download, training, evaluation and export are explicit
configuration options. It uses public weights and a separate torch 2.6/cu124,
Triton 3.2 environment. The [experiment index](../experiments/README.md) identifies
completed runs, intermediate failures and the authoritative benchmark suite.

Teacher-forced old, reference and update scoring explicitly passes
`use_cache=False`, while generation retains its configured KV cache. The helper
supports PEFT wrappers and chunked differentiable selected-token probabilities.
The historical runs record their original source commit; later scoring changes
do not retroactively change those records.

For a read-only 7B QLoRA check on a saved initial-policy sample:

```bash
python -m rl_trainer.diagnose_logps --config original/config.json --input original/train.jsonl \
  --initial-checkpoint original/checkpoints --out new-logp-diagnostic.json
```

The diagnostic compares cache modes, repeated calls, eval/training forwards and
stored probabilities, with gradients enabled for the training forward but no
backward/update. It verifies unchanged trainable weights and records versions,
trace and checkpoint identities. It never adjusts the recovery tolerance or
changes raw traces. Recovery rejects mismatched probabilities before updating.
