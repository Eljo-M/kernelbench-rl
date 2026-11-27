# KernelBench-RL architecture

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

Evidence: [strict comparison](experiments/retry_v2/comparison/retry_comparison.md) · [strict JSON](experiments/retry_v2/comparison/retry_comparison.json) · [frozen protocol](experiments/retry_v2/protocol.json) · [reproduction guide](REPRODUCING_RETRY.md) · [manuscript source](paper/manuscript.json) · [worker regression](experiments/retry_v2/verifier_regression) · [offline final report](docs/report.html).

Frozen study source: `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`. The later publication snapshot contains documentation, saved-byte packaging and result assets; its distinct source identity is recorded in release provenance.

V3 evidence: [separate strict comparison](experiments/retry_v3/comparison/retry_comparison.md) · [separate strict JSON](experiments/retry_v3/comparison/retry_comparison.json) · [prospective protocol](experiments/retry_v3/protocol.json) · [registered source provenance](experiments/retry_v3/source_provenance.json). V3 study source prefix `2c174abe2684` is 12 characters only; full source/configuration/tensor identities are retained in the linked manifests and manuscript metadata.

<!-- END KERNELBENCH_RL_V02 -->

The JSON contract in `contracts/kernel_eval.schema.json` connects evaluation,
reward calculation and reporting. Optional trace metadata carries the seed,
method, candidate source hash, repair turn and fixture provenance.

```text
Generator (fixed CPU fixtures / existing model endpoint / local policy)
  -> candidate group
  -> AST policy + disposable Python workers with timeouts
  -> multi-seed/stress numerical checks and repeated CPU/CUDA timing
  -> KernelEvalResult + provenance
  -> group-relative rewards -> optional policy update
  -> JSONL -> grouped tables, profiler import, plots and offline trace viewer
```

The NumPy development verifier and generated-candidate CUDA verifier have
separate paths. The CUDA baseline validator checks all 30 handwritten entries.
A device being present never changes the label of a CPU measurement. Unavailable
comparison baselines use zero fields and explicit diagnostic metadata, rather
than an estimated timing. Reporting excludes unavailable timing comparisons.

The local training implementation must sample from the same policy it updates.
Search against a vLLM server records no policy update because serving weights are
not automatically synchronized. Multi-turn repair belongs to search; the trainer
does not silently use repaired sequences as original on-policy samples.

In the historical v1 shaped-reward mode, kernel rewards penalize prohibited reference fallbacks, distinguish loading
failure from numerical failure, and reward measured speedup for correct programs.
Rewards are group-normalized before the clipped GRPO objective. The CUDA trainer
retains sampled completion IDs and old/reference probabilities, checks finite
gradients and records actual parameter changes. Full-model and adapter checkpoints
are atomic and preserve optimizer/RNG state. Evaluation generates base and trained
completions sequentially, then grades them after unloading the models. Best-of-K
selection uses a separate validation seed before final test grading. Reporting
checks complete operator/method coverage and keeps hardware/precision runs separate.
A checkpoint alone is not evidence of learning; the completed historical v1 laptop ablation
showed no held-out quality gain.

The development execution boundary is a separate process with a reduced
environment, conservative NumPy API and timeout. This is not OS-level isolation.
For arbitrary untrusted kernels, supply an external VM/container boundary with
resource and access restrictions before execution. GPU event timing and native
compiler execution require their own suitable CUDA environment.

Track responsibilities and verified/remaining work are recorded in
[TRACK_SYNC.md](TRACK_SYNC.md).
