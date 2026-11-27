# Track coordination

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

Checkpoint tensors and optimizer/RNG files are excluded from Git. For versioned archives and SHA-256 manifests, see [GitHub Releases](https://github.com/Eljo-M/kernelbench-rl/releases).

The publication snapshot marks `experiments/**`, `docs/**` and `paper/**` as `-text` in `.gitattributes`, preserving exact saved bytes, existing line endings and embedded output/proof hashes in committed blobs and clean clones. Earlier Git snapshots used text normalization and may contain LF-normalized blobs where Windows saved files contained CRLF. Earlier release ZIPs are unchanged; their recorded hashes remain their own identities. This packaging correction does not rerun experiments or retroactively change study commits.

Evidence: [strict comparison](experiments/retry_v2/comparison/retry_comparison.md) · [strict JSON](experiments/retry_v2/comparison/retry_comparison.json) · [frozen protocol](experiments/retry_v2/protocol.json) · [reproduction guide](REPRODUCING_RETRY.md) · [manuscript source](paper/manuscript.json) · [worker regression](experiments/retry_v2/verifier_regression) · [offline final report](docs/report.html).

Frozen study source: `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`. The later publication snapshot contains documentation, saved-byte packaging and result assets; its distinct source identity is recorded in release provenance.

V3 evidence: [separate strict comparison](experiments/retry_v3/comparison/retry_comparison.md) · [separate strict JSON](experiments/retry_v3/comparison/retry_comparison.json) · [prospective protocol](experiments/retry_v3/protocol.json) · [registered source provenance](experiments/retry_v3/source_provenance.json). V3 study source prefix `2c174abe2684` is 12 characters only; full source/configuration/tensor identities are retained in the linked manifests and manuscript metadata.

<!-- END KERNELBENCH_RL_V02 -->

## Historical v1: coordination record

Project: `Eljo-M/kernelbench-rl`. Integration branch: `codex/gpu-benchmarks-training`.
This file records the actual implementation state and remaining experiments.

| Track | State | Remaining work |
| :--- | :--- | :--- |
| Runner | CPU worker, timeout, reduced environment, static checks, multi-seed/stress checks, mutation and timing validation; Rust supervisory CLI verified in CI | Supply external isolation for adversarial execution |
| Kernels | All 30 Triton baselines integrated; final canonical/stress CUDA suite 180/180 PASS in fp32/bf16 on three seeds | Windows SSM compiled comparison unavailable; 29 comparable operators |
| Training | Completed real 0.49B full-parameter and free 7B QLoRA training/held-out studies; one and four optimizer updates respectively | No held-out quality gain; FSDP/multi-node training not implemented |
| Evaluation | Strict ingestion, grouped summaries, plots, offline viewer, profiler import and paired held-out report | Three held-out workloads per model limit generalization claims |

## 2026-11-26 integration

- @Track2 supplied the operator expansion, baseline sources, PyTorch references and
  CUDA validator, then fixed review findings in strict JSON, bf16 input rounding,
  output overwrite behavior, metadata, lazy compilation status and timing.
- @Lead integrated the files, removed estimated `torch.compile` timings, preserved
  `numpy_dev` labels, bounded CPU execution, and added schema/timeout regressions.
- @Track3 implemented and checked the local rollout/objective/update plumbing. A
  tiny local policy fixture checks parameter updates and checkpoint writes. This
  test is not a learned large-model result.
- @Track4 implemented reporting and checked the offline viewer against rollout
  source records. Synthetic sample traces and dry-run fixtures are explicitly
  labeled and must not be used as performance evidence.
- Native PyTorch references pass 50 numerical comparisons on CPU across canonical
  and stress cases in fp32/bf16. Triton compilation and GPU performance remain
  unverified locally. Cost models are analytical estimates, not measured HBM data.
- All 67 local Python tests passed. The initial implementation commit
  `5f9299075b869873b8a5d7365d0a98ef7f318bc0` passed
  [GitHub CI](https://github.com/Eljo-M/kernelbench-rl/actions/runs/45870454904),
  including Python tests, operator verification, report generation and Rust tests.

## Completed free-resource studies

The fresh free-T4 study and held-out evaluation completed. Its checkpoint and
raw records have been exported. The corrected worker passed a real T4 preflight.
The initial attempt produced
no updates because child processes lacked the Linux driver environment; the
failed recovery also stopped before any update after a probability mismatch.
Both attempts remain separate from fresh training results. Only final test-phase
records enter comparisons, with equal pass@1 budgets for base and trained models.

## Free-resource experiments

- RTX 5060 Laptop: final 180/180 baseline checks passed for 30 operators, fp32/bf16 and
  seeds 42/43/44. CUDA graph timings and dispatch-sensitive event timings are
  retained separately under `experiments/rtx5060`.
- Free Colab T4: 33/33 fp32 baseline checks passed for the same 11 operators and
  three seeds. Source commit and raw records are under `experiments/colab_t4`.
- The 0.49B full-parameter FP32/Adafactor experiment completed eight groups and
  one optimizer update. The full-model weight change was verified; all 16
  generated training candidates failed. This establishes an execution run,
  not a learned kernel-quality gain. Checkpoint tensors and optimizer/RNG files are excluded from Git.
  For versioned archives and SHA-256 manifests, see [GitHub Releases](https://github.com/Eljo-M/kernelbench-rl/releases). Held-out base pass@1, best-of-8 and trained checkpoint each passed
  0/3; the result shows no quality improvement. The real checkpoint restored
  model weights, optimizer and RNG with a matching trainable-weight checksum.
- The 7B NF4/rank-8 QLoRA run uses a free T4 with 5,046,272 trainable adapter
  parameters. Fresh training completed four finite, nonzero adapter updates.
  Base pass@1, best-of-8 and trained pass@1 each passed 0/3 held-out workloads.
  Independent saved-adapter comparison measured L2 change 0.19911.
- The actual T4 checkpoint restored adapter weights, optimizer and RNG with an
  exact checksum at step 8, preserving four optimizer updates and adding no
  training steps. The live probability diagnostic reproduced the 0.013374
  cache-mode difference on the original 395-token sample; repeated no-cache
  eval and grad-enabled training forwards matched exactly. Strict summaries are
  retained in `experiments/colab_t4/live_verification.json`.
- Commit `3e9de7c9bf1cd83a523282e26e41feaf4505c66b` passed
  [CI](https://github.com/Eljo-M/kernelbench-rl/actions/runs/45877414778), including
  actual Torch backward, optimizer and checkpoint tests.
