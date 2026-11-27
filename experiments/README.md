# Recorded experiments

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

Previously inspected development passes, kept separate from the fresh cohort:

| Model track | Base pass@1 | SFT-only pass@1 | Hybrid pass@1 | Best-of-eight |
| --- | --- | --- | --- | --- |
| v2 0.49B FP32 | 0/3 | 0/3 | 0/3 | Unavailable |
| v2 7B rank 8 | 0/3 | 1/3 | 1/3 | Unavailable |
| v3 7B rank 16 | 0/3 | 1/3 | 1/3 | Unavailable |

V2 fresh targets were `l2_normalize_2048`, `fused_bias_silu_4096` and `matmul_residual_add_512`; they are training-seen in the separate v3 follow-up when present. Each prospective trio tests programs within shared training families, not unseen-family out-of-distribution behavior. Previously inspected RMSNorm-4096, INT8 weight-only GEMM and attention remain development regressions. Protocols, model/precision tracks and different fresh cohorts are not pooled. Historical `contract_v1` and later `contract_v2` outcomes are unpaired comparisons. The hybrid includes SFT and establishes no RL-only causal effect; N=3 and one training seed per study do not establish significance or broad generalization.

The laptop 7B studies update LoRA adapters over a frozen NF4 base with an FP16 GPU decoder and fixed FP32 CPU embedding/output computations. They are separate from both the historical free-T4 placement and full-parameter 0.49B FP32 training. FSDP, multi-node training and full-parameter 7B remain outside implemented and validated scope.

Zero GRPO updates were recorded for v2 0.49B FP32. Each corresponding final trainable-tensor digest equals its SFT-only digest. Sampled-output or timing differences between those arms are not RL parameter gains; skipped-update zero diagnostics are sentinels.

Fixed baselines are curated implementations prepared with coding-assistant support, distinct from programs sampled by the evaluated policies; historical files' 'handwritten' wording does not assert exclusively human authorship. The historical v1 **180/180** campaign and geometric means remain unchanged: FP32 **2.494x eager / 1.089x compiled**, BF16 **5.417x / 1.248x**, with 30 eager and 29 available compiled operators. The separate updated-worker correctness regression passed **60/60** operator/precision records at seed 142 and independent seed 10142. It checks all 30 operators in FP32/BF16 and does not re-estimate the historical speedup campaign.

From the repository root, the v0.2 entrypoint is `python -m rl_trainer.run_study --config experiments/retry_v2/laptop/config.json` (use `experiments/retry_v2/laptop_7b/config.json` for the separate 7B track). The reproduction guide gives the complete setup, fixed model revisions and checkpoint restoration layout. The legacy `train_grpo` development/search path remains a separate entrypoint.

Checkpoint tensors and optimizer/RNG files are excluded from Git. For versioned archives and SHA-256 manifests, see [GitHub Releases](https://github.com/Eljo-M/kernelbench-rl/releases).

The publication snapshot marks `experiments/**`, `docs/**` and `paper/**` as `-text` in `.gitattributes`, preserving exact saved bytes, existing line endings and embedded output/proof hashes in committed blobs and clean clones. Earlier Git snapshots used text normalization and may contain LF-normalized blobs where Windows saved files contained CRLF. Earlier release ZIPs are unchanged; their recorded hashes remain their own identities. This packaging correction does not rerun experiments or retroactively change study commits.

Evidence: [strict comparison](retry_v2/comparison/retry_comparison.md) · [strict JSON](retry_v2/comparison/retry_comparison.json) · [frozen protocol](retry_v2/protocol.json) · [reproduction guide](../REPRODUCING_RETRY.md) · [manuscript source](../paper/manuscript.json) · [worker regression](retry_v2/verifier_regression) · [offline final report](../docs/report.html).

Frozen study source: `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`. The later publication snapshot contains documentation, saved-byte packaging and result assets; its distinct source identity is recorded in release provenance.

V3 evidence: [separate strict comparison](retry_v3/comparison/retry_comparison.md) · [separate strict JSON](retry_v3/comparison/retry_comparison.json) · [prospective protocol](retry_v3/protocol.json) · [registered source provenance](retry_v3/source_provenance.json). V3 study source prefix `2c174abe2684` is 12 characters only; full source/configuration/tensor identities are retained in the linked manifests and manuscript metadata.

<!-- END KERNELBENCH_RL_V02 -->

## Historical v1: experiment index

These directories contain actual local/Colab records. Synthetic unit-test data
and dry-run fixtures are kept elsewhere and do not enter the results below.

| Directory | Hardware / scope | Interpretation |
| :--- | :--- | :--- |
| `rtx5060/full30` | RTX 5060 Laptop, all 30 operators, fp32/bf16, seeds 42/43/44 | Final 180/180 passing canonical/stress checks; authoritative laptop baseline suite |
| `rtx5060/cuda_graph` | Original 11 operators, same laptop, two precisions and three seeds | Earlier 66/66 passing baseline suite |
| `rtx5060/extended` | First 19-extension attempt | Incomplete; graph-capture errors retained as diagnostics, not final baseline evidence |
| `rtx5060/extended_v2` | Intermediate 19-extension suite | 114/114 passing before the additional stress cases in `full30` |
| `colab_t4/baselines` | Free Tesla T4, original 11 operators, fp32 and three seeds | 33/33 passing; separate hardware track |
| `colab_t4/failed_worker_attempt` | Initial 7B generation attempt | All child workers lacked CUDA driver environment; zero optimizer updates; not a trained result |
| `laptop_fullparam` | Full 494,032,768-parameter Qwen coder, FP32/Adafactor | Eight groups, 16 samples, one verified update; exact model/optimizer/RNG resume; held-out base/search/trained each 0/3 |
| `colab_t4/qlora7b_fresh` | Free Tesla T4, 7B NF4/FP16 QLoRA, 5,046,272 adapter parameters | Eight groups, 16 samples, four verified updates; held-out base/search/trained each 0/3 |
| `colab_t4/qlora7b_recovered` | Regrading of the initial zero-update attempt | Recovery rejected before any update due to probability mismatch; excluded from trained results |
| `heldout_comparison` | Separate laptop and T4 panels | Final paired report: nine test rows per model, with 24 selection rows per model excluded |

Both bounded training and held-out studies are complete. Neither observed a
kernel-quality improvement. The independent T4 adapter comparison is recorded in
`colab_t4/qlora7b_fresh/adapter_delta_verified.json`.
The live T4 checkpoint-resume and cache-mode diagnostic summaries are in
[live verification](colab_t4/live_verification.json) and
[checkpoint resume](colab_t4/qlora7b_fresh/resume_verified.json).

## Regenerate reports

From the repository root, using the ordinary report dependencies:

```bash
python -m eval.benchmark_report --input experiments/rtx5060/full30/baselines.jsonl --out experiments/rtx5060/full30/report
python -m eval.summarize_baselines --input experiments/rtx5060/full30/baselines.jsonl --out experiments/rtx5060/full30/report
python -m eval.compare_heldout --run-dir experiments/laptop_fullparam --run-dir experiments/colab_t4/qlora7b_fresh --out experiments/heldout_comparison
```

The final suite reports geometric mean speedups across 30 operators versus
native eager and 29 versus verified `torch.compile`, separately per precision.
Windows Inductor cannot compile the SSM reference; its comparison is explicitly
unavailable. The Triton SSM kernel passed correctness checks. Graph replay times
exclude Python dispatch and compilation. Event-mode records are retained
separately and must not be pooled with graph replay.

Model-generated programs are graded with the same operator contracts, multiple
input seeds, input-mutation checks, source policy and timing procedure. Final
held-out comparisons use only `eval_phase=heldout_test`; best-of-8 selection uses
a separate validation seed. Failed candidates remain in pass-rate denominators.
No training-quality gain is inferred from a checkpoint or handwritten baseline.

Checkpoint tensors and optimizer/RNG files are excluded from Git.
For versioned archives and SHA-256 manifests, see [GitHub Releases](https://github.com/Eljo-M/kernelbench-rl/releases). Raw completions, tokens, probabilities,
source hashes, configuration, pinned model revisions and environment metadata
preserve the experiment audit trail. GPU timing and learned quality depend on
these recorded configurations; this is a bounded experiment with three held-out
workloads, not a broad model or hardware ranking.
