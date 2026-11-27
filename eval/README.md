# Evaluation reports

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

From the repository root, regenerate the strict retry comparison with:

```bash
python -m eval.compare_retry --run-dir experiments/retry_v2/laptop --run-dir experiments/retry_v2/laptop_7b --out experiments/retry_v2/comparison
```

`compare_retry` verifies config/model revision, GPU/precision, exact row inventories, prompt/source/checkpoint identity, preregistered seeds and the validation-selected winner; `development_test` never enters fresh held-out counts.

Regenerate v3 separately with `python -m eval.compare_retry --run-dir experiments/retry_v3/laptop_7b --protocol experiments/retry_v3/protocol.json --out experiments/retry_v3/comparison`. Preserve the original v2 comparison JSON; a duplicate basename `laptop_7b` is not a unique study identity.

Checkpoint tensors and optimizer/RNG files are excluded from Git. For versioned archives and SHA-256 manifests, see [GitHub Releases](https://github.com/Eljo-M/kernelbench-rl/releases).

The publication snapshot marks `experiments/**`, `docs/**` and `paper/**` as `-text` in `.gitattributes`, preserving exact saved bytes, existing line endings and embedded output/proof hashes in committed blobs and clean clones. Earlier Git snapshots used text normalization and may contain LF-normalized blobs where Windows saved files contained CRLF. Earlier release ZIPs are unchanged; their recorded hashes remain their own identities. This packaging correction does not rerun experiments or retroactively change study commits.

Evidence: [strict comparison](../experiments/retry_v2/comparison/retry_comparison.md) · [strict JSON](../experiments/retry_v2/comparison/retry_comparison.json) · [frozen protocol](../experiments/retry_v2/protocol.json) · [reproduction guide](../REPRODUCING_RETRY.md) · [manuscript source](../paper/manuscript.json) · [worker regression](../experiments/retry_v2/verifier_regression) · [offline final report](../docs/report.html).

Frozen study source: `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`. The later publication snapshot contains documentation, saved-byte packaging and result assets; its distinct source identity is recorded in release provenance.

V3 evidence: [separate strict comparison](../experiments/retry_v3/comparison/retry_comparison.md) · [separate strict JSON](../experiments/retry_v3/comparison/retry_comparison.json) · [prospective protocol](../experiments/retry_v3/protocol.json) · [registered source provenance](../experiments/retry_v3/source_provenance.json). V3 study source prefix `2c174abe2684` is 12 characters only; full source/configuration/tensor identities are retained in the linked manifests and manuscript metadata.

<!-- END KERNELBENCH_RL_V02 -->

Generate a report from contract-compatible JSONL:

```bash
python -m eval.benchmark_report --input eval/results/run.jsonl --out eval/out
```

The output contains `summary.md`, `summary.json`, `speedups.png`, `roofline.png`,
and an offline `report.html`. The source `eval/report.html` also opens directly
and accepts a local JSONL file. Source, errors, and metric fields are rendered
as text. The viewer makes no network requests.

## Trace metadata

`contracts/kernel_eval.schema.json` defines required fields. Optional metadata
recognized by reporting includes:

| Field | Meaning |
| :--- | :--- |
| `method` | Explicit experimental arm, such as `base`, `best_of_8`, or `grpo` |
| `seed` | Integer or string identifying an independent run seed |
| `fixture` | Boolean identifying synthetic test data |
| `source_code` / `candidate_source` / `source` | Kernel source displayed in the viewer |
| `nsight` | Optional profiler metadata displayed in the full record |

Reports never infer the method from a run ID. Synthetic fixtures must be
explicitly labeled; their values are not benchmark evidence. Grouped tables
keep method, backend, operator level, and fixture status separate.

## Metric definitions

- Compile rate counts `compiled=true` among all records.
- Correctness requires `PASS`, `compiled=true`, `correct=true`, and no cheat flag.
- Fast_1.0x uses the all-record denominator and counts only correct candidates
  with positive candidate/eager timing and recorded speedup greater than one.
- Fast_compile uses all GPU records with a positive `compile_ms` baseline as
  its denominator, including failed candidates. If no such baseline exists,
  the metric is unavailable.
- Geometric mean speedup uses only correct, positively timed candidates. It is
  conditional on success; it must not be presented as an all-operator speedup.
- Zero latency or throughput means unavailable, not a measured zero.
- `numpy_dev` timing is CPU development data. It never supplies a
  `torch.compile` comparison and is excluded from the GPU throughput plot.

The optional 95% interval uses a deterministic 2,000-resample percentile
bootstrap over whole seeds. Each seed contributes its correct-only geometric
mean with equal weight. The interval field reports that equal-seed estimate,
which can differ from the pooled-record geometric mean when seed sizes differ.
At least two seeds and complete seed metadata are required. Mixed methods,
backends, and fixture status do not get a combined interval. A small seed count,
different operator coverage, and conditioning on success limit interpretation;
the interval does not establish a paired training ablation.

The throughput plot uses the supplied `achieved_tflops` and
`arithmetic_intensity` fields for correct, timed GPU records. Baseline throughput
is derived using the same work and its recorded baseline latency. Operator
FLOP/byte accounting determines the interpretation; a plot cannot verify its
cost model. A hardware roof is drawn only with explicit parameters:

```bash
python -m eval.benchmark_report --input traces.jsonl --out eval/out \
  --peak-tflops <measured-or-specified-peak> --bandwidth-gbps <bandwidth>
```

Both parameters must be positive. No GPU peak or bandwidth is guessed.

## Historical v1: paired held-out model comparison

```bash
python -m eval.compare_heldout --run-dir experiments/laptop_fullparam --out eval/out/heldout
```

Repeat `--run-dir` for independent runs. Each directory supplies `config.json`,
`heldout_environment.json` and raw `heldout.jsonl`. The reporter checks config,
model revision, precision, hardware, per-operator seeds and complete final
coverage. Selection, fixture and oracle records are excluded. It writes Markdown,
strict JSON and a PNG, preserving failures in the denominator and separating
models, GPU types and precisions. Three held-out workloads do not support a broad
generalization claim. Best-of-K and pass@1 have different search budgets.

## Nsight Compute CSV

```bash
python -m eval.parse_ncu --input profile.csv --out eval/out/nsight.json
```

The parser supports long-form exports containing `Kernel Name`, `Metric Name`,
`Metric Unit`, and `Metric Value`, and raw exports containing metric columns
plus an optional units row. Launches remain separated by process, launch ID,
context, stream, and kernel name. No evaluation `run_id` is inferred: join with
execution records only when the profiling run has a known association.

Known DRAM percentage, DRAM bytes/second, duration, and Tensor pipe metrics
receive normalized fields. All raw metric names and sections are retained.
Unavailable values remain null. Tensor active-cycle and elapsed-cycle
percentages have different denominators and remain separate. Unrecognized
Tensor pipelines stay in the named metric maps rather than being relabeled as
an aggregate utilization figure. Time converts to milliseconds; bandwidth
uses decimal SI prefixes and converts to GB/s.

Unit and CSV behavior follows the official [Nsight Compute metrics and units
guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#metrics-and-units)
and [Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
