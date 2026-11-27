# KernelBench-RL

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

Execution-feedback tooling for generating and checking numerical kernels, with
30 deterministic operator workloads, each with a Triton baseline and native
PyTorch reference.

The CPU pipeline runs locally: candidate source → bounded worker → numerical
checks → reward and group advantages → strict JSONL → report. The repository
also includes optional policy-search/training adapters and a CUDA baseline
validation path. Measured laptop baseline results are in
[`experiments/rtx5060/full30`](experiments/rtx5060/full30).
In historical v1, both the 0.49B full-parameter and free-T4 7B QLoRA experiments completed training
and held-out evaluation. Neither showed a kernel-quality gain. Baseline speedups
measure handwritten kernels, separately from model-generated results. See the
[final report](RESULTS.md) and [paired held-out comparison](experiments/heldout_comparison/comparison.md).

## Run locally

Use Python 3.10 or newer from the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m kernels.op_registry --verify
python -m rl_trainer.train_grpo --dry-run --out eval/results/dry_run.jsonl
python -m eval.benchmark_report --input eval/results/dry_run.jsonl --out eval/out
```

Open `eval/out/report.html` to inspect traces and source, or read
`eval/out/summary.md`. The dry run grades four fixed fixture programs; it does
not generate candidates from a model or update weights. `eval/sample_traces.jsonl`
is synthetic format test data, explicitly marked `fixture: true`.

## Components and current scope

| Component | Implemented | Remaining validation or scope |
| :--- | :--- | :--- |
| `contracts/` | Shared Python/Rust evaluation record and JSON Schema | Keep field names stable across tracks |
| `kernels/` | 30 NumPy references, Triton baselines and native PyTorch references; 180/180 laptop fp32/bf16 checks passed | Generalization is limited to the registered canonical/stress cases |
| `sandbox_runner/` | AST checks, child-process timeout, input mutation checks, multiple seeds and stress inputs, finite JSON | Development runner, not a security sandbox; Windows has no hard memory cap |
| `rl_trainer/` | Bounded search/repair, CUDA grading, clipped GRPO, completed full-parameter 0.49B and 7B QLoRA experiments, checkpoint support | FSDP and multi-node training are not implemented |
| `eval/` | Grouped metrics, failure counts, plots, offline viewer, Nsight CSV import and strict paired held-out comparison | Three held-out workloads per run are a small test set |

CPU records always use `backend: numpy_dev`. `compiled: true` means Python source
loaded successfully on this path. `compile_ms: 0` and `speedup_vs_compile: 0`
mean that a `torch.compile` comparison is unavailable. Failure records use finite
zero latency/error sentinels; inspect `status` and `error_message` before using
those numbers. Timing excludes input generation/copying; values vary between
iterations and every timed output is checked afterwards. These CPU timings are
development diagnostics, not CUDA measurements.

FLOP counts and bytes moved are analytical workload cost models. They are not
measured HBM traffic or exact hardware instruction counts; sparse and simulated
quantized references can execute dense work. The FP8 workload simulates scaled
low-precision values and does not establish native FP8 performance.

Run adversarial candidate code only in a separately isolated execution environment.
A Python AST filter and a child-process timeout do not provide OS-level containment.
The Rust CLI supervises the same Python development verifier; it does not add GPU
support or a container boundary. Linux workers apply a 2 GiB address-space limit;
this limit is platform-dependent.

See [architecture](ARCHITECTURE.md), [training usage](rl_trainer/README.md),
[kernel and CUDA validation usage](kernels/README.md),
[evaluation usage](eval/README.md), and [track status](TRACK_SYNC.md).

## Measured laptop baselines

RTX 5060 Laptop (8 GB), PyTorch 2.11/cu128, Triton Windows 3.6; 30 operators,
three input seeds, canonical/stress checks, and two precisions. All 180 checks
passed. CUDA graphs replay 16 calls, with 50 event measurements after warmup;
these are steady-state device timings, excluding Python dispatch and compilation.

| Precision | Geomean vs native PyTorch eager | Geomean vs verified torch.compile |
| :--- | ---: | ---: |
| fp32 | 2.494x (30 operators) | 1.089x (29 operators) |
| bf16 | 5.417x (30 operators) | 1.248x (29 operators) |

These are hand-written baseline comparisons, not trained-model results. Precision,
GPU and timing modes are grouped separately. Raw records, environment, per-op
values and seed ranges are retained alongside the [comparison chart](experiments/rtx5060/full30/report/baseline_comparison.png).
The SSM reference fails Windows Inductor compilation, so its compiled comparison
is unavailable; its Triton correctness and timing checks passed.

## Historical v1: laptop training experiment

Qwen2.5-Coder-0.5B-Instruct (494,032,768 parameters) completed eight groups / 16
samples in FP32 with Adafactor. One group produced a finite optimizer update,
with verified weight L2 change 0.85014 and a matching checksum after checkpoint
resume. All training candidates failed verification.

On the three held-out workloads, base pass@1, validation-selected best-of-8,
and trained pass@1 each passed **0/3**. This bounded run demonstrated the training
and evaluation pipeline, with no observed quality improvement. Best-of-8 has a
larger inference budget; only base and trained pass@1 use matched budgets. See
the [held-out comparison](experiments/laptop_fullparam/comparison/comparison.md)
and raw records for failure categories and per-operator results.

## Historical v1: free-T4 experiment

The 7B NF4/FP16 QLoRA run completed eight groups / 16 samples and four verified
optimizer updates to 5,046,272 adapter parameters. Independent comparison of
the saved initial and trained adapters measured L2 change 0.19911. Training
took 21 minutes 28 seconds on a free Tesla T4, followed by a 28 minute 37 second
held-out evaluation. All generated training candidates failed verification.

Base pass@1, validation-selected best-of-8 and trained pass@1 each passed **0/3**.
The raw [T4 records](experiments/colab_t4/qlora7b_fresh) retain source commit,
configuration, model revision, completion tokens, probabilities and checkpoint
identity. This is adapter training over a frozen 7B base, not full-parameter 7B
training. The initial failed worker/recovery attempts remain separate diagnostics.
