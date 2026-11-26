# KernelBench-RL

Execution-feedback tooling for generating and checking numerical kernels, with
30 deterministic operator workloads, each with a Triton baseline and native
PyTorch reference.

The CPU pipeline runs locally: candidate source → bounded worker → numerical
checks → reward and group advantages → strict JSONL → report. The repository
also includes optional policy-search/training adapters and a CUDA baseline
validation path. Measured laptop baseline results are in
[`experiments/rtx5060/full30`](experiments/rtx5060/full30).
The 0.49B full-parameter training and held-out experiment completed and showed
no kernel-quality gain. The free-T4 7B QLoRA experiment is running. Baseline
speedups measure handwritten kernels, separately from model-generated results.

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
| `rl_trainer/` | Bounded search/repair, CUDA grading, clipped GRPO, full-parameter 0.49B and 7B QLoRA paths, verified optimizer/weight/RNG resume | Free-T4 experiment in progress; FSDP and multi-node training are not implemented |
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

## Completed laptop training experiment

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
