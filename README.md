# KernelBench-RL

Execution-feedback tooling for generating and checking numerical kernels, with
30 deterministic operator workloads and 11 baseline-covered Triton entries.

The CPU pipeline runs locally: candidate source → bounded worker → numerical
checks → reward and group advantages → strict JSONL → report. The repository
also includes optional policy-search/training adapters and a CUDA baseline
validation path. Measured laptop baseline results are in
[`experiments/rtx5060/cuda_graph`](experiments/rtx5060/cuda_graph).
Real-model training experiments are in progress; these baseline results do not
establish learning gains or distributed training results.

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
| `kernels/` | 30 NumPy workload references; 11 Triton entries with native PyTorch references; 66/66 laptop fp32/bf16 checks passed | GPU results cover the 11 baseline entries, not all 30 workloads |
| `sandbox_runner/` | AST checks, child-process timeout, input mutation checks, multiple seeds and stress inputs, finite JSON | Development runner, not a security sandbox; Windows has no hard memory cap |
| `rl_trainer/` | Bounded asynchronous search/repair, group rewards, optional single-process on-policy GRPO update and checkpoint code | Large-model, CUDA-kernel and distributed training experiments remain unverified |
| `eval/` | Metrics by method/backend/level, seed intervals, failure counts, PNG plots, offline viewer, Nsight CSV import | Controlled held-out ablations require actual independent runs |

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

RTX 5060 Laptop (8 GB), PyTorch 2.11/cu128, Triton Windows 3.6; 11 operators,
three input seeds, canonical/stress checks, and two precisions. All 66 checks
passed. CUDA graphs replay 16 calls, with 50 event measurements after warmup;
these are steady-state device timings, excluding Python dispatch and compilation.

| Precision | Geomean vs native PyTorch eager | Geomean vs verified torch.compile |
| :--- | ---: | ---: |
| fp32 | 2.310x | 1.024x |
| bf16 | 5.893x | 1.139x |

These are hand-written baseline comparisons, not trained-model results. Precision,
GPU and timing modes are grouped separately. Raw records, environment, per-op
values and seed ranges are retained alongside the [comparison chart](experiments/rtx5060/cuda_graph/report/baseline_comparison.png).
