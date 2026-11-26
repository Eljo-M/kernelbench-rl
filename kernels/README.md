# Kernel workloads and CUDA validation

`python -m kernels.op_registry --list` lists 30 deterministic NumPy workloads.
`python -m kernels.op_registry --verify` checks reference determinism, output
finiteness, and the positive analytical workload cost models. It does not compile
or run Triton.

All 30 registry entries have Triton baseline source and native PyTorch references.
`tests/test_torch_references.py` checks their canonical and stress cases on CPU in
fp32 and bf16 when PyTorch is installed. Integer quantization inputs keep their
integer dtype. The bf16 NumPy oracle receives the same rounded inputs as PyTorch.

## CUDA run

Use a compatible CUDA/PyTorch/Triton environment. Native Windows was validated on
an RTX 5060 Laptop with torch 2.11/cu128 and triton-windows 3.6. Install dependencies in your own virtual
environment, then run:

```bash
python -m pip install -e ".[gpu,dev]"
python -m kernels.validate_cuda --ops rmsnorm_2048 --dtypes fp32 bf16 \
  --out eval/results/rmsnorm_cuda.jsonl
python -m kernels.validate_cuda --out eval/results/triton_baselines_cuda.jsonl
python -m eval.benchmark_report --input eval/results/triton_baselines_cuda.jsonl --out eval/out
```

The validator executes repository-owned baseline source, not arbitrary generated
kernels. It checks NumPy and PyTorch agreement before timing canonical GPU inputs,
uses CUDA graph replay by default (16 calls, event-median timing), and checks compiled-reference correctness before accepting a
`torch.compile` comparison. Unsupported/failed comparisons remain unavailable.
Failure records carry finite JSON values and diagnostics. Missing CUDA/Triton
exits with code 2 and writes no output file. Existing files require `--overwrite`.

The output includes seed, dtype, method and GPU model. FLOPs follow the workload
model; successful arithmetic intensity uses the actual input/output tensor byte
sizes. This is an idealized traffic lower bound, not measured DRAM traffic.

The current validation covers registered canonical/stress inputs. It is not a
general tensor shape/device validator for arbitrary callers. Live Triton numerical
execution is recorded for all 30 baseline entries in fp32/bf16 across three input
seeds: 180/180 checks passed on the laptop. Raw records and per-op comparisons are
bundled under `experiments/rtx5060/full30`. The SSM reference's Windows Inductor
compilation is unavailable; compiled speedup summaries cover 29 operators.
`--timing-mode events` retains
the separate dispatch-sensitive timing path. Graph timings report steady-state
device execution, excluding host dispatch and JIT compilation.

Generated candidates use `sandbox_runner.gpu_verifier.verify_candidate_gpu`,
which imports a real source file in a child process, checks canonical, stress and
independent-seed inputs, rejects input mutation, and applies a wall-clock timeout.
The GPU-specific source policy requires a reachable Triton load/store kernel and
rejects host Torch arithmetic, reference fallbacks and output memoization. A
source-policy checksum accompanies generated-candidate records.
The child uses the same graph timing procedure. This is a trusted development
boundary; execute adversarial code in a separately isolated environment.
