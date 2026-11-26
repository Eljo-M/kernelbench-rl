# Corrected, resource-constrained retry

`protocol.json` freezes the training schedule, configuration hashes and evaluation
split before either quality study. The previous three test workloads have been
inspected and are now development regressions. The three fresh held-out workloads
are evaluated separately. This is a custom workload suite, not a complete run of
the original KernelBench benchmark.

The retry adds training-only supervised warmup, a consistent dtype-aware prompt,
correctness-gated rewards, strict old/new policy likelihood checks and an SFT-only
ablation. The hybrid result cannot establish an RL-only causal improvement.

The free Colab GPU quota was unavailable. Both configurations use the same laptop
sequentially: 0.49B full-parameter FP32 training, then 7B NF4 QLoRA with fixed CPU
embedding/output computations and GPU decoder/adapters. The 7B base is frozen.
`t4/config.json` is a planned, unused configuration, not evidence of a T4 retry.

Preflight files establish that the eight training target kernels pass the GPU
verifier. The 7B readiness probe establishes fit and a finite parameter update on
one training example; its weights were discarded. Neither is a held-out quality
result. Failed sandbox launch attempts are preserved separately from the passing
kernel preflight.

Run from the repository root with the recorded CUDA/Triton dependencies:

```powershell
python -m rl_trainer.run_study --config experiments/retry_v2/laptop/config.json --threads 4
python -m rl_trainer.run_study --config experiments/retry_v2/laptop_7b/config.json --threads 4
python -m eval.compare_retry --run-dir experiments/retry_v2/laptop --run-dir experiments/retry_v2/laptop_7b --out experiments/retry_v2/comparison
```

Use `--resume-study` only with the exact recorded source/configuration and durable
checkpoint. Interrupted warmup without a durable SFT checkpoint cannot be silently
repeated. Historical results and exports remain separate and unchanged.

Each completed study records source/configuration identity, actual optimizer
steps, checkpoint checksums, exact model/optimizer/RNG restoration, fresh held-out
evaluation, development regressions and the SFT-only control. QLoRA studies also
check frozen parameter, buffer and quantization-state integrity. These are bounded
empirical and integrity checks, not a formal correctness proof or an OS security
boundary.
