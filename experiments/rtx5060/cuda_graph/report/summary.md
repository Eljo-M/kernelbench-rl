# KernelBench-RL Evaluation Summary

This report describes the supplied records; it does not establish a controlled training ablation.

| Metric | Value |
| :--- | ---: |
| Total rollouts | 66 |
| Compile rate | 100.0% |
| Correctness rate | 100.0% |
| Fast_1.0x (vs eager; all-rollout denominator) | 95.5% |
| Fast_compile | 72.7% of 66 comparable records |
| Timed correct candidates | 66 |
| Geomean speedup (timed correct only) | 3.6894x |
| Geomean speedup 95% interval | Unavailable (requires ≥2 seeds) |

Speedup excludes failed, cheating, or untimed records. Zero baseline latency means unavailable. The interval resamples whole seeds with equal weight; it is conditional on timed correct records and is not a paired ablation test.

## Method, backend, precision, and level

| Method | Backend | Dtype | GPU | Timing | Level | Fixture | N | Compile % | Correct % | Fast_1.0x % | Correct-only geomean | 95% interval |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| triton_baseline | triton | bf16 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 1 | False | 15 | 100.0 | 100.0 | 100.0 | 10.9217x | 10.9217x [10.8478, 11.032] (3 seeds) |
| triton_baseline | triton | fp32 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 1 | False | 15 | 100.0 | 100.0 | 100.0 | 2.8798x | 2.8798x [2.8447, 2.9132] (3 seeds) |
| triton_baseline | triton | bf16 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 2 | False | 15 | 100.0 | 100.0 | 100.0 | 4.1101x | 4.1101x [4.0481, 4.1853] (3 seeds) |
| triton_baseline | triton | fp32 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 2 | False | 15 | 100.0 | 100.0 | 100.0 | 2.4402x | 2.4402x [2.4096, 2.4819] (3 seeds) |
| triton_baseline | triton | bf16 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 3 | False | 3 | 100.0 | 100.0 | 100.0 | 1.6338x | 1.6338x [1.5876, 1.7121] (3 seeds) |
| triton_baseline | triton | fp32 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 3 | False | 3 | 100.0 | 100.0 | 0.0 | 0.5823x | 0.5823x [0.5695, 0.5901] (3 seeds) |

## Failure counts

| Failure | Count |
| :--- | ---: |
| None | 0 |

GPU throughput plots use only correct GPU records containing positive achieved_tflops and arithmetic_intensity. Hardware ceilings are drawn only when explicitly supplied. FLOP/byte estimates depend on the operator cost model.
