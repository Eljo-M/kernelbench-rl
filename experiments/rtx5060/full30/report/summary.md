# KernelBench-RL Evaluation Summary

This report describes the supplied records; it does not establish a controlled training ablation.

| Metric | Value |
| :--- | ---: |
| Total rollouts | 180 |
| Compile rate | 100.0% |
| Correctness rate | 100.0% |
| Fast_1.0x (vs eager; all-rollout denominator) | 94.4% |
| Fast_compile | 64.9% of 174 comparable records |
| Timed correct candidates | 180 |
| Geomean speedup (timed correct only) | 3.6756x |
| Geomean speedup 95% interval | Unavailable (requires ≥2 seeds) |

Speedup excludes failed, cheating, or untimed records. Zero baseline latency means unavailable. The interval resamples whole seeds with equal weight; it is conditional on timed correct records and is not a paired ablation test.

## Method, backend, precision, and level

| Method | Backend | Dtype | GPU | Timing | Level | Fixture | N | Compile % | Correct % | Fast_1.0x % | Correct-only geomean | 95% interval |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| triton_baseline | triton | bf16 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 1 | False | 36 | 100.0 | 100.0 | 100.0 | 10.1313x | 10.1313x [10.0267, 10.2691] (3 seeds) |
| triton_baseline | triton | fp32 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 1 | False | 36 | 100.0 | 100.0 | 88.9 | 2.5113x | 2.5113x [2.4729, 2.5318] (3 seeds) |
| triton_baseline | triton | bf16 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 2 | False | 30 | 100.0 | 100.0 | 100.0 | 3.5652x | 3.5652x [3.5029, 3.6828] (3 seeds) |
| triton_baseline | triton | fp32 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 2 | False | 30 | 100.0 | 100.0 | 100.0 | 2.4345x | 2.4345x [2.3868, 2.4928] (3 seeds) |
| triton_baseline | triton | bf16 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 3 | False | 24 | 100.0 | 100.0 | 100.0 | 3.5724x | 3.5724x [3.5317, 3.6413] (3 seeds) |
| triton_baseline | triton | fp32 | NVIDIA GeForce RTX 5060 Laptop GPU | cuda_graph | 3 | False | 24 | 100.0 | 100.0 | 75.0 | 2.5442x | 2.5442x [2.5081, 2.5705] (3 seeds) |

## Failure counts

| Failure | Count |
| :--- | ---: |
| None | 0 |

GPU throughput plots use only correct GPU records containing positive achieved_tflops and arithmetic_intensity. Hardware ceilings are drawn only when explicitly supplied. FLOP/byte estimates depend on the operator cost model.
