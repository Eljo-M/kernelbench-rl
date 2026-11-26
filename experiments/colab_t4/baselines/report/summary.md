# KernelBench-RL Evaluation Summary

This report describes the supplied records; it does not establish a controlled training ablation.

| Metric | Value |
| :--- | ---: |
| Total rollouts | 33 |
| Compile rate | 100.0% |
| Correctness rate | 100.0% |
| Fast_1.0x (vs eager; all-rollout denominator) | 90.9% |
| Fast_compile | 27.3% of 33 comparable records |
| Timed correct candidates | 33 |
| Geomean speedup (timed correct only) | 2.8558x |
| Geomean speedup 95% interval | 2.8558x [2.798, 2.9009] (3 seeds) |

Speedup excludes failed, cheating, or untimed records. Zero baseline latency means unavailable. The interval resamples whole seeds with equal weight; it is conditional on timed correct records and is not a paired ablation test.

## Method, backend, precision, and level

| Method | Backend | Dtype | GPU | Timing | Level | Fixture | N | Compile % | Correct % | Fast_1.0x % | Correct-only geomean | 95% interval |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| triton_baseline | triton | fp32 | Tesla T4 | cuda_graph | 1 | False | 15 | 100.0 | 100.0 | 100.0 | 3.7379x | 3.7379x [3.7007, 3.8076] (3 seeds) |
| triton_baseline | triton | fp32 | Tesla T4 | cuda_graph | 2 | False | 15 | 100.0 | 100.0 | 100.0 | 2.843x | 2.843x [2.6778, 2.979] (3 seeds) |
| triton_baseline | triton | fp32 | Tesla T4 | cuda_graph | 3 | False | 3 | 100.0 | 100.0 | 0.0 | 0.7602x | 0.7602x [0.7468, 0.7828] (3 seeds) |

## Failure counts

| Failure | Count |
| :--- | ---: |
| None | 0 |

GPU throughput plots use only correct GPU records containing positive achieved_tflops and arithmetic_intensity. Hardware ceilings are drawn only when explicitly supplied. FLOP/byte estimates depend on the operator cost model.
