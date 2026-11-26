# KernelBench-RL Evaluation Summary

This report describes the supplied records; it does not establish a controlled training ablation.

Excluded 24 validation records used to select best-of-K candidates; metrics below use final test records.

| Metric | Value |
| :--- | ---: |
| Total rollouts | 9 |
| Compile rate | 0.0% |
| Correctness rate | 0.0% |
| Fast_1.0x (vs eager; all-rollout denominator) | 0.0% |
| Fast_compile | Unavailable |
| Timed correct candidates | 0 |
| Geomean speedup (timed correct only) | 0.0x |
| Geomean speedup 95% interval | Unavailable (requires ≥2 seeds) |

Speedup excludes failed, cheating, or untimed records. Zero baseline latency means unavailable. The interval resamples whole seeds with equal weight; it is conditional on timed correct records and is not a paired ablation test.

## Method, backend, precision, and level

| Method | Backend | Dtype | GPU | Timing | Level | Fixture | N | Compile % | Correct % | Fast_1.0x % | Correct-only geomean | 95% interval |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| base_best_of_8 | triton | fp16 | unavailable | cuda_graph | 1 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |
| base_best_of_8 | triton | fp16 | unavailable | cuda_graph | 2 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |
| base_best_of_8 | triton | fp16 | Tesla T4 | cuda_graph | 3 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |
| base_pass1 | triton | fp16 | unavailable | cuda_graph | 1 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |
| base_pass1 | triton | fp16 | unavailable | cuda_graph | 2 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |
| base_pass1 | triton | fp16 | Tesla T4 | cuda_graph | 3 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |
| trained_checkpoint | triton | fp16 | unavailable | cuda_graph | 1 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |
| trained_checkpoint | triton | fp16 | unavailable | cuda_graph | 2 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |
| trained_checkpoint | triton | fp16 | unavailable | cuda_graph | 3 | False | 1 | 0.0 | 0.0 | 0.0 | 0.0x | Unavailable (requires ≥2 seeds) |

## Failure counts

| Failure | Count |
| :--- | ---: |
| CHEAT | 2 |
| COMPILE_ERROR | 7 |

GPU throughput plots use only correct GPU records containing positive achieved_tflops and arithmetic_intensity. Hardware ceilings are drawn only when explicitly supplied. FLOP/byte estimates depend on the operator cost model.
