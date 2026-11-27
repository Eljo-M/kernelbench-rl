# KernelBench-RL frozen retry comparison

Each model, GPU, precision and configuration is reported separately. No pooled metric or significance claim is made.

The trained checkpoint combines a supervised bootstrap with GRPO. Base-to-trained changes cannot be attributed to RL alone. The separately evaluated SFT-only checkpoint describes the incremental comparison within this frozen schedule; N=3 does not establish a general causal effect.

Previously inspected development operators never enter fresh held-out denominators. Development Best-of-K is unavailable because it was not budgeted. All failures remain in pass-rate denominators; geometric mean speedups are correct-only.

Historical contract_v1 results and this contract_v2 retry use different prompts, operator cohorts and training protocols. Historical results are context, not a paired same-protocol before/after comparison.

## laptop

Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct` · revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` · `fp32` · `NVIDIA GeForce RTX 5060 Laptop GPU`.

Config `76f5ae26c4334710` · source commit `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e` · prompt `contract_v2`.

SFT updates: 32; GRPO optimizer updates: 0. Identical checkpoint hashes are permitted when GRPO makes no weight update.

SFT checkpoint: `checkpoint:0:3f63aaff6dd8077f8f6a974fda3bdc10bc0ea73439550fd1f1252801bcbb523d`; hybrid checkpoint: `checkpoint:8:3f63aaff6dd8077f8f6a974fda3bdc10bc0ea73439550fd1f1252801bcbb523d`.

### Fresh held-out operators (N=3)

`l2_normalize_2048`, `fused_bias_silu_4096`, `matmul_residual_add_512`

| Method | Budget | Passed | Fast vs eager | Fast vs compile | Correct-only geomean | Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `base_pass1` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | CHEAT=3 |
| `base_best_of_8` | 8 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | COMPILE_ERROR=3 |
| `sft_only_checkpoint` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | COMPILE_ERROR=3 |
| `trained_checkpoint` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | COMPILE_ERROR=3 |

### Previously inspected development operators (N=3)

`rmsnorm_4096`, `int8_weight_only_gemm_512`, `scaled_dot_product_attn_128`

| Method | Budget | Passed | Fast vs eager | Fast vs compile | Correct-only geomean | Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `base_pass1` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | CHEAT=1, COMPILE_ERROR=2 |
| `sft_only_checkpoint` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | COMPILE_ERROR=3 |
| `trained_checkpoint` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | COMPILE_ERROR=3 |
| `base_best_of_8` | unavailable | unavailable | unavailable | unavailable | unavailable | Not budgeted |

Inventory: main_rows=39, main_selection_rows=24, main_fresh_final_rows=9, main_development_final_rows=6, sft_only_rows=6, sft_fresh_final_rows=3, sft_development_final_rows=3.

## laptop_7b

Model: `Qwen/Qwen2.5-Coder-7B-Instruct` · revision `c03e6d358207e414f1eca0bb1891e29f1db0e242` · `fp16` · `NVIDIA GeForce RTX 5060 Laptop GPU`.

Config `aac728c27b3b726c` · source commit `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e` · prompt `contract_v2`.

SFT updates: 32; GRPO optimizer updates: 1. Identical checkpoint hashes are permitted when GRPO makes no weight update.

SFT checkpoint: `checkpoint:0:c0216d947043e140a9158e077f414e0efb1f44456214ca978ef54ca3fb3e8c90`; hybrid checkpoint: `checkpoint:8:25537324dbd97f50e7af20cbf808e0aa4215d62236dc9b5f301d6d8e9b51ff08`.

### Fresh held-out operators (N=3)

`l2_normalize_2048`, `fused_bias_silu_4096`, `matmul_residual_add_512`

| Method | Budget | Passed | Fast vs eager | Fast vs compile | Correct-only geomean | Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `base_pass1` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | CHEAT=1, COMPILE_ERROR=2 |
| `base_best_of_8` | 8 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | COMPILE_ERROR=2, NUMERICAL_MISMATCH=1 |
| `sft_only_checkpoint` | 1 | 2/3 | 2/3 | 2/3 | 6.4038x (2/3) | NUMERICAL_MISMATCH=1 |
| `trained_checkpoint` | 1 | 2/3 | 2/3 | 2/3 | 6.3394x (2/3) | RUNTIME_ERROR=1 |

### Previously inspected development operators (N=3)

`rmsnorm_4096`, `int8_weight_only_gemm_512`, `scaled_dot_product_attn_128`

| Method | Budget | Passed | Fast vs eager | Fast vs compile | Correct-only geomean | Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `base_pass1` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | CHEAT=2, COMPILE_ERROR=1 |
| `sft_only_checkpoint` | 1 | 1/3 | 1/3 | 0/3 | 7.9333x (1/3) | CHEAT=1, RUNTIME_ERROR=1 |
| `trained_checkpoint` | 1 | 1/3 | 1/3 | 1/3 | 8.0557x (1/3) | COMPILE_ERROR=1, RUNTIME_ERROR=1 |
| `base_best_of_8` | unavailable | unavailable | unavailable | unavailable | unavailable | Not budgeted |

Inventory: main_rows=39, main_selection_rows=24, main_fresh_final_rows=9, main_development_final_rows=6, sft_only_rows=6, sft_fresh_final_rows=3, sft_development_final_rows=3.

## Historical context (different protocol; unpaired)

- laptop_fullparam: Qwen/Qwen2.5-Coder-0.5B-Instruct, fp32, NVIDIA GeForce RTX 5060 Laptop GPU; base_pass1=0/3, base_best_of_8=0/3, trained_checkpoint=0/3.
- qlora7b_fresh: Qwen/Qwen2.5-Coder-7B-Instruct, fp16, Tesla T4; base_pass1=0/3, base_best_of_8=0/3, trained_checkpoint=0/3.
