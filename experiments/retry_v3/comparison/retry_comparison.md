# KernelBench-RL frozen retry comparison

Each model, GPU, precision and configuration is reported separately. No pooled metric or significance claim is made.

The trained checkpoint combines a supervised bootstrap with GRPO. Base-to-trained changes cannot be attributed to RL alone. The separately evaluated SFT-only checkpoint describes the incremental comparison within this frozen schedule; N=3 does not establish a general causal effect.

Previously inspected development operators never enter fresh held-out denominators. Development Best-of-K is unavailable because it was not budgeted. All failures remain in pass-rate denominators; geometric mean speedups are correct-only.

Historical contract_v1 results and this contract_v2 retry use different prompts, operator cohorts and training protocols. Historical results are context, not a paired same-protocol before/after comparison.

## laptop_7b

Model: `Qwen/Qwen2.5-Coder-7B-Instruct` · revision `c03e6d358207e414f1eca0bb1891e29f1db0e242` · `fp16` · `NVIDIA GeForce RTX 5060 Laptop GPU`.

Config `0dfb0a14abad3067` · source commit `2c174abe268495159c10d538c92d071e7733828d` · prompt `contract_v2`.

SFT updates: 88; GRPO optimizer updates: 5. Identical checkpoint hashes are permitted when GRPO makes no weight update.

SFT checkpoint: `checkpoint:0:018e3d755dd3d5e80b077017a7602d0a45890ea4174e2e6c221d479862d9e4b8`; hybrid checkpoint: `checkpoint:16:592e6c37c97d170b87d50fd966ea99d610a9f2ba93739d1cc919f442080bb6c2`.

### Fresh held-out operators (N=3)

`cosine_similarity_1024`, `fused_bias_gelu_4096`, `matmul_bias_silu_512`

| Method | Budget | Passed | Fast vs eager | Fast vs compile | Correct-only geomean | Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `base_pass1` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | COMPILE_ERROR=3 |
| `base_best_of_8` | 8 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | COMPILE_ERROR=3 |
| `sft_only_checkpoint` | 1 | 2/3 | 2/3 | 2/3 | 5.5927x (2/3) | COMPILE_ERROR=1 |
| `trained_checkpoint` | 1 | 2/3 | 2/3 | 2/3 | 5.6764x (2/3) | COMPILE_ERROR=1 |

### Previously inspected development operators (N=3)

`rmsnorm_4096`, `int8_weight_only_gemm_512`, `scaled_dot_product_attn_128`

| Method | Budget | Passed | Fast vs eager | Fast vs compile | Correct-only geomean | Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `base_pass1` | 1 | 0/3 | 0/3 | 0/3 | unavailable (0 correct) | CHEAT=2, COMPILE_ERROR=1 |
| `sft_only_checkpoint` | 1 | 1/3 | 1/3 | 0/3 | 7.7957x (1/3) | CHEAT=1, COMPILE_ERROR=1 |
| `trained_checkpoint` | 1 | 1/3 | 1/3 | 0/3 | 7.9263x (1/3) | COMPILE_ERROR=2 |
| `base_best_of_8` | unavailable | unavailable | unavailable | unavailable | unavailable | Not budgeted |

Inventory: main_rows=39, main_selection_rows=24, main_fresh_final_rows=9, main_development_final_rows=6, sft_only_rows=6, sft_fresh_final_rows=3, sft_development_final_rows=3.
