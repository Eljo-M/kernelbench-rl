# KernelBench-RL Held-Out Evaluation Comparison

## Scope and Methodological Notes

- **Small held-out set:** Each run evaluates only its configured held-out operators (N=3, N=3). No statistical significance claims are made.
- **Different inference search budgets:** `base_pass1` (k=1) and `base_best_of_8` (k=8, validation-seed selected and regraded on the independent test seed) use different test-time sampling budgets; only `base_pass1` vs `trained_checkpoint` are matched at k=1.
- **Separate model, precision, and GPU tracks:** Runs are never pooled across different models, training modes (`Full-Parameter 0.49B` vs `7B QLoRA (NF4)`), dtypes (`fp32`/`bf16`/`fp16`), or GPUs (e.g., laptop GPU vs Colab T4).
- **Denominator and speedup definitions:** Failures remain in the denominator for pass rate, `fast_vs_eager`, and `fast_vs_compile`. Passing rows where `torch.compile` was unavailable are tracked separately rather than counted as fast or silently dropped. Geomean speedups are explicitly **correct-only**.
- **Strict test-phase filtering:** Only `eval_phase=heldout_test` non-fixture, non-oracle rows are included; `selection_val`, fixture, and oracle rows are excluded.

## Run: laptop_fullparam — Full-Parameter 0.49B (FP32 on NVIDIA GeForce RTX 5060 Laptop GPU)

- **Model:** `Qwen/Qwen2.5-Coder-0.5B-Instruct` (revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`)
- **Training Mode:** Full-Parameter 0.49B (`full_param_0.49b`)
- **Precision / Backend / GPU:** `fp32` / `triton` / `NVIDIA GeForce RTX 5060 Laptop GPU`
- **Config Hash:** `b06d4b72437f6cb2` | **Source Commit:** `03468e3c717e8b8d90b40766e3eff80eff9403b3`
- **Checkpoint Weights SHA:** `14e3f9d1c22f78473f97a59f94366d197e4c1ed32e285bc4088e5030ef4a0a00`
- **Held-Out Operators (N=3):** `rmsnorm_4096`, `int8_weight_only_gemm_512`, `scaled_dot_product_attn_128`
- **Excluded Non-Test Rows:** selection_val=24, fixture=0, oracle=0, other_eval_phase=0

| Method | Budget | Pass (k/N) | Fast vs Eager | Fast vs Compile (Compile Unavail. Passing) | Geomean vs Eager (Correct-Only) | Geomean vs Compile (Correct-Only) | Failures |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `base_pass1` | k=1 | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | N/A (0/3 correct) | N/A (0/3 correct) | 3 (CHEAT=3) |
| `base_best_of_8` | k=8 | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | N/A (0/3 correct) | N/A (0/3 correct) | 3 (COMPILE_ERROR=3) |
| `trained_checkpoint` | k=1 | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | N/A (0/3 correct) | N/A (0/3 correct) | 3 (COMPILE_ERROR=2, CHEAT=1) |

### Paired Per-Operator Results

| Operator | Seed | `base_pass1` (k=1) | `base_best_of_8` (k=8) | `trained_checkpoint` (k=1) | Paired (`pass1` vs `trained`) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `rmsnorm_4096` | 50042 | CHEAT (gpu=unavailable) | COMPILE_ERROR (gpu=unavailable) | COMPILE_ERROR (gpu=unavailable) | `both_fail` |
| `int8_weight_only_gemm_512` | 50043 | CHEAT (gpu=unavailable) | COMPILE_ERROR (gpu=unavailable) | COMPILE_ERROR (gpu=unavailable) | `both_fail` |
| `scaled_dot_product_attn_128` | 50044 | CHEAT (gpu=unavailable) | COMPILE_ERROR (gpu=unavailable) | CHEAT (gpu=unavailable) | `both_fail` |

## Run: qlora7b_fresh — 7B QLoRA (NF4) (FP16 on Tesla T4)

- **Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`)
- **Training Mode:** 7B QLoRA (NF4) (`qlora_7b_nf4`)
- **Precision / Backend / GPU:** `fp16` / `triton` / `Tesla T4`
- **Config Hash:** `5ec1b65de79cdea6` | **Source Commit:** `e72c42e794643ed4c198937e86427c3640b977c5`
- **Checkpoint Weights SHA:** `18281524fcaa3c625a8b77bb617b8d5cb812e8f2c73a6892953fe7c77abbfa46`
- **Held-Out Operators (N=3):** `rmsnorm_4096`, `int8_weight_only_gemm_512`, `scaled_dot_product_attn_128`
- **Excluded Non-Test Rows:** selection_val=24, fixture=0, oracle=0, other_eval_phase=0

| Method | Budget | Pass (k/N) | Fast vs Eager | Fast vs Compile (Compile Unavail. Passing) | Geomean vs Eager (Correct-Only) | Geomean vs Compile (Correct-Only) | Failures |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `base_pass1` | k=1 | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | N/A (0/3 correct) | N/A (0/3 correct) | 3 (COMPILE_ERROR=2, CHEAT=1) |
| `base_best_of_8` | k=8 | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | N/A (0/3 correct) | N/A (0/3 correct) | 3 (COMPILE_ERROR=3) |
| `trained_checkpoint` | k=1 | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | N/A (0/3 correct) | N/A (0/3 correct) | 3 (COMPILE_ERROR=2, CHEAT=1) |

### Paired Per-Operator Results

| Operator | Seed | `base_pass1` (k=1) | `base_best_of_8` (k=8) | `trained_checkpoint` (k=1) | Paired (`pass1` vs `trained`) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `rmsnorm_4096` | 50042 | COMPILE_ERROR (gpu=unavailable) | COMPILE_ERROR (gpu=unavailable) | COMPILE_ERROR (gpu=unavailable) | `both_fail` |
| `int8_weight_only_gemm_512` | 50043 | COMPILE_ERROR (gpu=Tesla T4) | COMPILE_ERROR (gpu=Tesla T4) | COMPILE_ERROR (gpu=unavailable) | `both_fail` |
| `scaled_dot_product_attn_128` | 50044 | CHEAT (gpu=unavailable) | COMPILE_ERROR (gpu=unavailable) | CHEAT (gpu=unavailable) | `both_fail` |
