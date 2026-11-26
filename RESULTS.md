# KernelBench-RL: final free-resource experiment report

The implementation, GPU baseline suite, two bounded model-training runs and
paired held-out evaluations completed using the laptop and a free Colab T4.
All 30 handwritten Triton baselines passed the final laptop suite. Both trained
models passed **0/3** held-out workloads, as did their untrained and best-of-8
comparisons. The experiments demonstrate an executable pipeline and real weight
updates; they do not demonstrate improved generated-kernel quality.

## Handwritten GPU baselines

The final RTX 5060 Laptop suite contains 30 operators × two precisions × three
input seeds (42/43/44): **180/180 PASS**. Canonical and registered stress cases,
NumPy/native-Torch agreement, input mutation and compiled-reference correctness
are checked. CUDA graph replay uses 16 calls, 50 event measurements and warmup;
reported steady-state timings exclude Python dispatch and JIT compilation.

| Precision | Geomean vs native eager | Geomean vs verified torch.compile |
| :--- | ---: | ---: |
| FP32 | 2.494× across 30 operators | 1.089× across 29 operators |
| BF16 | 5.417× across 30 operators | 1.248× across 29 operators |

Windows Inductor fails to compile the SSM reference. Its compiled comparison is
unavailable; its Triton correctness/timing checks passed. The original 11-op
suite also passed **33/33** FP32 checks across three seeds on the free T4.
Hardware, precision and timing modes remain separate. These speedups describe
handwritten baselines, not learned policies. Analytical FLOP/byte estimates do
not measure HBM traffic; the simulated FP8 workload does not establish native
FP8 performance.

Raw data and plots: [final laptop suite](experiments/rtx5060/full30),
[T4 baseline suite](experiments/colab_t4/baselines).

## Actual model training

| Track | Parameters updated | Precision / optimizer | Samples / groups | Verified optimizer updates |
| :--- | ---: | :--- | :---: | ---: |
| Laptop Qwen2.5-Coder-0.5B | All 494,032,768 | FP32 / Adafactor | 16 / 8 | 1 |
| Free-T4 Qwen2.5-Coder-7B QLoRA | 5,046,272 adapters; frozen NF4 base | FP16 compute / FP32 adapter AdamW | 16 / 8 | 4 |

The full-parameter run measured weight L2 change **0.85014** and verified exact
model, optimizer and RNG restoration at checkpoint 8. The T4 run completed in
**1,287.75 seconds**, peaking at **10,282,289,664 allocated CUDA bytes**.
Independent CPU comparison of matching saved adapter tensors measured total
L2 change **0.19911** from its initial checkpoint. Adapter artifact hashes and
trainable-model hashes are recorded separately because they hash different
representations.

All 16 training candidates failed verification in each run. The T4 categories
were eight source-policy rejections (`CHEAT`), seven compile errors and one
runtime error from generated `tl.meshgrid` usage. A `CHEAT` label records a
prohibited source construct; it is not evidence of a correct or accelerated
kernel. Updates arose from differing failure rewards, not successful kernels.

The T4 trained model checksum is
`18281524fcaa3c625a8b77bb617b8d5cb812e8f2c73a6892953fe7c77abbfa46`.
The laptop checksum is
`14e3f9d1c22f78473f97a59f94366d197e4c1ed32e285bc4088e5030ef4a0a00`.

## Held-out results

Each model evaluates three disjoint held-out workloads: RMSNorm with a new width,
INT8 weight-only GEMM and scaled-dot-product attention. Test seeds are
50042/50043/50044. Best-of-8 selection uses a separate validation seed and then
regrades only its winner on the test seed. Per model, the raw log has 24
selection records and nine final test records; selection records, fixtures and
oracles are excluded from the comparison.

| Model / hardware | Base pass@1 (k=1) | Base best-of-8 (k=8) | Trained pass@1 (k=1) |
| :--- | :---: | :---: | :---: |
| Full-parameter 0.49B / laptop FP32 | 0/3 | 0/3 | 0/3 |
| 7B QLoRA / free T4 FP16 | 0/3 | 0/3 | 0/3 |

Failures remain in denominators. With no correct generated candidates, learned
speedup geometric means are **unavailable**, not measured zero-speedup results.
Only base and trained pass@1 have matched inference budgets. The two model /
hardware / precision tracks are not pooled or treated as a hardware comparison.
N=3 and one training seed per model do not support significance or broad
generalization claims.

See the [validated paired report](experiments/heldout_comparison/comparison.md),
[strict JSON](experiments/heldout_comparison/comparison.json) and
[comparison plot](experiments/heldout_comparison/comparison.png).

## Provenance, failures and reproduction

T4 training and evaluation used source commit
`e72c42e794643ed4c198937e86427c3640b977c5`, model revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`, and config hash
`5ec1b65de79cdea6`. The laptop model revision is
`ea3f2471cf1b1f0db85067f1ef93848e38e88c25`; its config hash is
`b06d4b72437f6cb2`. Each run retains its original environment and source metadata.

The initial T4 attempt made zero updates because GPU child processes lacked the
driver-library environment. A subsequent recovery rejected old/recomputed
probabilities differing by 0.013374 at tolerance 0.001, before updating. Both
remain diagnostics, separate from the fresh trained experiment. The later
`73a9220` scoring fix explicitly disables KV caching for teacher-forced scoring
without changing generation's cache configuration. It does not retroactively
relabel the historical experiments or prove a cause for the earlier mismatch.

All **134 local tests passed**, including real tiny-Torch backward, optimizer,
checkpoint, cache and reporting regressions. The scoring implementation passed
[GitHub CI](https://github.com/Eljo-M/kernelbench-rl/actions/runs/45896849506),
including the Rust supervisory CLI. The [free-T4 notebook](notebooks/free_t4.ipynb)
defaults to a preflight and provides an explicit full-run configuration.
Pinned dependencies, raw data, checkpoint identities and report regeneration
commands are in the [experiment index](experiments/README.md).

## Scope limits

This release completes the bounded single-GPU experiments described above. It
does not establish converged training, generated-kernel performance gains,
full-parameter 7B training, FSDP/multi-node operation or native FP8 performance.
The source-policy checker and sanitized child process are development guards,
not OS-level containment for adversarial programs. Free Colab availability is
not guaranteed. Further training and stronger generated kernels are research
work beyond the measured results in this release.
