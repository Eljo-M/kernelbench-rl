# KernelBench-RL results

<!-- BEGIN KERNELBENCH_RL_V02 -->

## v0.2: completed corrected studies

Publication record: **2026-11-26**. These separately identified v2/v3 protocols use the custom 30-operator registry; they do not claim coverage of the complete official KernelBench benchmark. Original v1 results below remain historical evidence.

[Project website](https://eljo-m.github.io/kernelbench-rl/) · [Paper landing page](https://eljo-m.github.io/kernelbench-rl/paper.html) · [Searchable PDF](https://eljo-m.github.io/kernelbench-rl/assets/kernelbench-rl-paper.pdf) · [Study report](https://eljo-m.github.io/kernelbench-rl/report.html) · [Release index](https://github.com/Eljo-M/kernelbench-rl/releases)

The v2 schedule uses the dtype-aware `contract_v2` prompt, 32 training-only supervised updates, then eight groups of two completions with correctness-gated GRPO. Every non-PASS candidate receives zero training reward; valid PASS speed rewards are preserved rather than replaced by binary values. Equal rewards yield zero relative advantage; even correct groups can have equal or differing speed rewards. The SFT-only checkpoint is saved before GRPO. The original pre-SFT base anchors reference regularization.

The separate prospective **v3 7B rank 16** study starts from the pinned original base with fresh adapters, not transferred v2 weights or exact resume. Rank 16/alpha 32 targets q/k/v/o plus gate/up/down projections. It records 88 SFT updates over 11 training operators and 16 GRPO groups; actual GRPO updates are reported below. All three former v2 fresh targets are now training-seen and excluded from v3 fresh/development denominators. New prospective fresh operators are `cosine_similarity_1024`, `fused_bias_gelu_4096`, `matmul_bias_silu_512`. Their generated programs were not previously inspected in the recorded repository exposure audit; public fixed references and related families were already exposed. Rank, targets, schedule, seed and fresh tasks differ from v2, so no causal SFT/GRPO, model-capacity or pooled comparison follows.

| Model track | SFT updates | GRPO groups | GRPO candidates | GRPO updates | Training statuses |
| --- | --- | --- | --- | --- | --- |
| v2 0.49B FP32 | 32 | 8 | 16 | 0 | `COMPILE_ERROR`=14, `RUNTIME_ERROR`=2 |
| v2 7B rank 8 | 32 | 8 | 16 | 1 | `CHEAT`=1, `COMPILE_ERROR`=5, `NUMERICAL_MISMATCH`=3, `PASS`=5, `RUNTIME_ERROR`=2 |
| v3 7B rank 16 | 88 | 16 | 32 | 5 | `PASS`=31, `RUNTIME_ERROR`=1 |

V2 fresh held-out passes (N=3 per method; different fresh operators across protocols):

| Model track | Base pass@1 | Base best-of-8 | SFT-only pass@1 | Hybrid pass@1 |
| --- | --- | --- | --- | --- |
| v2 0.49B FP32 | 0/3 | 0/3 | 0/3 | 0/3 |
| v2 7B rank 8 | 0/3 | 0/3 | 2/3 | 2/3 |

V3 fresh held-out passes (N=3 per method; different fresh operators across protocols):

| Model track | Base pass@1 | Base best-of-8 | SFT-only pass@1 | Hybrid pass@1 |
| --- | --- | --- | --- | --- |
| v3 7B rank 16 | 0/3 | 0/3 | 2/3 | 2/3 |

Pass@1 methods use one completion; base best-of-eight uses eight validation-selection samples per fresh operator, followed by independent regrading of its winner. Selection records are excluded from final denominators. Failures remain in pass-rate and fast-versus-eager denominators; correct-only speedups and unavailable compiled comparisons are labeled separately.

- v2 0.49B FP32: 39 main rows = 24 selection + 9 fresh final + 6 development final; 6 separate SFT-only rows = 3 fresh + 3 development.
- v2 7B rank 8: 39 main rows = 24 selection + 9 fresh final + 6 development final; 6 separate SFT-only rows = 3 fresh + 3 development.
- v3 7B rank 16: 39 main rows = 24 selection + 9 fresh final + 6 development final; 6 separate SFT-only rows = 3 fresh + 3 development.

Previously inspected development passes, kept separate from the fresh cohort:

| Model track | Base pass@1 | SFT-only pass@1 | Hybrid pass@1 | Best-of-eight |
| --- | --- | --- | --- | --- |
| v2 0.49B FP32 | 0/3 | 0/3 | 0/3 | Unavailable |
| v2 7B rank 8 | 0/3 | 1/3 | 1/3 | Unavailable |
| v3 7B rank 16 | 0/3 | 1/3 | 1/3 | Unavailable |

V2 fresh targets were `l2_normalize_2048`, `fused_bias_silu_4096` and `matmul_residual_add_512`; they are training-seen in the separate v3 follow-up when present. Each prospective trio tests programs within shared training families, not unseen-family out-of-distribution behavior. Previously inspected RMSNorm-4096, INT8 weight-only GEMM and attention remain development regressions. Protocols, model/precision tracks and different fresh cohorts are not pooled. Historical `contract_v1` and later `contract_v2` outcomes are unpaired comparisons. The hybrid includes SFT and establishes no RL-only causal effect; N=3 and one training seed per study do not establish significance or broad generalization.

The laptop 7B studies update LoRA adapters over a frozen NF4 base with an FP16 GPU decoder and fixed FP32 CPU embedding/output computations. They are separate from both the historical free-T4 placement and full-parameter 0.49B FP32 training. FSDP, multi-node training and full-parameter 7B remain outside implemented and validated scope.

Zero GRPO updates were recorded for v2 0.49B FP32. Each corresponding final trainable-tensor digest equals its SFT-only digest. Sampled-output or timing differences between those arms are not RL parameter gains; skipped-update zero diagnostics are sentinels.

Fixed baselines are curated implementations prepared with coding-assistant support, distinct from programs sampled by the evaluated policies; historical files' 'handwritten' wording does not assert exclusively human authorship. The historical v1 **180/180** campaign and geometric means remain unchanged: FP32 **2.494x eager / 1.089x compiled**, BF16 **5.417x / 1.248x**, with 30 eager and 29 available compiled operators. The separate updated-worker correctness regression passed **60/60** operator/precision records at seed 142 and independent seed 10142. It checks all 30 operators in FP32/BF16 and does not re-estimate the historical speedup campaign.

| Model track | Training seconds | Parent peak allocated CUDA GiB |
| --- | --- | --- |
| v2 0.49B FP32 | 607.42 | 4.741 |
| v2 7B rank 8 | 2079.19 | 4.692 |
| v3 7B rank 16 | 4958.95 | 5.295 |

The CUDA peak is the training parent process's PyTorch allocator peak; it excludes child-worker and other applications' device allocations. A checksum-verified checkpoint reload restores optimizer/RNG state without extra training steps; this is not an independent next-step trajectory replication or a separate optimizer/RNG digest comparison.

Checkpoint tensors and optimizer/RNG files are excluded from Git. For versioned archives and SHA-256 manifests, see [GitHub Releases](https://github.com/Eljo-M/kernelbench-rl/releases).

The publication snapshot marks `experiments/**`, `docs/**` and `paper/**` as `-text` in `.gitattributes`, preserving exact saved bytes, existing line endings and embedded output/proof hashes in committed blobs and clean clones. Earlier Git snapshots used text normalization and may contain LF-normalized blobs where Windows saved files contained CRLF. Earlier release ZIPs are unchanged; their recorded hashes remain their own identities. This packaging correction does not rerun experiments or retroactively change study commits.

Evidence: [strict comparison](experiments/retry_v2/comparison/retry_comparison.md) · [strict JSON](experiments/retry_v2/comparison/retry_comparison.json) · [frozen protocol](experiments/retry_v2/protocol.json) · [reproduction guide](REPRODUCING_RETRY.md) · [manuscript source](paper/manuscript.json) · [worker regression](experiments/retry_v2/verifier_regression) · [offline final report](docs/report.html).

Frozen study source: `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`. The later publication snapshot contains documentation, saved-byte packaging and result assets; its distinct source identity is recorded in release provenance.

V3 evidence: [separate strict comparison](experiments/retry_v3/comparison/retry_comparison.md) · [separate strict JSON](experiments/retry_v3/comparison/retry_comparison.json) · [prospective protocol](experiments/retry_v3/protocol.json) · [registered source provenance](experiments/retry_v3/source_provenance.json). V3 study source prefix `2c174abe2684` is 12 characters only; full source/configuration/tensor identities are retained in the linked manifests and manuscript metadata.

<!-- END KERNELBENCH_RL_V02 -->

## Historical v1: original free-resource results

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

The live T4 resume probe restored checkpoint 8, all four optimizer steps and RNG
state with the exact recorded trainable-weight checksum. It performed no
additional training. The [live verification summary](experiments/colab_t4/live_verification.json)
records this result separately from the original training run.

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
without changing generation's cache configuration. A live T4 diagnostic on the
recorded 395-token initial-policy sample reproduced the exact **0.0133743286**
maximum difference by changing only the cache flag. The original stored values
matched cache-enabled scoring exactly; repeated cache-disabled evaluation and
the gradient-enabled cache-disabled training forward matched exactly. This
supports a cache-path cause on that sample. It does not retroactively relabel
or rerun the historical experiments, and the recovery tolerance remains 0.001.

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
