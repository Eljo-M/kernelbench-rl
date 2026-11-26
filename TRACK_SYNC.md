# Track coordination

Project: `Eljo-M/kernelbench-rl`. Integration branch: `codex/gpu-benchmarks-training`.
This file records the actual implementation state and remaining experiments.

| Track | State | Remaining work |
| :--- | :--- | :--- |
| Runner | CPU worker, timeout, reduced environment, static checks, multi-seed/stress checks, mutation and timing validation; Rust supervisory CLI verified in CI | Supply external isolation for adversarial execution |
| Kernels | All 30 Triton baselines integrated; final canonical/stress CUDA suite 180/180 PASS in fp32/bf16 on three seeds | Windows SSM compiled comparison unavailable; 29 comparable operators |
| Training | Real 0.49B full-parameter run and held-out evaluation completed; exact checkpoint resume verified; free 7B QLoRA has performed verified updates | Finish T4 held-out evaluation; FSDP/multi-node training not implemented |
| Evaluation | Strict ingestion, grouped summaries, plots, offline viewer, profiler import and paired held-out report | Three held-out workloads per model limit generalization claims |

## 2026-11-26 integration

- @Track2 supplied the operator expansion, baseline sources, PyTorch references and
  CUDA validator, then fixed review findings in strict JSON, bf16 input rounding,
  output overwrite behavior, metadata, lazy compilation status and timing.
- @Lead integrated the files, removed estimated `torch.compile` timings, preserved
  `numpy_dev` labels, bounded CPU execution, and added schema/timeout regressions.
- @Track3 implemented and checked the local rollout/objective/update plumbing. A
  tiny local policy fixture checks parameter updates and checkpoint writes. This
  test is not a learned large-model result.
- @Track4 implemented reporting and checked the offline viewer against rollout
  source records. Synthetic sample traces and dry-run fixtures are explicitly
  labeled and must not be used as performance evidence.
- Native PyTorch references pass 50 numerical comparisons on CPU across canonical
  and stress cases in fp32/bf16. Triton compilation and GPU performance remain
  unverified locally. Cost models are analytical estimates, not measured HBM data.
- All 67 local Python tests passed. The initial implementation commit
  `5f9299075b869873b8a5d7365d0a98ef7f318bc0` passed
  [GitHub CI](https://github.com/Eljo-M/kernelbench-rl/actions/runs/45870454904),
  including Python tests, operator verification, report generation and Rust tests.

## Current experiment

Finish the fresh free-T4 study and collect the checkpoint and held-out records.
The corrected worker passed a real T4 preflight. The initial attempt produced
no updates because child processes lacked the Linux driver environment; the
failed recovery also stopped before any update after a probability mismatch.
Both attempts remain separate from fresh training results. Only final test-phase
records enter comparisons, with equal pass@1 budgets for base and trained models.

## Free-resource experiments

- RTX 5060 Laptop: final 180/180 baseline checks passed for 30 operators, fp32/bf16 and
  seeds 42/43/44. CUDA graph timings and dispatch-sensitive event timings are
  retained separately under `experiments/rtx5060`.
- Free Colab T4: 33/33 fp32 baseline checks passed for the same 11 operators and
  three seeds. Source commit and raw records are under `experiments/colab_t4`.
- The 0.49B full-parameter FP32/Adafactor experiment completed eight groups and
  one optimizer update. The full-model weight change was verified; all 16
  generated training candidates failed. This establishes an execution run,
  not a learned kernel-quality gain. Checkpoints are stored locally and excluded
  from Git. Held-out base pass@1, best-of-8 and trained checkpoint each passed
  0/3; the result shows no quality improvement. The real checkpoint restored
  model weights, optimizer and RNG with a matching trainable-weight checksum.
- The 7B NF4/rank-8 QLoRA run uses a free T4 with 5,046,272 trainable adapter
  parameters. Fresh training has performed two finite, nonzero adapter updates;
  completion and held-out quality results have not yet been verified.
- Commit `3e9de7c9bf1cd83a523282e26e41feaf4505c66b` passed
  [CI](https://github.com/Eljo-M/kernelbench-rl/actions/runs/45877414778), including
  actual Torch backward, optimizer and checkpoint tests.
