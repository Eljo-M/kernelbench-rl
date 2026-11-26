# Track coordination

Project: `Eljo-M/kernelbench-rl`. Integration branch: `codex/integrated-pipeline`.
This file records the actual implementation state and remaining experiments.

| Track | State | Remaining work |
| :--- | :--- | :--- |
| Runner | CPU worker, timeout, reduced environment, static checks, multi-seed/stress checks, mutation and timing validation; Rust supervisory CLI verified in CI | Supply external isolation for adversarial execution |
| Kernels | 30 NumPy operators; 11 Triton baseline entries and native PyTorch references; CUDA validation CLI | Execute Triton fp32/bf16 checks and event timing on a compatible CUDA host |
| Training | CPU fixtures, bounded search/repair, single-process on-policy update/checkpoint implementation | Full model validation, CUDA-kernel grading, held-out ablations; distributed/LoRA and resume not implemented |
| Evaluation | Strict trace ingestion, grouped summaries, seed intervals, profiler CSV import, plots, offline viewer | Use actual independent GPU runs for performance and learning claims |

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

## Next experiment

Run `python -m kernels.validate_cuda` in the environment described in
[kernels/README.md](kernels/README.md). Keep the resulting JSONL with GPU/dtype/seed
metadata, investigate every failure, and only then connect CUDA grading to the
training loop. Compare held-out base pass@1, best-of-group search and a trained
checkpoint using the same operators, seeds and budgets. Do not infer gains from
fixtures or a checkpoint file.
