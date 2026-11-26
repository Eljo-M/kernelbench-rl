# Recorded experiments

These directories contain actual local/Colab records. Synthetic unit-test data
and dry-run fixtures are kept elsewhere and do not enter the results below.

| Directory | Hardware / scope | Interpretation |
| :--- | :--- | :--- |
| `rtx5060/full30` | RTX 5060 Laptop, all 30 operators, fp32/bf16, seeds 42/43/44 | Final 180/180 passing canonical/stress checks; authoritative laptop baseline suite |
| `rtx5060/cuda_graph` | Original 11 operators, same laptop, two precisions and three seeds | Earlier 66/66 passing baseline suite |
| `rtx5060/extended` | First 19-extension attempt | Incomplete; graph-capture errors retained as diagnostics, not final baseline evidence |
| `rtx5060/extended_v2` | Intermediate 19-extension suite | 114/114 passing before the additional stress cases in `full30` |
| `colab_t4/baselines` | Free Tesla T4, original 11 operators, fp32 and three seeds | 33/33 passing; separate hardware track |
| `colab_t4/failed_worker_attempt` | Initial 7B generation attempt | All child workers lacked CUDA driver environment; zero optimizer updates; not a trained result |
| `laptop_fullparam` | Full 494,032,768-parameter Qwen coder, FP32/Adafactor | Eight groups, 16 samples, one verified update; exact model/optimizer/RNG resume; held-out base/search/trained each 0/3 |

The T4 7B QLoRA training and held-out study is in progress. Completion will be
recorded only after actual optimizer, checkpoint and evaluation artifacts exist.

## Regenerate reports

From the repository root, using the ordinary report dependencies:

```bash
python -m eval.benchmark_report --input experiments/rtx5060/full30/baselines.jsonl --out experiments/rtx5060/full30/report
python -m eval.summarize_baselines --input experiments/rtx5060/full30/baselines.jsonl --out experiments/rtx5060/full30/report
python -m eval.compare_heldout --run-dir experiments/laptop_fullparam --out experiments/laptop_fullparam/comparison
```

The final suite reports geometric mean speedups across 30 operators versus
native eager and 29 versus verified `torch.compile`, separately per precision.
Windows Inductor cannot compile the SSM reference; its comparison is explicitly
unavailable. The Triton SSM kernel passed correctness checks. Graph replay times
exclude Python dispatch and compilation. Event-mode records are retained
separately and must not be pooled with graph replay.

Model-generated programs are graded with the same operator contracts, multiple
input seeds, input-mutation checks, source policy and timing procedure. Final
held-out comparisons use only `eval_phase=heldout_test`; best-of-8 selection uses
a separate validation seed. Failed candidates remain in pass-rate denominators.
No training-quality gain is inferred from a checkpoint or handwritten baseline.

Model/adapter, optimizer and RNG checkpoint files are excluded from Git. They
remain local or in exported Colab bundles. Raw completions, tokens, probabilities,
source hashes, configuration, pinned model revisions and environment metadata
preserve the experiment audit trail. GPU timing and learned quality depend on
these recorded configurations; this is a bounded experiment with three held-out
workloads, not a broad model or hardware ranking.
