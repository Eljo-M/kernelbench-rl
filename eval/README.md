# Evaluation reports

Generate a report from contract-compatible JSONL:

```bash
python -m eval.benchmark_report --input eval/results/run.jsonl --out eval/out
```

The output contains `summary.md`, `summary.json`, `speedups.png`, `roofline.png`,
and an offline `report.html`. The source `eval/report.html` also opens directly
and accepts a local JSONL file. Source, errors, and metric fields are rendered
as text. The viewer makes no network requests.

## Trace metadata

`contracts/kernel_eval.schema.json` defines required fields. Optional metadata
recognized by reporting includes:

| Field | Meaning |
| :--- | :--- |
| `method` | Explicit experimental arm, such as `base`, `best_of_8`, or `grpo` |
| `seed` | Integer or string identifying an independent run seed |
| `fixture` | Boolean identifying synthetic test data |
| `source_code` / `candidate_source` / `source` | Kernel source displayed in the viewer |
| `nsight` | Optional profiler metadata displayed in the full record |

Reports never infer the method from a run ID. Synthetic fixtures must be
explicitly labeled; their values are not benchmark evidence. Grouped tables
keep method, backend, operator level, and fixture status separate.

## Metric definitions

- Compile rate counts `compiled=true` among all records.
- Correctness requires `PASS`, `compiled=true`, `correct=true`, and no cheat flag.
- Fast_1.0x uses the all-record denominator and counts only correct candidates
  with positive candidate/eager timing and recorded speedup greater than one.
- Fast_compile uses all GPU records with a positive `compile_ms` baseline as
  its denominator, including failed candidates. If no such baseline exists,
  the metric is unavailable.
- Geometric mean speedup uses only correct, positively timed candidates. It is
  conditional on success; it must not be presented as an all-operator speedup.
- Zero latency or throughput means unavailable, not a measured zero.
- `numpy_dev` timing is CPU development data. It never supplies a
  `torch.compile` comparison and is excluded from the GPU throughput plot.

The optional 95% interval uses a deterministic 2,000-resample percentile
bootstrap over whole seeds. Each seed contributes its correct-only geometric
mean with equal weight. The interval field reports that equal-seed estimate,
which can differ from the pooled-record geometric mean when seed sizes differ.
At least two seeds and complete seed metadata are required. Mixed methods,
backends, and fixture status do not get a combined interval. A small seed count,
different operator coverage, and conditioning on success limit interpretation;
the interval does not establish a paired training ablation.

The throughput plot uses the supplied `achieved_tflops` and
`arithmetic_intensity` fields for correct, timed GPU records. Baseline throughput
is derived using the same work and its recorded baseline latency. Operator
FLOP/byte accounting determines the interpretation; a plot cannot verify its
cost model. A hardware roof is drawn only with explicit parameters:

```bash
python -m eval.benchmark_report --input traces.jsonl --out eval/out \
  --peak-tflops <measured-or-specified-peak> --bandwidth-gbps <bandwidth>
```

Both parameters must be positive. No GPU peak or bandwidth is guessed.

## Paired held-out model comparison

```bash
python -m eval.compare_heldout --run-dir experiments/laptop_fullparam --out eval/out/heldout
```

Repeat `--run-dir` for independent runs. Each directory supplies `config.json`,
`heldout_environment.json` and raw `heldout.jsonl`. The reporter checks config,
model revision, precision, hardware, per-operator seeds and complete final
coverage. Selection, fixture and oracle records are excluded. It writes Markdown,
strict JSON and a PNG, preserving failures in the denominator and separating
models, GPU types and precisions. Three held-out workloads do not support a broad
generalization claim. Best-of-K and pass@1 have different search budgets.

## Nsight Compute CSV

```bash
python -m eval.parse_ncu --input profile.csv --out eval/out/nsight.json
```

The parser supports long-form exports containing `Kernel Name`, `Metric Name`,
`Metric Unit`, and `Metric Value`, and raw exports containing metric columns
plus an optional units row. Launches remain separated by process, launch ID,
context, stream, and kernel name. No evaluation `run_id` is inferred: join with
execution records only when the profiling run has a known association.

Known DRAM percentage, DRAM bytes/second, duration, and Tensor pipe metrics
receive normalized fields. All raw metric names and sections are retained.
Unavailable values remain null. Tensor active-cycle and elapsed-cycle
percentages have different denominators and remain separate. Unrecognized
Tensor pipelines stay in the named metric maps rather than being relabeled as
an aggregate utilization figure. Time converts to milliseconds; bandwidth
uses decimal SI prefixes and converts to GB/s.

Unit and CSV behavior follows the official [Nsight Compute metrics and units
guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#metrics-and-units)
and [Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
