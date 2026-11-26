"""Compare live baseline measurements within one GPU and timing procedure."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from eval.benchmark_report import load_traces


def summarize(rows: list[dict]) -> list[dict]:
    if len({(r.get("gpu_model"), r.get("timing_mode")) for r in rows}) > 1:
        raise ValueError("Use one GPU and timing procedure per baseline comparison")
    groups = defaultdict(list)
    for row in rows:
        if row.get("fixture") or row.get("backend") != "triton" or row.get("method") != "triton_baseline":
            raise ValueError("Only real Triton baseline records belong in this comparison")
        if row["status"] != "PASS" or not row["correct"] or row["cheat_detected"] or row["candidate_ms"] <= 0:
            raise ValueError("Resolve failed or untimed baselines before summarizing")
        groups[(row["op_name"], row.get("dtype", "unspecified"))].append(row)
    results = []
    for (op, dtype), group in sorted(groups.items()):
        seeds = [r.get("seed") for r in group]
        if None in seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Each workload/dtype needs distinct explicit input seeds")
        record = dict(op_name=op, dtype=dtype, seeds=len(seeds), candidate_ms=statistics.median(r["candidate_ms"] for r in group))
        for baseline in ("eager", "compile"):
            values = [r[f"speedup_vs_{baseline}"] for r in group]
            available = all(r[f"{'eager' if baseline == 'eager' else 'compile'}_ms"] > 0 and x > 0 for r, x in zip(group, values))
            record[f"speedup_{baseline}"] = math.exp(statistics.fmean(math.log(x) for x in values)) if available else None
            record[f"speedup_{baseline}_min"] = min(values) if available else None
            record[f"speedup_{baseline}_max"] = max(values) if available else None
        results.append(record)
    return results


def write_comparison(path: Path, out: Path) -> None:
    rows = load_traces(path)
    results = summarize(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    if not results:
        return
    with (out / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ops = sorted({r["op_name"] for r in results})
    dtypes = sorted({r["dtype"] for r in results})
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(ops) * .42)), sharey=True, dpi=180)
    colors = ["#087f8c", "#2255a4", "#924d19"]
    width = .7 / max(len(dtypes), 1)
    for ax, baseline in zip(axes, ("eager", "compile")):
        for index, dtype in enumerate(dtypes):
            subset = {r["op_name"]: r for r in results if r["dtype"] == dtype}
            valid = [(i, subset[op]) for i, op in enumerate(ops) if op in subset and subset[op][f"speedup_{baseline}"] is not None]
            positions = [i + (index - (len(dtypes) - 1) / 2) * width for i, _ in valid]
            values = [r[f"speedup_{baseline}"] for _, r in valid]
            errors = [[r[f"speedup_{baseline}"] - r[f"speedup_{baseline}_min"] for _, r in valid],
                      [r[f"speedup_{baseline}_max"] - r[f"speedup_{baseline}"] for _, r in valid]]
            ax.barh(positions, values, height=width * .9, label=dtype, color=colors[index % len(colors)], xerr=errors, capsize=2)
        ax.axvline(1, color="#64748b", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"Geometric mean speedup vs {'eager PyTorch' if baseline == 'eager' else 'torch.compile'} (×)")
        ax.grid(axis="x", alpha=.15)
        ax.legend(frameon=False)
    axes[0].set_yticks(range(len(ops)), ops, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(f"Triton baseline comparison · {rows[0].get('gpu_model', 'recorded GPU')}", fontsize=13)
    fig.text(.5, .01, "Whiskers show the range across input seeds; timings use the recorded procedure. These are handwritten baselines, not learned-policy gains.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(out / "baseline_comparison.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_comparison(args.input, args.out)


if __name__ == "__main__":
    main()
