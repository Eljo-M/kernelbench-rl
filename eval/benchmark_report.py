"""Report measured evaluation traces; CPU development runs are not GPU evidence.

Speedup summaries are conditional on correct, timed candidates. Confidence
intervals resample whole seeds instead of treating rollouts as independent runs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics

STATUSES = {"PASS", "COMPILE_ERROR", "NUMERICAL_MISMATCH", "TIMEOUT", "OOM", "RUNTIME_ERROR", "CHEAT"}
BACKENDS = {"triton", "cuda", "numpy_dev"}
NUMERIC_FIELDS = ("max_abs_err", "eager_ms", "compile_ms", "candidate_ms", "speedup_vs_eager", "speedup_vs_compile", "reward")


def validate_row(row: dict, label: str = "record") -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{label}: expected a JSON object")
    for name in ("run_id", "op_name"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f"{label}: {name} must be a nonempty string")
    if type(row.get("level")) is not int or row["level"] not in (1, 2, 3):
        raise ValueError(f"{label}: level must be 1, 2, or 3")
    if row.get("status") not in STATUSES or row.get("backend") not in BACKENDS:
        raise ValueError(f"{label}: unknown status or backend")
    for name in ("compiled", "correct", "cheat_detected"):
        if type(row.get(name)) is not bool:
            raise ValueError(f"{label}: {name} must be boolean")
    for name in NUMERIC_FIELDS + ("achieved_tflops", "arithmetic_intensity"):
        if name not in row and name in ("achieved_tflops", "arithmetic_intensity"):
            continue
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label}: {name} must be a finite number")
        if name != "reward" and value < 0:
            raise ValueError(f"{label}: {name} must be nonnegative")
    if "seed" in row and (type(row["seed"]) not in (int, str) or str(row["seed"]) == ""):
        raise ValueError(f"{label}: seed must be an integer or nonempty string")
    if "method" in row and (not isinstance(row["method"], str) or not row["method"]):
        raise ValueError(f"{label}: method must be a nonempty string")
    if "fixture" in row and type(row["fixture"]) is not bool:
        raise ValueError(f"{label}: fixture must be boolean")
    return row


def load_traces(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite {value}")))
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        rows.append(validate_row(row, f"{path}:{lineno}"))
    return rows


def _passed(row: dict) -> bool:
    return row["status"] == "PASS" and row["compiled"] and row["correct"] and not row["cheat_detected"]


def _timed(row: dict) -> bool:
    return _passed(row) and row["eager_ms"] > 0 and row["candidate_ms"] > 0 and row["speedup_vs_eager"] > 0


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(x) for x in values)) if values else 0.0


def _seed_ci(rows: list[dict]) -> dict | None:
    if not rows or any("seed" not in row for row in rows):
        return None
    if len({(row.get("method", "unspecified"), row["backend"], row.get("fixture", False),
             row.get("dtype", "unspecified"), row.get("gpu_model", "unspecified"),
             row.get("timing_mode", "unspecified")) for row in rows}) != 1:
        return None
    grouped = defaultdict(list)
    for row in rows:
        grouped[(type(row["seed"]).__name__, str(row["seed"]))].append(row["speedup_vs_eager"])
    if len(grouped) < 2:
        return None
    means = [_geomean(grouped[key]) for key in sorted(grouped)]
    rng = random.Random(0)
    estimates = sorted(_geomean(rng.choices(means, k=len(means))) for _ in range(2000))
    return {"estimate": round(_geomean(means), 4), "low": round(estimates[49], 4), "high": round(estimates[1949], 4),
            "seeds": len(means), "confidence": 0.95, "method": "seed-cluster percentile bootstrap", "resamples": 2000}


def summarize_traces(rows: list[dict]) -> dict:
    for index, row in enumerate(rows, 1):
        validate_row(row, f"record {index}")
    n = len(rows)
    timed = [row for row in rows if _timed(row)]
    # Old NumPy development records may contain a placeholder compile_ms.
    comparable = [row for row in rows if row["backend"] != "numpy_dev" and row["compile_ms"] > 0]
    pct = lambda count, total: round(100 * count / total, 1) if total else 0.0
    failures = Counter()
    for row in rows:
        if not _passed(row):
            status = row["status"] if row["status"] != "PASS" else "INCONSISTENT_PASS"
            if status == "COMPILE_ERROR" and "syntaxerror" in str(row.get("error_message", "")).lower():
                status = "SYNTAX_ERROR"
            failures[status] += 1
    return {
        "n": n,
        "compile_pct": pct(sum(row["compiled"] for row in rows), n),
        "correct_pct": pct(sum(_passed(row) for row in rows), n),
        "fast_1x_pct": pct(sum(row["speedup_vs_eager"] > 1 for row in timed), n),
        "fast_compile_pct": pct(sum(_timed(row) and row["speedup_vs_compile"] > 1 for row in comparable), len(comparable)) if comparable else None,
        "compile_baseline_n": len(comparable),
        "timed_correct_n": len(timed),
        "geomean_speedup": round(_geomean([row["speedup_vs_eager"] for row in timed]), 4),
        "speedup_ci95": _seed_ci(timed),
        "failure_counts": dict(sorted(failures.items())),
        "development_n": sum(row["backend"] == "numpy_dev" for row in rows),
        "fixture_n": sum(bool(row.get("fixture", False)) for row in rows),
    }


def grouped_summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("method", "unspecified"), row["backend"], row["level"], bool(row.get("fixture", False)),
                row.get("dtype", "unspecified"), row.get("gpu_model", "unspecified"), row.get("timing_mode", "unspecified"))].append(row)
    return [{"method": key[0], "backend": key[1], "level": key[2], "fixture": key[3],
             "dtype": key[4], "gpu_model": key[5], "timing_mode": key[6], **summarize_traces(group)}
            for key, group in sorted(groups.items())]


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _format_ci(ci: dict | None) -> str:
    return f"{ci['estimate']}x [{ci['low']}, {ci['high']}] ({ci['seeds']} seeds)" if ci else "Unavailable (requires ≥2 seeds)"


def _plots(rows: list[dict], out_dir: Path, peak_tflops: float | None, bandwidth_gbps: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passing = [row for row in rows if _timed(row)]
    fig, ax = plt.subplots(figsize=(9, max(3, min(24, len(passing) * .3 + 1.5))), dpi=180)
    if passing:
        shown = passing[:60]
        ax.barh(range(len(shown)), [row["speedup_vs_eager"] for row in shown], color=["#64748b" if row["backend"] == "numpy_dev" else "#2563eb" for row in shown])
        ax.set_yticks(range(len(shown)), [f"{row['op_name']} · {row.get('dtype', 'unspecified')} · seed {row.get('seed', '?')}" for row in shown], fontsize=7)
        ax.axvline(1, color="#475569", linestyle="--", linewidth=1)
        if len(passing) > len(shown):
            ax.text(.99, .02, "First 60 timed correct records; full data in report.html", transform=ax.transAxes, ha="right", fontsize=8)
    else:
        ax.text(.5, .5, "No correct candidates with measured timing", transform=ax.transAxes, ha="center")
    fixture_note = " · synthetic fixture" if any(row.get("fixture") for row in rows) else ""
    ax.set(xlabel="Reported speedup vs eager reference (x)", title=f"KernelBench-RL candidate timings{fixture_note}")
    fig.tight_layout()
    fig.savefig(out_dir / "speedups.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    gpu = [row for row in passing if row["backend"] in ("triton", "cuda") and row.get("achieved_tflops", 0) > 0 and row.get("arithmetic_intensity", 0) > 0]
    for method in sorted({row.get("method", "unspecified") for row in gpu}):
        group = [row for row in gpu if row.get("method", "unspecified") == method]
        ax.scatter([row["arithmetic_intensity"] for row in group], [row["achieved_tflops"] for row in group], label=method, alpha=.8)
    if gpu:
        # Same operator work divided by independently recorded baseline latency.
        ax.scatter([row["arithmetic_intensity"] for row in gpu], [row["achieved_tflops"] * row["candidate_ms"] / row["eager_ms"] for row in gpu], marker="x", label="Eager reference (recorded latency)", alpha=.6)
        compiled = [row for row in gpu if row["compile_ms"] > 0]
        if compiled:
            ax.scatter([row["arithmetic_intensity"] for row in compiled], [row["achieved_tflops"] * row["candidate_ms"] / row["compile_ms"] for row in compiled], marker="+", label="torch.compile (recorded latency)", alpha=.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.text(.5, .5, "No measured GPU throughput records\nCPU development traces are excluded", transform=ax.transAxes, ha="center")
    if peak_tflops and bandwidth_gbps:
        xs = [10 ** (-3 + i * .05) for i in range(181)]
        ax.plot(xs, [min(peak_tflops, x * bandwidth_gbps / 1000) for x in xs], linestyle="--", color="#475569", label="User-supplied hardware roof")
        ax.set_xscale("log")
        ax.set_yscale("log")
    if gpu or (peak_tflops and bandwidth_gbps):
        ax.legend(fontsize=8)
    ax.set(xlabel="Arithmetic intensity (FLOP / byte)", ylabel="Achieved TFLOP/s", title=f"GPU throughput from supplied traces{fixture_note}")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir / "roofline.png")
    plt.close(fig)


def write_report(input_jsonl: Path, out_dir: Path, *, peak_tflops: float | None = None, bandwidth_gbps: float | None = None) -> None:
    if (peak_tflops is None) != (bandwidth_gbps is None):
        raise ValueError("Supply both peak_tflops and bandwidth_gbps, or neither")
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in (peak_tflops, bandwidth_gbps)):
        raise ValueError("Hardware roof parameters must be finite and positive")
    rows = load_traces(Path(input_jsonl))
    stats = summarize_traces(rows)
    groups = grouped_summaries(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compile_value = f"{stats['fast_compile_pct']}% of {stats['compile_baseline_n']} comparable records" if stats['fast_compile_pct'] is not None else "Unavailable"
    lines = ["# KernelBench-RL Evaluation Summary", "", "This report describes the supplied records; it does not establish a controlled training ablation.", ""]
    if stats["fixture_n"]:
        lines.extend([f"**Synthetic fixtures: {stats['fixture_n']} records. These values are test data, not benchmark evidence.**", ""])
    if stats["development_n"]:
        lines.extend([f"CPU development records: {stats['development_n']}. NumPy timing is not CUDA performance or a torch.compile baseline.", ""])
    lines.extend(["| Metric | Value |", "| :--- | ---: |", f"| Total rollouts | {stats['n']} |", f"| Compile rate | {stats['compile_pct']}% |", f"| Correctness rate | {stats['correct_pct']}% |", f"| Fast_1.0x (vs eager; all-rollout denominator) | {stats['fast_1x_pct']}% |", f"| Fast_compile | {compile_value} |", f"| Timed correct candidates | {stats['timed_correct_n']} |", f"| Geomean speedup (timed correct only) | {stats['geomean_speedup']}x |", f"| Geomean speedup 95% interval | {_format_ci(stats['speedup_ci95'])} |", "", "Speedup excludes failed, cheating, or untimed records. Zero baseline latency means unavailable. The interval resamples whole seeds with equal weight; it is conditional on timed correct records and is not a paired ablation test.", "", "## Method, backend, precision, and level", "", "| Method | Backend | Dtype | GPU | Timing | Level | Fixture | N | Compile % | Correct % | Fast_1.0x % | Correct-only geomean | 95% interval |", "| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |"])
    for group in groups:
        lines.append(f"| {_md(group['method'])} | {_md(group['backend'])} | {_md(group['dtype'])} | {_md(group['gpu_model'])} | {_md(group['timing_mode'])} | {group['level']} | {group['fixture']} | {group['n']} | {group['compile_pct']} | {group['correct_pct']} | {group['fast_1x_pct']} | {group['geomean_speedup']}x | {_format_ci(group['speedup_ci95'])} |")
    lines.extend(["", "## Failure counts", "", "| Failure | Count |", "| :--- | ---: |"])
    lines.extend(f"| {_md(status)} | {count} |" for status, count in stats["failure_counts"].items())
    if not stats["failure_counts"]:
        lines.append("| None | 0 |")
    lines.extend(["", "GPU throughput plots use only correct GPU records containing positive achieved_tflops and arithmetic_intensity. Hardware ceilings are drawn only when explicitly supplied. FLOP/byte estimates depend on the operator cost model.", ""])
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"overall": stats, "groups": groups, "hardware_roof": {"peak_tflops": peak_tflops, "bandwidth_gbps": bandwidth_gbps}}, indent=2, allow_nan=False), encoding="utf-8")
    _plots(rows, out_dir, peak_tflops, bandwidth_gbps)
    template = Path(__file__).with_name("report.html").read_text(encoding="utf-8")
    safe_json = json.dumps(rows, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (out_dir / "report.html").write_text(template.replace("<!-- TRACE_DATA -->", safe_json), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--bandwidth-gbps", type=float)
    args = parser.parse_args()
    write_report(args.input, args.out, peak_tflops=args.peak_tflops, bandwidth_gbps=args.bandwidth_gbps)
    print(f"Wrote evaluation report to {args.out}")


if __name__ == "__main__":
    main()
