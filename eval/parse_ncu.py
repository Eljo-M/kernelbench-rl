"""Parse Nsight Compute long-form and raw CSV exports into per-launch JSON.

Numeric units use SI prefixes, matching Nsight Compute's display convention.
Uncollected metrics remain null. Tensor active-cycle and elapsed-cycle metrics
are kept separate because their denominators differ. No run_id is inferred.
See https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html and
https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#metrics-and-units.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path


def _number(value: str) -> float | None:
    value = value.strip().replace(",", "")
    if value.lower() in ("", "n/a", "na", "nan", "inf", "-inf", "not supported"):
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid Nsight metric value {value!r}") from exc
    return result if math.isfinite(result) else None


def convert_metric(value: str, unit: str) -> tuple[float | None, str]:
    """Convert elapsed time to ms and bandwidth to GB/s; retain other units."""
    number = _number(value)
    unit = unit.strip()
    time_factors = {"nsecond": 1e-6, "ns": 1e-6, "usecond": 1e-3, "us": 1e-3, "µs": 1e-3, "μs": 1e-3, "msecond": 1, "ms": 1, "second": 1000, "s": 1000}
    bandwidth_factors = {"byte/second": 1e-9, "byte/s": 1e-9, "B/s": 1e-9, "Kbyte/second": 1e-6, "Kbyte/s": 1e-6, "KB/s": 1e-6, "Mbyte/second": 1e-3, "Mbyte/s": 1e-3, "MB/s": 1e-3, "Gbyte/second": 1, "Gbyte/s": 1, "GB/s": 1, "Tbyte/second": 1000, "Tbyte/s": 1000, "TB/s": 1000}
    if unit in time_factors:
        return (number * time_factors[unit] if number is not None else None), "ms"
    if unit in bandwidth_factors:
        return (number * bandwidth_factors[unit] if number is not None else None), "GB/s"
    if unit in ("%", "percent"):
        return number, "%"
    return number, unit


def parse_ncu_csv(source: str | Path) -> list[dict]:
    """Accept a CSV string or Path; return launch records with raw metric names."""
    text = source.read_text(encoding="utf-8-sig") if isinstance(source, Path) else source.lstrip("\ufeff")
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("==")]
    reader = list(csv.reader(io.StringIO("\n".join(lines))))
    records: dict[tuple, dict] = {}
    headers: list[str] | None = None
    raw_units: dict[str, str] | None = None
    saw_header = False

    def launch(row: dict) -> dict:
        identity = tuple(row.get(name, "") for name in ("Process ID", "ID", "Context", "Stream", "Kernel Name"))
        if identity not in records:
            records[identity] = {"process_id": identity[0], "launch_id": identity[1], "context": identity[2], "stream": identity[3], "kernel_name": identity[4], "dram_bandwidth_pct": None, "dram_bandwidth_gbps": None, "tensor_elapsed_pct": None, "tensor_active_pct": None, "duration_ms": None, "metrics": {}}
        return records[identity]

    def add_metric(record: dict, name: str, value: str, unit: str, section: str = "") -> None:
        normalized, normalized_unit = convert_metric(value, unit)
        entry = {"value": normalized, "unit": normalized_unit, "source_unit": unit, "section": section}
        # Full names are retained so a downstream join can inspect denominator
        # semantics and detect duplicate metric names from different sections.
        key = f"{section}::{name}" if section else name
        record["metrics"][key] = entry
        if name == "dram__throughput.avg.pct_of_peak_sustained_elapsed" and normalized_unit == "%":
            record["dram_bandwidth_pct"] = normalized
        elif name == "dram__bytes.sum.per_second" and normalized_unit == "GB/s":
            record["dram_bandwidth_gbps"] = normalized
        elif name == "gpu__time_duration.sum" and normalized_unit == "ms":
            record["duration_ms"] = normalized
        elif "__pipe_tensor" in name and name.endswith(".pct_of_peak_sustained_elapsed") and normalized_unit == "%":
            # Multiple Tensor pipelines are different counters; avoid silently
            # overwriting one aggregate with a different pipeline's value.
            record.setdefault("tensor_elapsed_metrics", {})[name] = normalized
            if name == "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed":
                record["tensor_elapsed_pct"] = normalized
        elif "__pipe_tensor" in name and name.endswith(".pct_of_peak_sustained_active") and normalized_unit == "%":
            record.setdefault("tensor_active_metrics", {})[name] = normalized
            if name == "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active":
                record["tensor_active_pct"] = normalized

    for fields in reader:
        if "Kernel Name" in fields and ("Metric Name" in fields or any("__" in field for field in fields)):
            headers = [field.strip() for field in fields]
            raw_units = None
            saw_header = True
            continue
        if headers is None:
            continue
        if len(fields) != len(headers):
            raise ValueError(f"Nsight CSV row has {len(fields)} fields; expected {len(headers)}")
        row = dict(zip(headers, fields))
        if "Metric Name" in row:
            if not row.get("Metric Name"):
                continue  # Rule descriptions are not metrics.
            record = launch(row)
            add_metric(record, row["Metric Name"], row.get("Metric Value", ""), row.get("Metric Unit", ""), row.get("Section Name", ""))
        else:
            if raw_units is None and not row.get("Kernel Name") and not row.get("ID"):
                raw_units = row
                continue
            record = launch(row)
            for name in headers:
                if "__" in name:
                    add_metric(record, name, row[name], (raw_units or {}).get(name, ""))
    if not saw_header:
        raise ValueError("No supported Nsight CSV header found (requires Kernel Name and metric columns)")
    return list(records.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = parse_ncu_csv(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Parsed {len(records)} profiled launches into {args.out}")


if __name__ == "__main__":
    main()
