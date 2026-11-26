import csv
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from eval.benchmark_report import grouped_summaries, load_traces, summarize_traces, write_report
from eval.parse_ncu import convert_metric, parse_ncu_csv


def trace(**changes):
    result = {"run_id": "test", "op_name": "rmsnorm_2048", "level": 1, "backend": "triton", "status": "PASS", "compiled": True, "correct": True, "cheat_detected": False, "max_abs_err": 0.0, "eager_ms": 2.0, "compile_ms": 1.0, "candidate_ms": 1.0, "speedup_vs_eager": 2.0, "speedup_vs_compile": 1.0, "reward": 1.0}
    result.update(changes)
    return result


class TestEvaluation(unittest.TestCase):
    def test_report_excludes_validation_candidates_used_for_selection(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "heldout.jsonl"
            rows = [trace(eval_phase="selection_val", speedup_vs_eager=100),
                    trace(eval_phase="heldout_test", speedup_vs_eager=2)]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            write_report(path, Path(td) / "report")
            stats = json.loads((Path(td)/"report/summary.json").read_text())
            self.assertEqual(stats["selection_records_excluded"], 1)
            self.assertEqual(stats["overall"]["n"], 1)
            self.assertEqual(stats["overall"]["geomean_speedup"], 2)

    def test_failed_cheating_and_untimed_records_do_not_inflate_speedup(self):
        stats = summarize_traces([trace(), trace(status="NUMERICAL_MISMATCH", correct=False, speedup_vs_eager=1000), trace(status="CHEAT", cheat_detected=True, speedup_vs_eager=1000), trace(candidate_ms=0, speedup_vs_eager=1000)])
        self.assertEqual(stats["timed_correct_n"], 1)
        self.assertEqual(stats["geomean_speedup"], 2.0)
        self.assertEqual(stats["fast_1x_pct"], 25.0)
        self.assertEqual(stats["failure_counts"], {"CHEAT": 1, "NUMERICAL_MISMATCH": 1})

    def test_numpy_never_has_torch_compile_baseline(self):
        stats = summarize_traces([trace(backend="numpy_dev", compile_ms=100, speedup_vs_compile=100)])
        self.assertIsNone(stats["fast_compile_pct"])
        self.assertEqual(stats["compile_baseline_n"], 0)
        self.assertEqual(stats["development_n"], 1)

    def test_failed_candidates_stay_in_compile_comparison_denominator(self):
        stats = summarize_traces([trace(speedup_vs_compile=2), trace(status="NUMERICAL_MISMATCH", correct=False, candidate_ms=0, speedup_vs_compile=0)])
        self.assertEqual(stats["compile_baseline_n"], 2)
        self.assertEqual(stats["fast_compile_pct"], 50)

    def test_empty_input_has_complete_zero_summary(self):
        stats = summarize_traces([])
        self.assertEqual(stats["n"], 0)
        self.assertEqual(stats["correct_pct"], 0)
        self.assertEqual(stats["geomean_speedup"], 0)
        self.assertIsNone(stats["speedup_ci95"])

    def test_seed_intervals_resample_seeds_and_require_complete_metadata(self):
        rows = [trace(seed=1, speedup_vs_eager=1), trace(seed=2, speedup_vs_eager=4)]
        stats = summarize_traces(rows)
        self.assertEqual(stats["speedup_ci95"]["seeds"], 2)
        self.assertEqual(stats["speedup_ci95"]["low"], 1)
        self.assertEqual(stats["speedup_ci95"]["high"], 4)
        self.assertEqual(stats, summarize_traces(rows))
        self.assertIsNone(summarize_traces(rows + [trace()])["speedup_ci95"])

    def test_method_level_backend_and_fixture_groups_stay_separate(self):
        groups = grouped_summaries([trace(method="base"), trace(method="trained"), trace(method="base", backend="numpy_dev"), trace(method="base", level=2), trace(method="base", fixture=True)])
        self.assertEqual(len(groups), 5)

    def test_precision_device_and_timing_modes_are_not_pooled(self):
        rows = [trace(seed=1, dtype="fp32", gpu_model="gpuA", timing_mode="events"),
                trace(seed=2, dtype="bf16", gpu_model="gpuA", timing_mode="events"),
                trace(seed=3, dtype="fp32", gpu_model="gpuB", timing_mode="events"),
                trace(seed=4, dtype="fp32", gpu_model="gpuA", timing_mode="cuda_graph")]
        self.assertEqual(len(grouped_summaries(rows)), 4)
        self.assertIsNone(summarize_traces(rows)["speedup_ci95"])

    def test_malformed_jsonl_reports_filename_and_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.jsonl"
            path.write_text(json.dumps(trace()) + '\n{"run_id": "bad"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"bad.jsonl:2:.*op_name"):
                load_traces(path)
            path.write_text(json.dumps(trace()).replace('"reward": 1.0', '"reward": NaN'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_traces(path)
        for bad in [trace(level=True), trace(correct="true"), trace(eager_ms=-1), trace(reward=float("inf"))]:
            with self.assertRaises(ValueError):
                summarize_traces([bad])

    def test_report_artifacts_and_embedded_html_cannot_execute_trace_source(self):
        hostile = '</script><script>alert("unsafe")</script><img src=x onerror=alert(1)>'
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "traces.jsonl"
            path.write_text(json.dumps(trace(run_id=hostile, source_code=hostile, fixture=True)) + "\n", encoding="utf-8")
            out = Path(td) / "out"
            write_report(path, out)
            for name in ("summary.md", "summary.json", "speedups.png", "roofline.png", "report.html"):
                self.assertTrue((out / name).is_file())
            html = (out / "report.html").read_text(encoding="utf-8")
            self.assertNotIn(hostile, html)
            self.assertIn("\\u003c/script\\u003e", html)
            self.assertIn(".textContent", html)
            self.assertNotIn("innerHTML", html)
            self.assertIn("Synthetic fixtures", (out / "summary.md").read_text(encoding="utf-8"))
            empty = Path(td) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            write_report(empty, out)
            self.assertEqual(json.loads((out / "summary.json").read_text())["overall"]["n"], 0)

    @unittest.skipUnless(shutil.which("node"), "Optional Node.js runtime is not installed")
    def test_viewer_selection_accepts_training_source_alias(self):
        html = (Path(__file__).parents[1] / "eval" / "report.html").read_text(encoding="utf-8")
        handler = re.search(r"^function inspect\(r\).*$", html, flags=re.MULTILINE).group(0)
        script = """
const assert = require('node:assert/strict');
let selected;
const elements = {selection: {}, source: {}, record: {}};
const el = id => elements[id];
""" + handler + """
const source = 'def kernel_forward(x):\\n    return x < 1\\n# </script>';
for (const key of ['source_code', 'candidate_source', 'source']) {
    const row = {run_id: 'run', op_name: 'operator', [key]: source};
    inspect(row);
    assert.equal(elements.source.textContent, source, key);
    assert.equal(selected, row);
}
inspect({run_id: 'run', op_name: 'operator'});
assert.equal(elements.source.textContent, 'No kernel source in this trace.');
"""
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class TestNsightParser(unittest.TestCase):
    def test_si_unit_conversion_and_missing_values(self):
        self.assertEqual(convert_metric("2,500", "usecond"), (2.5, "ms"))
        self.assertEqual(convert_metric("125", "Gbyte/second"), (125, "GB/s"))
        self.assertEqual(convert_metric("125000", "Mbyte/second"), (125, "GB/s"))
        self.assertEqual(convert_metric("n/a", "%"), (None, "%"))

    def test_long_form_groups_launches_and_preserves_tensor_denominator(self):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["ID", "Process ID", "Kernel Name", "Section Name", "Metric Name", "Metric Unit", "Metric Value"])
        for metric, unit, value in [("dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "65.5"), ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "%", "80"), ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "%", "50"), ("gpu__time_duration.sum", "nsecond", "2500000"), ("dram__bytes.sum.per_second", "Gbyte/second", "310")]:
            writer.writerow(["1", "42", "kernel,with,commas", "SpeedOfLight", metric, unit, value])
        writer.writerow(["2", "42", "other", "SpeedOfLight", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "n/a"])
        rows = parse_ncu_csv("==PROF== header\n" + stream.getvalue())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["kernel_name"], "kernel,with,commas")
        self.assertEqual(rows[0]["dram_bandwidth_pct"], 65.5)
        self.assertEqual(rows[0]["dram_bandwidth_gbps"], 310)
        self.assertEqual(rows[0]["tensor_active_pct"], 80)
        self.assertEqual(rows[0]["tensor_elapsed_pct"], 50)
        self.assertEqual(rows[0]["duration_ms"], 2.5)
        self.assertIsNone(rows[1]["dram_bandwidth_pct"])

    def test_raw_csv_unit_row(self):
        text = '"ID","Kernel Name","dram__bytes.sum.per_second","gpu__time_duration.sum"\n"","","byte/second","usecond"\n"0","kernel","1000000000","2000"\n'
        record = parse_ncu_csv(text)[0]
        self.assertEqual(record["dram_bandwidth_gbps"], 1)
        self.assertEqual(record["duration_ms"], 2)

    def test_invalid_csv_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "header"):
            parse_ncu_csv("not a csv export")
        with self.assertRaisesRegex(ValueError, "fields"):
            parse_ncu_csv("ID,Kernel Name,Metric Name,Metric Unit,Metric Value\n0,only-two\n")


if __name__ == "__main__":
    unittest.main()
