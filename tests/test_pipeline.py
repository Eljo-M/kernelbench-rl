import json
import tempfile
import unittest
from pathlib import Path

from eval.benchmark_report import summarize_traces, write_report
from rl_trainer.grpo_reward import compute_grpo_advantages, compute_kernel_reward
from rl_trainer.train_grpo import run_dry_loop
from sandbox_runner.verifier import detect_cheating, verify_candidate_source


class TestKernelBenchPipeline(unittest.TestCase):
    def test_cheat_detection_blocks_functional_and_os(self):
        cheated, _ = detect_cheating("import torch.nn.functional as F\ndef kernel_forward(x): return x")
        self.assertTrue(cheated)
        cheated_os, _ = detect_cheating("import os\ndef kernel_forward(x): return x")
        self.assertTrue(cheated_os)

    def test_valid_rmsnorm_passes_and_scores_positive(self):
        code = (
            "import numpy as np\n"
            "def kernel_forward(x, w, eps=1e-6):\n"
            "    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)\n"
            "    return (x / rms) * w\n"
        )
        res = verify_candidate_source("rmsnorm_2048", code, run_id="unit-pass")
        self.assertEqual(res.status, "PASS")
        self.assertTrue(res.compiled)
        self.assertTrue(res.correct)
        self.assertFalse(res.cheat_detected)
        self.assertLess(res.max_abs_err, 1e-4)
        self.assertGreater(res.reward, 0.5)

    def test_numerical_mismatch_rejected(self):
        code = "import numpy as np\ndef kernel_forward(x, w): return np.zeros_like(x)\n"
        res = verify_candidate_source("rmsnorm_2048", code, run_id="unit-bad")
        self.assertEqual(res.status, "NUMERICAL_MISMATCH")
        self.assertAlmostEqual(res.reward, 0.15)

    def test_grpo_advantage_normalization(self):
        adv = compute_grpo_advantages([-1.0, 0.0, 0.15, 2.2])
        self.assertAlmostEqual(sum(adv), 0.0, places=5)
        self.assertGreater(adv[-1], 0.0)
        self.assertLess(adv[0], 0.0)

    def test_reward_monotonicity(self):
        r_cheat = compute_kernel_reward(False, False, True, 0.0, 0.0)
        r_fail = compute_kernel_reward(False, False, False, 0.0, 0.0)
        r_wrong = compute_kernel_reward(True, False, False, 0.0, 0.0)
        r_slow = compute_kernel_reward(True, True, False, 0.8, 0.7)
        r_fast = compute_kernel_reward(True, True, False, 2.0, 1.3)
        self.assertTrue(r_cheat < r_fail < r_wrong < r_slow < r_fast)

    def test_dry_run_and_report_generation(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "dry.jsonl"
            out_dir = Path(td) / "out"
            run_dry_loop(trace_path)
            rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 4)
            stats = summarize_traces(rows)
            self.assertEqual(stats["n"], 4)
            write_report(trace_path, out_dir)
            self.assertTrue((out_dir / "summary.md").exists())
            self.assertTrue((out_dir / "speedups.png").exists())


if __name__ == "__main__":
    unittest.main()
