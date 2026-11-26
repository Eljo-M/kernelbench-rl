import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_trainer.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from rl_trainer.evaluate_heldout import evaluate_heldout_methods
from rl_trainer.experiment_config import DEFAULT_HELDOUT_OPS, DEFAULT_TRAIN_OPS, ExperimentConfig
from rl_trainer.gpu_train_grpo import SampledCandidate, extract_python_source, run_gpu_training_loop

VALID_RMSNORM_NP = (
    "```python\n"
    "import numpy as np\n"
    "def kernel_forward(x, w, eps=1e-6):\n"
    "    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)) * w\n"
    "```"
)
BAD_RMSNORM_NP = "import numpy as np\ndef kernel_forward(x, w, eps=1e-6): return x * 0.0\n"


class TestRlGpuBridgeAndCheckpointing(unittest.TestCase):
    def test_experiment_config_defaults_and_leakage_guard(self):
        cfg = ExperimentConfig()
        cfg.validate()
        self.assertEqual(cfg.model_name_or_path, "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.assertEqual(cfg.dtype, "bf16")
        self.assertEqual(cfg.group_size, 2)
        self.assertEqual(cfg.heldout_best_of_k, 8)
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.top_p, 1.0)
        self.assertEqual(cfg.top_k, 0)
        self.assertEqual(set(DEFAULT_TRAIN_OPS) & set(DEFAULT_HELDOUT_OPS), set())

        leaky = ExperimentConfig(train_ops=["rmsnorm_2048"], heldout_ops=["rmsnorm_2048"])
        with self.assertRaises(ValueError):
            leaky.validate()

    def test_checkpoint_save_and_resume_restores_rng_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td) / "ckpts"
            random.seed(123)
            np.random.seed(123)
            _ = random.random()
            _ = np.random.rand()

            state = CheckpointState(
                step=4,
                op_cursor=4,
                seed=42,
                config_hash="abc123def4560000",
                best_mean_reward=1.25,
                total_rollouts=8,
                is_fixture=True,
            )
            save_checkpoint(ckpt_dir, state, save_rng=True)
            expected_py = random.random()
            expected_np = float(np.random.rand())

            random.seed(999)
            np.random.seed(999)
            loaded = load_checkpoint(ckpt_dir, expected_config_hash="abc123def4560000", restore_rng=True)
            self.assertEqual(loaded.step, 4)
            self.assertAlmostEqual(random.random(), expected_py, places=12)
            self.assertAlmostEqual(float(np.random.rand()), expected_np, places=12)

            with self.assertRaises(ValueError):
                load_checkpoint(ckpt_dir, expected_config_hash="wrong_hash")

    def test_gpu_training_loop_offload_hooks_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td) / "ckpts"
            out_jsonl = Path(td) / "rollouts.jsonl"
            cfg = ExperimentConfig(
                backend="numpy_dev",
                dtype="fp32",
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                group_size=2,
                max_steps=2,
                checkpoint_every=1,
                checkpoint_dir=str(ckpt_dir),
                output_jsonl=str(out_jsonl),
            )
            hook_events: list[str] = []

            def _sample(prompt: str, n: int, gen_seed: int) -> list[SampledCandidate]:
                out = []
                for i in range(n):
                    raw = VALID_RMSNORM_NP if i % 2 == 0 else BAD_RMSNORM_NP
                    ext = extract_python_source(raw)
                    out.append(SampledCandidate([1, 2], [3, 4, 2], raw, ext, [-0.5, -0.5, -0.1], [-0.5, -0.5, -0.1]))
                return out

            st1 = run_gpu_training_loop(
                cfg,
                sample_fn=_sample,
                update_fn=lambda samples, advs: {"policy_updated": True, "policy_loss": -0.2, "mean_kl": 0.01},
                before_grade_hook=lambda: hook_events.append("before"),
                after_grade_hook=lambda: hook_events.append("after"),
                is_fixture=True,
            )
            self.assertEqual(st1.step, 2)
            self.assertEqual(st1.total_rollouts, 4)
            self.assertEqual(hook_events.count("before"), 2)
            self.assertEqual(hook_events.count("after"), 2)

            cfg_resume = ExperimentConfig(
                backend="numpy_dev",
                dtype="fp32",
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                group_size=2,
                max_steps=3,
                checkpoint_every=1,
                checkpoint_dir=str(ckpt_dir),
                resume_from=str(ckpt_dir),
                output_jsonl=str(out_jsonl),
            )
            st2 = run_gpu_training_loop(cfg_resume, sample_fn=_sample, is_fixture=True)
            self.assertEqual(st2.step, 3)
            self.assertEqual(st2.total_rollouts, 6)

    def test_heldout_evaluator_logs_policy_updated_false_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            out_jsonl = Path(td) / "heldout.jsonl"
            cfg = ExperimentConfig(
                backend="numpy_dev",
                dtype="fp32",
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                group_size=2,
                heldout_best_of_k=8,
            )

            def _base_gen(prompt: str, n: int, gen_seed: int) -> list[str]:
                return [BAD_RMSNORM_NP] * (n - 1) + [VALID_RMSNORM_NP] if n > 1 else [BAD_RMSNORM_NP]

            summaries = evaluate_heldout_methods(
                cfg,
                base_generate_fn=_base_gen,
                trained_generate_fn=lambda p, n, s: [VALID_RMSNORM_NP] * n,
                best_of_k=8,
                out_jsonl=out_jsonl,
                is_fixture=True,
            )
            self.assertEqual(summaries["base_pass1"]["pass_rate"], 0.0)
            self.assertEqual(summaries["base_best_of_8"]["pass_rate"], 100.0)
            self.assertEqual(summaries["trained_checkpoint"]["pass_rate"], 100.0)

            rows = [json.loads(ln) for ln in out_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 11)
            for row in rows:
                self.assertFalse(row["policy_updated"])
                self.assertEqual(row["split"], "heldout")
                self.assertIn("candidate_sha256", row)
                self.assertIn("sampling_config", row)


if __name__ == "__main__":
    unittest.main()
