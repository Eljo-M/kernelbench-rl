import json
import tempfile
import unittest
from pathlib import Path

from rl_trainer.evaluate_heldout import evaluate_heldout_methods
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import compute_clipped_grpo_loss_torch

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from rl_trainer.cpu_master_adamw import CpuMasterAdamW

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

VALID_RMSNORM_NP = (
    "import numpy as np\n"
    "def kernel_forward(x, w, eps=1e-6):\n"
    "    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)) * w\n"
)
BAD_RMSNORM_NP = "import numpy as np\ndef kernel_forward(x, w, eps=1e-6): return x * 0.0\n"


class TestPortabilityFixes(unittest.TestCase):
    def test_config_hash_changes_with_cpu_master_adamw_flag(self):
        c1 = ExperimentConfig(cpu_master_adamw=True)
        c2 = ExperimentConfig(cpu_master_adamw=False)
        self.assertNotEqual(c1.config_hash(), c2.config_hash())

    def test_best_of_k_uses_validation_seed_and_regrades_independent_test_seed(self):
        with tempfile.TemporaryDirectory() as td:
            out_jsonl = Path(td) / "heldout_val_regrade.jsonl"
            cfg = ExperimentConfig(
                backend="numpy_dev",
                dtype="fp32",
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                group_size=2,
                heldout_best_of_k=4,
            )

            seen_calls: list[tuple[str, int]] = []

            def _base_gen(prompt: str, n: int, gen_seed: int) -> list[str]:
                if n == 1:
                    return [BAD_RMSNORM_NP]
                return [BAD_RMSNORM_NP, VALID_RMSNORM_NP, BAD_RMSNORM_NP, BAD_RMSNORM_NP]

            from sandbox_runner.gpu_verifier import grade_candidate

            def _tracking_grader(op_name, src, **kwargs):
                seen_calls.append((kwargs["run_id"], kwargs["seed"]))
                return grade_candidate(op_name, src, **kwargs)

            summaries = evaluate_heldout_methods(
                cfg,
                base_generate_fn=_base_gen,
                grader_fn=_tracking_grader,
                best_of_k=4,
                selection_mode="validation_seed_regrade",
                out_jsonl=out_jsonl,
                is_fixture=True,
            )
            self.assertEqual(summaries["base_best_of_4"]["selection_method"], "validation_seed_regrade")
            self.assertEqual(summaries["base_best_of_4"]["pass_rate"], 100.0)

            rows = [json.loads(ln) for ln in out_jsonl.read_text(encoding="utf-8").splitlines()]
            val_rows = [r for r in rows if r["eval_phase"] == "selection_val"]
            test_sel_rows = [r for r in rows if r["method"] == "base_best_of_4" and r["eval_phase"] == "heldout_test"]
            self.assertEqual(len(val_rows), 4)
            self.assertEqual(len(test_sel_rows), 1)
            # Validation seed must differ from independent held-out test seed
            self.assertNotEqual(val_rows[0]["seed"], test_sel_rows[0]["seed"])
            self.assertEqual(val_rows[0]["seed"], test_sel_rows[0]["seed"] + cfg.selection_seed_offset)
            self.assertEqual(test_sel_rows[0]["sample_k"], 1)  # Index 1 was VALID_RMSNORM_NP

    def test_oracle_same_seed_mode_is_explicitly_labeled(self):
        with tempfile.TemporaryDirectory() as td:
            out_jsonl = Path(td) / "heldout_oracle.jsonl"
            cfg = ExperimentConfig(
                backend="numpy_dev",
                dtype="fp32",
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                group_size=2,
                heldout_best_of_k=2,
            )
            summaries = evaluate_heldout_methods(
                cfg,
                base_generate_fn=lambda p, n, s: [BAD_RMSNORM_NP, VALID_RMSNORM_NP][:n],
                best_of_k=2,
                selection_mode="oracle_same_seed",
                out_jsonl=out_jsonl,
                is_fixture=True,
            )
            self.assertEqual(summaries["base_best_of_2"]["selection_method"], "oracle_same_seed")
            rows = [json.loads(ln) for ln in out_jsonl.read_text(encoding="utf-8").splitlines()]
            oracle_rows = [r for r in rows if r["method"] == "base_best_of_2"]
            self.assertTrue(all(r["eval_phase"] == "heldout_test_oracle" for r in oracle_rows))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in this sandbox; runs automatically on Torch CPU/GPU hosts.")
    def test_cpu_master_adamw_updates_bf16_at_1e6_and_resumes_exact_state(self):
        torch.manual_seed(0)
        layer = nn.Linear(8, 4, bias=False).to(dtype=torch.bfloat16)
        opt = CpuMasterAdamW(layer.parameters(), lr=1e-6)
        sha_before = opt.master_weights_sha256()

        x = torch.randn(2, 8, dtype=torch.bfloat16)
        loss = layer(x).float().pow(2).mean()
        loss.backward()
        updated, delta = opt.copy_grads_and_step(max_grad_norm=1.0)
        self.assertTrue(updated)
        self.assertGreater(delta, 0.0)
        self.assertNotEqual(opt.master_weights_sha256(), sha_before)

        # Save & restore CpuMasterAdamW state cleanly
        saved = opt.state_dict()
        with torch.no_grad():
            layer.weight.zero_()
        opt.load_state_dict(saved)
        self.assertTrue(torch.allclose(layer.weight.float(), opt.master_params[0].to(torch.bfloat16).float()))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in this sandbox; runs automatically on Torch CPU/GPU hosts.")
    def test_objective_delegation_does_not_swallow_exceptions(self):
        def _broken_objective(new_lp, old_lp, ref_lp, adv, mask=None, clip=0.2, beta=0.04):
            raise RuntimeError("intentional objective failure")

        t = torch.zeros((4,), dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            compute_clipped_grpo_loss_torch(t, t, t, 1.0, 0.2, 0.04, objective_fn=_broken_objective)


if __name__ == "__main__":
    unittest.main()
