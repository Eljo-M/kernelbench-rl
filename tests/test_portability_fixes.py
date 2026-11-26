import json
import tempfile
import unittest
from pathlib import Path

from rl_trainer.evaluate_heldout import evaluate_heldout_methods, generate_cached_completions
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import compute_clipped_grpo_loss_torch
from rl_trainer.checkpointing import (CheckpointState, save_checkpoint, reconcile_resume_log,
                                     verify_compatible_run_paths)

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
    def test_interrupted_generation_resumes_only_missing_samples_and_validates_identity(self):
        cfg = ExperimentConfig(backend="numpy_dev", dtype="fp32",
            train_ops=["rmsnorm_2048"], heldout_ops=["rmsnorm_4096"])
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)/"generations.jsonl"
            first_calls = []
            def interrupted(prompt, n, seed):
                first_calls.append(seed)
                if len(first_calls) == 3:
                    raise RuntimeError("disconnected")
                return [str(seed)]
            kwargs = dict(method_budgets=[("base_best_of_4", 4)], cache_path=cache,
                          model_identity="base", prompt_formatter=lambda p: "formatted:"+p)
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                generate_cached_completions(cfg, generate_fn=interrupted, **kwargs)
            resumed_calls = []
            def resumed(prompt, n, seed):
                resumed_calls.append(seed)
                return [str(seed)]
            result = generate_cached_completions(cfg, generate_fn=resumed, **kwargs)
            self.assertEqual(resumed_calls, [50044, 50045])
            self.assertEqual(result["base_best_of_4"][0][3], ["50042", "50043", "50044", "50045"])
            self.assertEqual(len(cache.read_text().splitlines()), 4)
            with self.assertRaises(ValueError):
                generate_cached_completions(cfg, generate_fn=resumed, **dict(kwargs, model_identity="different"))
            with self.assertRaisesRegex(ValueError, "Prompt/seed"):
                generate_cached_completions(cfg, generate_fn=resumed,
                    **dict(kwargs, prompt_formatter=lambda p: "changed:"+p))

    def test_failed_checkpoint_never_replaces_latest_or_exposes_partial_weights(self):
        class BrokenModel:
            def save_pretrained(self, path):
                Path(path).mkdir()
                raise RuntimeError("interrupted weight write")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            save_checkpoint(root, CheckpointState(1, 1, 42, "hash", 0, 2), save_rng=False)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                save_checkpoint(root, CheckpointState(2, 2, 42, "hash", 0, 4),
                                model=BrokenModel(), adapter_only=True, save_rng=False)
            self.assertEqual(json.loads((root/"latest.json").read_text())["step"], 1)
            self.assertFalse((root/"step_00002").exists())
            with self.assertRaises(FileExistsError):
                save_checkpoint(root, CheckpointState(1, 1, 42, "hash", 0, 2), save_rng=False)

    def test_resume_archives_uncheckpointed_steps_and_checks_every_config_hash(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"rollouts.jsonl"
            original = ''.join(json.dumps({"step": s, "config_hash": "hash"})+'\n' for s in (1, 2))
            path.write_text(original)
            archive = reconcile_resume_log(path, 1)
            self.assertEqual(archive.read_text(), original)
            self.assertEqual([json.loads(s)["step"] for s in path.read_text().splitlines()], [1])
            self.assertIsNone(reconcile_resume_log(path, 1))
            path.write_text(original + json.dumps({"step": 3, "config_hash": "other"})+'\n')
            with self.assertRaises(ValueError):
                verify_compatible_run_paths(Path(td)/"ckpt", path, "hash", is_resuming=True)

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
