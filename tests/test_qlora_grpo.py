import contextlib
import json
import tempfile
import unittest
from pathlib import Path

from rl_trainer.checkpointing import CheckpointState, load_checkpoint, save_checkpoint, verify_compatible_run_paths
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import (
    SampledCandidate,
    compute_clipped_grpo_loss_numpy,
    run_gpu_training_loop,
)
from rl_trainer.qlora_grpo import compute_shared_base_ref_logps, select_trainable_adapter_params


class _FakeParam:
    def __init__(self, requires_grad: bool, name: str):
        self.requires_grad = requires_grad
        self.name = name


class _FakePeftModel:
    def __init__(self):
        self.adapter_enabled = True
        self._params = [
            _FakeParam(False, "base.weight_nf4"),
            _FakeParam(True, "lora_A.weight"),
            _FakeParam(True, "lora_B.weight"),
        ]
        self.saved_to: str | None = None
        self.loaded_from: str | None = None

    def parameters(self):
        return list(self._params)

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = prev

    def save_pretrained(self, out_dir: str):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}), encoding="utf-8")
        self.saved_to = out_dir

    def load_adapter(self, adapter_dir: str, adapter_name: str = "default"):
        self.loaded_from = adapter_dir


class TestQloraAndOnPolicyInvariants(unittest.TestCase):
    def test_t4_7b_qlora_preset_defaults_and_config_hash_sensitivity(self):
        cfg = ExperimentConfig.preset_t4_7b_qlora()
        self.assertEqual(cfg.training_mode, "qlora_7b_nf4")
        self.assertEqual(cfg.model_name_or_path, "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(cfg.dtype, "fp16")
        self.assertEqual(cfg.group_size, 2)
        self.assertEqual(cfg.heldout_best_of_k, 8)
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.top_p, 1.0)
        self.assertEqual(cfg.top_k, 0)
        self.assertEqual(cfg.lora_dropout, 0.0)

        # Config hash must change when LR, KL, clip_eps, or max_new_tokens changes
        base_hash = cfg.config_hash()
        self.assertNotEqual(ExperimentConfig.preset_t4_7b_qlora(learning_rate=1e-4).config_hash(), base_hash)
        self.assertNotEqual(ExperimentConfig.preset_t4_7b_qlora(kl_coef=0.1).config_hash(), base_hash)
        self.assertNotEqual(ExperimentConfig.preset_t4_7b_qlora(clip_eps=0.1).config_hash(), base_hash)
        self.assertNotEqual(ExperimentConfig.preset_t4_7b_qlora(max_new_tokens=256).config_hash(), base_hash)

    def test_strict_on_policy_rejects_temperature_or_top_p_warping(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(temperature=0.8).validate(strict_on_policy_sampling=True)
        with self.assertRaises(ValueError):
            ExperimentConfig(top_p=0.9).validate(strict_on_policy_sampling=True)
        with self.assertRaises(ValueError):
            ExperimentConfig(top_k=50).validate(strict_on_policy_sampling=True)

    def test_clipped_grpo_objective_clips_ratio_and_penalizes_kl(self):
        # At new==old==ref, ratio=1.0, kl=0.0 => loss = -adv
        loss_eq, kl_eq = compute_clipped_grpo_loss_numpy([-1.0, -2.0], [-1.0, -2.0], [-1.0, -2.0], advantage=1.5, clip_eps=0.2, kl_coef=0.05)
        self.assertAlmostEqual(kl_eq, 0.0, places=8)
        self.assertAlmostEqual(loss_eq, -1.5, places=8)

        # Large positive shift with positive advantage is clipped at (1 + clip_eps) * adv = 1.2 * 1.0
        loss_hi, kl_hi = compute_clipped_grpo_loss_numpy([0.0], [-5.0], [-5.0], advantage=1.0, clip_eps=0.2, kl_coef=0.0)
        self.assertAlmostEqual(loss_hi, -1.2, places=6)
        self.assertGreater(kl_hi, 0.0)

    def test_qlora_selects_trainable_adapters_and_uses_disable_adapter_for_ref(self):
        fake = _FakePeftModel()
        trainable = select_trainable_adapter_params(fake)
        self.assertEqual([p.name for p in trainable], ["lora_A.weight", "lora_B.weight"])

        observed_states = []

        def _fake_logp(model_obj, p_ids, c_ids):
            observed_states.append(model_obj.adapter_enabled)
            return [-0.5] * len(c_ids)

        self.assertTrue(fake.adapter_enabled)
        ref_lp = compute_shared_base_ref_logps(fake, _fake_logp, [1, 2], [3, 4])
        self.assertEqual(observed_states, [False])
        self.assertTrue(fake.adapter_enabled)
        self.assertEqual(ref_lp, [-0.5, -0.5])

    def test_adapter_checkpoint_save_load_and_incompatible_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td) / "qlora_ckpts"
            out_jsonl = Path(td) / "out.jsonl"
            fake_model = _FakePeftModel()
            state = CheckpointState(
                step=2,
                op_cursor=2,
                seed=42,
                config_hash="hash_aaa",
                best_mean_reward=0.8,
                total_rollouts=4,
                training_mode="qlora_7b_nf4",
            )
            save_checkpoint(ckpt_dir, state, model=fake_model, adapter_only=True, save_rng=True)
            self.assertIsNotNone(fake_model.saved_to)

            loaded = load_checkpoint(ckpt_dir, expected_config_hash="hash_aaa", model=fake_model)
            self.assertEqual(loaded.adapter_dir, "adapter")
            self.assertIsNotNone(fake_model.loaded_from)

            # Refuse overwriting checkpoint_dir with a different config_hash
            with self.assertRaises(ValueError):
                verify_compatible_run_paths(ckpt_dir, out_jsonl, "hash_bbb", is_resuming=False, overwrite=False)

            # Refuse appending to JSONL with a different config_hash
            out_jsonl.write_text(json.dumps({"config_hash": "hash_aaa"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_compatible_run_paths(ckpt_dir, out_jsonl, "hash_bbb", is_resuming=True, overwrite=False)

    def test_policy_updated_false_logged_when_all_advantages_zero(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(
                training_mode="full_param_0.49b",
                backend="numpy_dev",
                dtype="fp32",
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                group_size=2,
                max_steps=1,
                checkpoint_every=1,
                checkpoint_dir=str(Path(td) / "ckpts"),
                output_jsonl=str(Path(td) / "rollouts.jsonl"),
            )
            same_src = "import numpy as np\ndef kernel_forward(x, w, eps=1e-6): return x * 0.0\n"

            def _sampler(prompt, n, s):
                return [
                    SampledCandidate([1, 2], [3, 4, 2], same_src, same_src, [-1.0, -1.0, -0.1], [-1.0, -1.0, -0.1])
                    for _ in range(n)
                ]

            update_called = {"count": 0}

            def _upd(samples, advs):
                update_called["count"] += 1
                return {"policy_updated": True}

            run_gpu_training_loop(cfg, sample_fn=_sampler, update_fn=_upd, is_fixture=True)
            self.assertEqual(update_called["count"], 0)
            rows = [json.loads(ln) for ln in Path(cfg.output_jsonl).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertFalse(row["policy_updated"])
                self.assertEqual(row["split"], "train")
                self.assertIn("candidate_sha256", row)
                self.assertIn("sampling_config", row)
                self.assertEqual(row["config_hash"], cfg.config_hash())


if __name__ == "__main__":
    unittest.main()
