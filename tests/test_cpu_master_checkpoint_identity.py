"""CPU-only regression tests distinguish rounded policy weights from FP32 masters."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_trainer.checkpointing import CheckpointState, compute_trainable_weights_sha256, load_checkpoint, save_checkpoint
from rl_trainer.cpu_master_adamw import CpuMasterAdamW
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import SampledCandidate, perform_verified_grpo_step, run_gpu_training_loop

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class TestCpuMasterCheckpointIdentity(unittest.TestCase):
    def model(self, value=1.0):
        model = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([value], dtype=torch.bfloat16))])
        return model, CpuMasterAdamW(model.parameters(), lr=1e-4, weight_decay=0)

    @staticmethod
    def step(model, optimizer):
        parameter = next(model.parameters())
        sample = SampledCandidate([1], [2], "source", "source", old_token_logps=[-1.0], ref_token_logps=[-1.0])
        return perform_verified_grpo_step(policy=model, trainable_params=list(model.parameters()),
            optimizer=optimizer, samples=[sample], advantages=[1.0],
            logp_fn=lambda m, p, c: -parameter.float().expand(len(c)),
            clip_eps=.2, kl_coef=.04, max_grad_norm=1, device="cpu", old_logp_tolerance=.001)

    def test_real_bf16_update_reports_model_and_master_hashes_separately(self):
        model, optimizer = self.model()
        model_before = compute_trainable_weights_sha256(model.parameters())
        master_before = optimizer.master_weights_sha256()
        metrics = self.step(model, optimizer)
        self.assertTrue(metrics["policy_updated"])
        self.assertGreater(metrics["weight_l2_delta"], 0)
        # The small FP32 update is real but falls below one BF16 ULP.
        self.assertEqual(next(model.parameters()).item(), 1.0)
        self.assertLess(optimizer.master_params[0].item(), 1.0)
        self.assertEqual(metrics["trainable_weights_sha256"], model_before)
        self.assertEqual(metrics["trainable_weights_sha256"], compute_trainable_weights_sha256(model.parameters()))
        self.assertNotEqual(metrics["optimizer_master_weights_sha256"], master_before)
        self.assertEqual(metrics["optimizer_master_weights_sha256"], optimizer.master_weights_sha256())
        self.assertNotEqual(metrics["trainable_weights_sha256"], metrics["optimizer_master_weights_sha256"])

        no_update = perform_verified_grpo_step(policy=model, trainable_params=list(model.parameters()),
            optimizer=optimizer, samples=[], advantages=[], logp_fn=lambda *args: None,
            clip_eps=.2, kl_coef=.04, max_grad_norm=1, device="cpu")
        self.assertFalse(no_update["policy_updated"])
        for key in ("trainable_weights_sha256", "optimizer_master_weights_sha256"):
            self.assertEqual(no_update[key], metrics[key])

    def test_checkpoint_resume_verifies_both_hash_roles_and_loads_legacy_metadata(self):
        model, optimizer = self.model()
        metrics = self.step(model, optimizer)
        state = CheckpointState(1, 1, 42, "test-config", 1, 1,
            optimizer_step_count=1, trainable_weights_sha256=metrics["trainable_weights_sha256"],
            optimizer_master_weights_sha256=metrics["optimizer_master_weights_sha256"])
        with tempfile.TemporaryDirectory() as td:
            step_dir = save_checkpoint(Path(td)/"checkpoint", state, model=model, optimizer=optimizer, save_rng=False)
            restored, restored_optimizer = self.model(-2)
            loaded = load_checkpoint(step_dir, expected_config_hash="test-config", model=restored,
                                     optimizer=restored_optimizer, restore_rng=False)
            self.assertEqual(loaded.trainable_weights_sha256, compute_trainable_weights_sha256(restored.parameters()))
            self.assertEqual(loaded.optimizer_master_weights_sha256, restored_optimizer.master_weights_sha256())
            self.assertNotEqual(loaded.trainable_weights_sha256, loaded.optimizer_master_weights_sha256)
            torch.testing.assert_close(next(restored.parameters()), next(model.parameters()), rtol=0, atol=0)
            torch.testing.assert_close(restored_optimizer.master_params[0], optimizer.master_params[0], rtol=0, atol=0)
            self.assertEqual(restored_optimizer.optimizer.state[restored_optimizer.master_params[0]]["step"].item(), 1)

            metadata_path = step_dir / "checkpoint_state.json"
            metadata = json.loads(metadata_path.read_text())
            legacy = dict(metadata); legacy.pop("optimizer_master_weights_sha256")
            metadata_path.write_text(json.dumps(legacy), encoding="utf-8")
            old_model, old_optimizer = self.model(-3)
            old_state = load_checkpoint(step_dir, model=old_model, optimizer=old_optimizer, restore_rng=False)
            self.assertEqual(old_state.optimizer_master_weights_sha256, "")
            self.assertEqual(old_optimizer.master_weights_sha256(), metrics["optimizer_master_weights_sha256"])

            metadata_path.write_text(json.dumps({**metadata, "optimizer_master_weights_sha256": "0"*64}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "optimizer master weights"):
                load_checkpoint(step_dir, model=restored, optimizer=restored_optimizer, restore_rng=False)
            metadata_path.write_text(json.dumps({**metadata, "trainable_weights_sha256": "0"*64}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trainable weights"):
                load_checkpoint(step_dir, model=restored, optimizer=restored_optimizer, restore_rng=False)

    def test_loop_carries_post_sft_master_identity_through_no_update_and_resume(self):
        model, optimizer = self.model()
        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(backend="numpy_dev", dtype="bf16", sft_epochs=1,
                training_reward_mode="correctness_gated", max_steps=1, checkpoint_every=1,
                checkpoint_dir=str(Path(td)/"checkpoint"), output_jsonl=str(Path(td)/"train.jsonl"))

            def warmup(model, tokenizer, cfg, optimizer, **kwargs):
                # An actual CPU-master update stands in for the module's
                # dataset loop; the integration must retain its exact hashes.
                next(model.parameters()).float().sum().backward()
                updated, delta = optimizer.copy_grads_and_step(1)
                self.assertTrue(updated); self.assertGreater(delta, 0)
                return {"optimizer_step_count": 1}

            def sample(prompt, n, seed):
                return [SampledCandidate([1], [2], "bad", "bad") for _ in range(n)]

            def grader(op, source, **kwargs):
                return dict(status="COMPILE_ERROR", compiled=False, correct=False, cheat_detected=False, reward=0)

            with patch("rl_trainer.supervised_warmup.run_supervised_warmup", side_effect=warmup) as supervised:
                state = run_gpu_training_loop(cfg, model=model, optimizer=optimizer, tokenizer=object(),
                    sample_fn=sample, grader_fn=grader, is_fixture=True)
                supervised.assert_called_once()
            self.assertEqual(state.optimizer_step_count, 0)
            self.assertEqual(state.sft_optimizer_steps, 1)
            self.assertEqual(state.trainable_weights_sha256, compute_trainable_weights_sha256(model.parameters()))
            self.assertEqual(state.optimizer_master_weights_sha256, optimizer.master_weights_sha256())
            records = [json.loads(line) for line in Path(cfg.output_jsonl).read_text().splitlines()]
            self.assertTrue(all(row["optimizer_master_weights_sha256"] == state.optimizer_master_weights_sha256 for row in records))
            sft_state = load_checkpoint(Path(cfg.checkpoint_dir)/"sft_only", restore_rng=False)
            persisted = load_checkpoint(cfg.checkpoint_dir, restore_rng=False)
            for saved in (sft_state, persisted):
                self.assertEqual(saved.trainable_weights_sha256, state.trainable_weights_sha256)
                self.assertEqual(saved.optimizer_master_weights_sha256, state.optimizer_master_weights_sha256)
            restored, restored_optimizer = self.model(-4)
            cfg.resume_from = cfg.checkpoint_dir
            with patch("rl_trainer.supervised_warmup.run_supervised_warmup", side_effect=AssertionError("repeated SFT")):
                resumed = run_gpu_training_loop(cfg, model=restored, optimizer=restored_optimizer,
                    tokenizer=object(), sample_fn=lambda *args: self.fail("completed run sampled again"), is_fixture=True)
            self.assertEqual(resumed.trainable_weights_sha256, state.trainable_weights_sha256)
            self.assertEqual(resumed.optimizer_master_weights_sha256, state.optimizer_master_weights_sha256)

    def test_loop_carries_actual_grpo_master_update_into_returned_checkpoint_state(self):
        model, optimizer = self.model()
        with tempfile.TemporaryDirectory() as td:
            cfg = ExperimentConfig(backend="numpy_dev", dtype="bf16", max_steps=1, checkpoint_every=1,
                checkpoint_dir=str(Path(td)/"checkpoint"), output_jsonl=str(Path(td)/"train.jsonl"))

            def sample(prompt, n, seed):
                return [SampledCandidate([1], [2+i], "good" if i == 0 else "bad", "good" if i == 0 else "bad",
                    old_token_logps=[-float(2+i)], ref_token_logps=[-float(2+i)]) for i in range(n)]

            def grader(op, source, **kwargs):
                passed = source == "good"
                return dict(status="PASS" if passed else "COMPILE_ERROR", compiled=passed,
                            correct=passed, cheat_detected=False, reward=1.0 if passed else 0.0)

            def update(samples, advantages):
                return perform_verified_grpo_step(policy=model, trainable_params=list(model.parameters()),
                    optimizer=optimizer, samples=samples, advantages=advantages,
                    logp_fn=lambda m, p, c: -next(m.parameters()).float().expand(len(c))*c[0],
                    clip_eps=.2, kl_coef=.04, max_grad_norm=1, device="cpu", old_logp_tolerance=.001)

            state = run_gpu_training_loop(cfg, model=model, optimizer=optimizer,
                sample_fn=sample, grader_fn=grader, update_fn=update, is_fixture=True)
            persisted = load_checkpoint(cfg.checkpoint_dir, restore_rng=False)
            self.assertEqual(state.optimizer_step_count, 1)
            self.assertEqual(state.trainable_weights_sha256, persisted.trainable_weights_sha256)
            self.assertEqual(state.optimizer_master_weights_sha256, persisted.optimizer_master_weights_sha256)
            self.assertNotEqual(state.trainable_weights_sha256, state.optimizer_master_weights_sha256)


if __name__ == "__main__":
    unittest.main()
