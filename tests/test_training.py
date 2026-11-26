import asyncio
import importlib.util
import json
import math
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from contracts.schemas import KernelEvalResult
from rl_trainer.grpo_reward import compute_grpo_advantages
from rl_trainer.objective import grpo_loss_numpy, grpo_loss_torch
from rl_trainer.train_grpo import (DRY_SOURCES, RunConfig, VLLMGenerator, extract_source,
                                   grade_sources_sync, main, run_dry_loop, run_search, run_training)


def result(index=0, correct=False):
    return KernelEvalResult(run_id=str(index), op_name="rmsnorm_2048", level=1, backend="numpy_dev",
                            status="PASS" if correct else "COMPILE_ERROR", compiled=correct, correct=correct,
                            cheat_detected=False, max_abs_err=0.0, eager_ms=1.0 if correct else 0.0,
                            compile_ms=0.0, candidate_ms=1.0 if correct else 0.0,
                            speedup_vs_eager=1.0 if correct else 0.0, speedup_vs_compile=0.0,
                            reward=float(correct), error_message="" if correct else "bad syntax")


class TestTraining(unittest.TestCase):
    def test_config_rejects_invalid_limits(self):
        for kwargs in ({"group_size": 1}, {"workers": 0}, {"timeout_secs": float("nan")},
                       {"repair_turns": 4}, {"learning_rate": -1}, {"clip": 1.1}, {"beta": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RunConfig(**kwargs).validate()

    def test_advantages_handle_equal_and_invalid_rewards(self):
        self.assertEqual(compute_grpo_advantages([1, 1, 1]), [0, 0, 0])
        self.assertEqual(compute_grpo_advantages([]), [])
        for rewards in ([float("inf")], [float("nan")], [[1], [2]]):
            with self.assertRaises(ValueError):
                compute_grpo_advantages(rewards)

    def test_source_extraction(self):
        self.assertEqual(extract_source("Answer:\n```python\nx = 1\n```"), "x = 1\n")
        with self.assertRaises(ValueError):
            extract_source("```py\na=1\n```\n```py\nb=2\n```")

    def test_objective_mask_clipping_and_kl(self):
        old = np.zeros((2, 2))
        new = np.log([[2.0, 1.0], [0.5, 1.0]])
        ref = new.copy()
        mask = [[1, 0], [1, 0]]
        # +advantage uses capped 1.2; -advantage uses capped 0.8.
        self.assertAlmostEqual(grpo_loss_numpy(new, old, ref, [1, -1], mask), -0.2)
        self.assertEqual(grpo_loss_numpy(old, old, old, [0, 0], [[1, 1], [1, 1]]), 0.0)
        self.assertGreater(grpo_loss_numpy(old, old, old + 1, [0, 0], mask), 0)
        with self.assertRaises(ValueError):
            grpo_loss_numpy(old, old, old, [0, 0], [[0, 0], [1, 1]])

    def test_grading_is_bounded_and_ordered(self):
        active, maximum = 0, 0
        lock = threading.Lock()
        def verify(op, source, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.025)
            with lock:
                active -= 1
            return result(correct=source == "pass")
        config = RunConfig(group_size=4, workers=2, timeout_secs=3)
        with patch("rl_trainer.train_grpo.verify_candidate_source", side_effect=verify):
            rows = grade_sources_sync("rmsnorm_2048", ["fail", "pass", "pass", "fail"], group_id="group", config=config)
        self.assertEqual(maximum, 2)
        self.assertEqual([row["candidate_index"] for row in rows], [0, 1, 2, 3])
        self.assertAlmostEqual(sum(row["grpo_advantage"] for row in rows), 0)
        self.assertEqual(len(rows[0]["source_sha256"]), 64)

    def test_search_repair_keeps_group_provenance(self):
        class FakeGenerator:
            async def generate(self, prompt, count, seed):
                return ["pass"] if "Verifier feedback:" in prompt else ["fail", "pass"]
        def verify(op, source, **kwargs):
            return result(correct=source.strip() == "pass")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search.jsonl"
            with patch("rl_trainer.train_grpo.verify_candidate_source", side_effect=verify):
                asyncio.run(run_search(FakeGenerator(), ["rmsnorm_2048"], path, RunConfig(group_size=2, repair_turns=2)))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["repair_turn"] for row in rows], [0, 0, 1, 1])
            self.assertTrue(all(row["correct"] for row in rows[2:]))
            self.assertTrue(all(not row["policy_updated"] for row in rows))

    def test_vllm_url_validation_and_async_order(self):
        for url in ("file:///tmp/model", "https://user:secret@example.com/v1", "https://example.com/v1?key=secret"):
            with self.assertRaises(ValueError):
                VLLMGenerator(url, "model")
        generator = VLLMGenerator("http://localhost:8000/v1", "model")
        with patch.object(generator, "_request", side_effect=lambda prompt, seed: str(seed)):
            self.assertEqual(asyncio.run(generator.generate("prompt", 3, 4)), ["4", "5", "6"])

    def test_dry_run_is_strict_fixture_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dry.jsonl"
            run_dry_loop(path)
            rows = [json.loads(line, parse_constant=lambda value: self.fail(value)) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["status"] for row in rows}, {"PASS", "NUMERICAL_MISMATCH", "CHEAT", "COMPILE_ERROR"})
            self.assertTrue(all(row["fixture"] and not row["policy_updated"] for row in rows))
            self.assertTrue(all(row["compile_ms"] == row["speedup_vs_compile"] == 0 for row in rows))

    def test_training_refuses_unsupported_repairs_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "repair trajectories"):
            run_training(["rmsnorm_2048"], Path("unused"), Path("unused"), RunConfig(repair_turns=1))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch package unavailable")
    def test_torch_objective_matches_numpy_and_updates_parameters(self):
        import torch
        parameter = torch.tensor([[0.1, 0.2], [-0.1, 0.05]], dtype=torch.float64, requires_grad=True)
        old = torch.zeros_like(parameter)
        ref = torch.zeros_like(parameter)
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        loss = grpo_loss_torch(parameter, old, ref, advantages, mask)
        self.assertAlmostEqual(float(loss.detach()), grpo_loss_numpy(parameter.detach(), old, ref, advantages, mask), places=10)
        loss.backward()
        self.assertEqual(parameter.grad[0, 1].item(), 0.0)
        before = parameter.detach().clone()
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        optimizer.step()
        self.assertFalse(torch.equal(before, parameter.detach()))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch package unavailable")
    def test_training_updates_local_tiny_policy_and_saves_checkpoint(self):
        """Exercise orchestration/autograd with a tiny local fixture; no model download."""
        import torch
        models = []
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(4))
                self.config = types.SimpleNamespace(is_encoder_decoder=False)
            def forward(self, input_ids, **kwargs):
                logits = self.bias.reshape(1, 1, 4).expand(input_ids.shape[0], input_ids.shape[1], 4)
                return types.SimpleNamespace(logits=logits)
            def generate(self, input_ids, generation_config, **kwargs):
                self.last_generation_config = generation_config
                return torch.tensor([[1, 2, 0], [1, 3, 0]], device=input_ids.device)
            def save_pretrained(self, path):
                torch.save(self.state_dict(), Path(path) / "fixture-policy.pt")
        class Batch(dict):
            def to(self, device):
                return Batch({key: value.to(device) for key, value in self.items()})
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0
            chat_template = None
            def __call__(self, prompt, **kwargs):
                return Batch(input_ids=torch.tensor([[1]]))
            def decode(self, tokens, **kwargs):
                return "pass" if int(tokens[0]) == 2 else "fail"
            def save_pretrained(self, path):
                (Path(path) / "fixture-tokenizer.json").write_text("{}")
        def load_policy(*args, **kwargs):
            self.assertTrue(kwargs["local_files_only"])
            self.assertFalse(kwargs["trust_remote_code"])
            model = TinyPolicy()
            models.append(model)
            return model
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load_policy),
            GenerationConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        )
        def verify(op, source, **kwargs):
            return result(correct=source.strip() == "pass")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rollouts.jsonl"
            checkpoints = Path(directory) / "checkpoints"
            with patch.dict("sys.modules", {"transformers": fake_transformers}), \
                 patch("rl_trainer.train_grpo.verify_candidate_source", side_effect=verify):
                run_training(["rmsnorm_2048"], output, checkpoints,
                             RunConfig(model="local-fixture", group_size=2, learning_rate=0.01), device="cpu")
            self.assertFalse(torch.equal(models[0].bias.detach(), torch.zeros(4)))
            self.assertTrue(torch.equal(models[1].bias.detach(), torch.zeros(4)))
            self.assertEqual(models[0].last_generation_config.top_k, 0)
            self.assertTrue((checkpoints / "step-000001" / "optimizer.pt").exists())
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(all(row["policy_updated"] and math.isfinite(row["loss"]) for row in rows))


if __name__ == "__main__":
    unittest.main()
