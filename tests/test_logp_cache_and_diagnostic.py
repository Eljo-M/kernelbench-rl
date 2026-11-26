"""Unit tests for teacher-forced use_cache=False forwarding and rl_trainer/diagnose_logps.py."""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from rl_trainer.checkpointing import CheckpointState, save_checkpoint
from rl_trainer.diagnose_logps import run_logp_diagnostic
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import _forward_supports_kwarg, compute_selected_token_logps

try:
    import torch  # type: ignore

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestForwardKwargDetectionAndDiagnostic(unittest.TestCase):
    def test_forward_kwarg_detection_across_plain_peft_and_kwargs_wrappers(self) -> None:
        class TinyNoKwargs:
            def forward(self, input_ids: Any) -> Any:
                return input_ids

        class FullCausalLM:
            def forward(self, input_ids: Any, use_cache: bool | None = None, logits_to_keep: int = 0) -> Any:
                return input_ids

        class PeftStyleWrapper:
            def __init__(self, base: Any) -> None:
                self._base = base

            def get_base_model(self) -> Any:
                return self._base

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self._base.forward(*args, **kwargs)

        class PureKwargsMock:
            def forward(self, input_ids: Any, **kwargs: Any) -> Any:
                return input_ids

        self.assertFalse(_forward_supports_kwarg(TinyNoKwargs(), "use_cache"))
        self.assertFalse(_forward_supports_kwarg(TinyNoKwargs(), "logits_to_keep"))

        self.assertTrue(_forward_supports_kwarg(FullCausalLM(), "use_cache"))
        self.assertTrue(_forward_supports_kwarg(FullCausalLM(), "logits_to_keep"))

        peft_full = PeftStyleWrapper(FullCausalLM())
        self.assertTrue(_forward_supports_kwarg(peft_full, "use_cache"))
        self.assertTrue(_forward_supports_kwarg(peft_full, "logits_to_keep"))

        peft_tiny = PeftStyleWrapper(TinyNoKwargs())
        self.assertFalse(_forward_supports_kwarg(peft_tiny, "use_cache"))
        self.assertFalse(_forward_supports_kwarg(peft_tiny, "logits_to_keep"))

        self.assertTrue(_forward_supports_kwarg(PureKwargsMock(), "use_cache"))
        self.assertTrue(_forward_supports_kwarg(PureKwargsMock(), "logits_to_keep"))

    def test_diagnose_logps_records_cache_diff_repeatability_and_initial_checkpoint_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = ExperimentConfig(
                model_name_or_path="tiny-local-model",
                model_revision="rev-local-0",
                training_mode="qlora_7b_nf4",
                backend="numpy_dev",
                dtype="fp16",
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                max_steps=2,
                checkpoint_every=1,
            )
            cfg_path = root / "config.json"
            cfg.to_json(cfg_path)

            # Create a zero-update initial checkpoint with an adapter file
            ckpt_root = root / "init_ckpt"
            step_dir = save_checkpoint(
                ckpt_root,
                CheckpointState(
                    step=0,
                    op_cursor=0,
                    seed=cfg.seed,
                    config_hash=cfg.config_hash(),
                    best_mean_reward=0.0,
                    total_rollouts=0,
                    optimizer_step_count=0,
                    training_mode=cfg.training_mode,
                ),
                model=None,
                optimizer=None,
                adapter_only=True,
            )
            adapter_dir = step_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_model.bin").write_bytes(b"zero-update-adapter-bytes")
            state_path = step_dir / "checkpoint_state.json"
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            state_data["adapter_dir"] = "adapter"
            state_path.write_text(json.dumps(state_data, indent=2) + "\n", encoding="utf-8")

            # Stored trace with old_token_logps matching use_cache=True
            stored_old = [-1.2, -0.8, -2.1, -0.5]
            stored_ref = [-1.2, -0.8, -2.1, -0.5]
            trace_path = root / "train.jsonl"
            trace_row = {
                "run_id": "step1-g0",
                "op_name": "rmsnorm_2048",
                "config_hash": cfg.config_hash(),
                "policy_updated": False,
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4, 5, 6, 7],
                "old_token_logps": stored_old,
                "reference_token_logps": stored_ref,
            }
            trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
            orig_trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()

            class FakePolicy:
                def __init__(self) -> None:
                    self.config = types.SimpleNamespace(use_cache=True)
                    self.training = False
                    self._adapter_disabled = False

                def eval(self) -> "FakePolicy":
                    self.training = False
                    return self

                def train(self) -> "FakePolicy":
                    self.training = True
                    return self

                def modules(self) -> list[Any]:
                    return []

                @contextlib.contextmanager
                def disable_adapter(self) -> Any:
                    self._adapter_disabled = True
                    try:
                        yield
                    finally:
                        self._adapter_disabled = False

            fake_policy = FakePolicy()

            def _fake_eval(pol: Any, p_ids: list[int], c_ids: list[int], use_cache: bool) -> list[float]:
                base = np.asarray(stored_old, dtype=np.float64)
                if not use_cache:
                    # Simulate a 0.013374 shift between use_cache=True and use_cache=False
                    base = base + np.asarray([0.013374, 0.002, 0.001, 0.0], dtype=np.float64)
                if pol.training:
                    base = base + np.asarray([0.0, 0.0005, 0.0, 0.0], dtype=np.float64)
                return base.tolist()

            out_json = root / "diag.json"
            report = run_logp_diagnostic(
                config_path=cfg_path,
                input_jsonl=trace_path,
                out_json=out_json,
                initial_checkpoint=ckpt_root,
                sample_index=0,
                policy_loader=lambda _c: (fake_policy, [], None, None, "cpu"),
                logp_eval_fn=_fake_eval,
            )

            # Raw trace untouched, config.use_cache preserved, strict finite JSON written
            self.assertEqual(hashlib.sha256(trace_path.read_bytes()).hexdigest(), orig_trace_sha)
            self.assertTrue(fake_policy.config.use_cache)
            self.assertFalse(report["policy_updated"])
            self.assertFalse(report["tolerance_changed"])
            self.assertIsNotNone(report["initial_checkpoint"])
            self.assertEqual(report["initial_checkpoint"]["optimizer_step_count"], 0)
            self.assertEqual(len(report["initial_checkpoint"]["adapter_weights_sha256"]), 64)

            cmp_map = report["comparisons"]
            self.assertAlmostEqual(cmp_map["use_cache_true_vs_false_eval"]["max_abs_diff"], 0.013374, places=6)
            self.assertEqual(cmp_map["same_process_repeatability_eval_use_cache_false"]["max_abs_diff"], 0.0)
            self.assertAlmostEqual(cmp_map["eval_vs_train_dropout_off_use_cache_false"]["max_abs_diff"], 0.0005, places=6)
            self.assertEqual(cmp_map["stored_old_vs_recomputed_eval_use_cache_true"]["max_abs_diff"], 0.0)
            self.assertAlmostEqual(
                cmp_map["stored_old_vs_recomputed_eval_use_cache_false"]["max_abs_diff"], 0.013374, places=6
            )

            # Refuses to overwrite existing output JSON
            with self.assertRaises(FileExistsError):
                run_logp_diagnostic(
                    config_path=cfg_path,
                    input_jsonl=trace_path,
                    out_json=out_json,
                    policy_loader=lambda _c: (fake_policy, [], None, None, "cpu"),
                    logp_eval_fn=_fake_eval,
                )

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in this sandbox; runs automatically on Torch CPU/GPU hosts.")
    def test_real_torch_use_cache_false_reaches_forward_preserves_config_and_matches_full_autograd(self) -> None:
        torch.manual_seed(0)

        class TinyCausalWithCache(torch.nn.Module):
            def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(use_cache=True)
                self.emb = torch.nn.Embedding(vocab, hidden)
                self.head = torch.nn.Linear(hidden, vocab)
                self.recorded_calls: list[dict[str, Any]] = []

            def forward(
                self,
                input_ids: Any,
                use_cache: bool | None = None,
                logits_to_keep: int = 0,
            ) -> Any:
                self.recorded_calls.append({"use_cache": use_cache, "logits_to_keep": logits_to_keep})
                h = self.emb(input_ids)
                if logits_to_keep and logits_to_keep > 0:
                    h = h[:, -logits_to_keep:, :]
                return types.SimpleNamespace(logits=self.head(h))

        class PeftLikeWrapper(torch.nn.Module):
            def __init__(self, base: torch.nn.Module) -> None:
                super().__init__()
                self._base = base
                self.config = base.config

            def get_base_model(self) -> torch.nn.Module:
                return self._base

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self._base(*args, **kwargs)

        class TinyNoCacheModel(torch.nn.Module):
            def __init__(self, base: TinyCausalWithCache) -> None:
                super().__init__()
                self.emb = base.emb
                self.head = base.head

            def forward(self, input_ids: Any) -> Any:
                return types.SimpleNamespace(logits=self.head(self.emb(input_ids)))

        base_lm = TinyCausalWithCache()
        peft_lm = PeftLikeWrapper(base_lm)
        full_lm = TinyNoCacheModel(base_lm)

        prompt_ids = [1, 4, 7, 2]
        comp_ids = [5, 9, 3, 8, 6]

        # 1. Through PEFT-style **kwargs wrapper with outer config.use_cache=True
        lp_trunc = compute_selected_token_logps(peft_lm, prompt_ids, comp_ids, "cpu", chunk_size=2)
        self.assertTrue(peft_lm.config.use_cache)
        self.assertTrue(base_lm.config.use_cache)
        self.assertEqual(len(base_lm.recorded_calls), 1)
        self.assertIs(base_lm.recorded_calls[0]["use_cache"], False)
        self.assertEqual(base_lm.recorded_calls[0]["logits_to_keep"], len(comp_ids) + 1)

        # Verify autograd gradients match full-sequence model (without use_cache / logits_to_keep)
        base_lm.zero_grad()
        lp_trunc.sum().backward()
        grad_emb_trunc = base_lm.emb.weight.grad.detach().clone()
        grad_head_trunc = base_lm.head.weight.grad.detach().clone()

        base_lm.zero_grad()
        lp_full = compute_selected_token_logps(full_lm, prompt_ids, comp_ids, "cpu", chunk_size=2)
        lp_full.sum().backward()
        grad_emb_full = base_lm.emb.weight.grad.detach().clone()
        grad_head_full = base_lm.head.weight.grad.detach().clone()

        self.assertTrue(torch.allclose(lp_trunc.detach(), lp_full.detach(), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(grad_emb_trunc, grad_emb_full, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(grad_head_trunc, grad_head_full, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
