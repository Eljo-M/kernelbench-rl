"""Focused unit tests for rl_trainer/supervised_warmup.py."""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Iterator, Mapping

from kernels.triton_baselines import TRITON_BASELINES
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.supervised_warmup import (
    build_supervised_dataset,
    load_verified_triton_sources,
    run_supervised_warmup,
)

try:
    import torch  # type: ignore

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _TinyCharTokenizer:
    """Deterministic small tokenizer with chat template and EOS for unit testing."""

    eos_token_id: int = 2
    vocab_size: int = 128
    chat_template: str = "tiny-chat-template"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [(ord(ch) % 120) + 3 for ch in text]


class _TrackingSourceMap(Mapping[str, str]):
    """Mapping wrapper that records every key accessed to prove heldout_ops are never touched."""

    def __init__(self, backing: dict[str, str], forbidden_keys: set[str]) -> None:
        self._backing = backing
        self._forbidden = forbidden_keys
        self.accessed_keys: list[str] = []

    def __getitem__(self, key: str) -> str:
        if key in self._forbidden:
            raise AssertionError(f"Held-out operator '{key}' must NEVER be looked up during supervised warmup!")
        self.accessed_keys.append(key)
        return self._backing[key]

    def __contains__(self, key: object) -> bool:
        if key in self._forbidden:
            raise AssertionError(f"Held-out operator '{key}' must NEVER be checked during supervised warmup!")
        return key in self._backing

    def __iter__(self) -> Iterator[str]:
        return iter(self._backing)

    def __len__(self) -> int:
        return len(self._backing)


class TestSupervisedWarmup(unittest.TestCase):
    def test_load_verified_sources_covers_all_30_ops_and_never_touches_heldout(self) -> None:
        sources = load_verified_triton_sources()
        self.assertEqual(len(sources), 30)

        cfg = ExperimentConfig(
            train_ops=["rmsnorm_2048", "layernorm_2048"],
            heldout_ops=["rmsnorm_4096", "int8_weight_only_gemm_512"],
            backend="triton",
        )
        tok = _TinyCharTokenizer()
        tracker = _TrackingSourceMap(sources, forbidden_keys=set(cfg.heldout_ops))
        examples = build_supervised_dataset(cfg, tok, source_map=tracker)

        self.assertEqual([ex.op_name for ex in examples], ["rmsnorm_2048", "layernorm_2048"])
        self.assertEqual(tracker.accessed_keys, ["rmsnorm_2048", "layernorm_2048"])
        for ex in examples:
            self.assertTrue(ex.chat_template_applied)
            self.assertTrue(ex.target_completion.startswith("```python\n"))
            self.assertTrue(ex.target_completion.endswith("\n```"))
            self.assertEqual(ex.completion_ids[-1], tok.eos_token_id)

    def test_rejects_overlap_bad_policy_and_overlength_without_truncation(self) -> None:
        tok = _TinyCharTokenizer()

        # 1. Overlap between train_ops and heldout_ops rejected
        cfg_leak = types.SimpleNamespace(
            train_ops=["rmsnorm_2048", "rmsnorm_4096"],
            heldout_ops=["rmsnorm_4096"],
            backend="triton",
        )
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            build_supervised_dataset(cfg_leak, tok)

        # 2. Training source violating strict GPU source policy rejected
        bad_source = "import torch\ndef kernel_forward(x, w):\n    return x.float() * w\n"
        cfg_bad = types.SimpleNamespace(
            train_ops=["rmsnorm_2048"],
            heldout_ops=["rmsnorm_4096"],
            backend="triton",
        )
        with self.assertRaisesRegex(ValueError, "strict GPU source policy"):
            build_supervised_dataset(cfg_bad, tok, source_map={"rmsnorm_2048": bad_source})

        # 3. Overlength sample rejected explicitly rather than truncated
        cfg_ok = types.SimpleNamespace(
            train_ops=["rmsnorm_2048"],
            heldout_ops=["rmsnorm_4096"],
            backend="triton",
        )
        with self.assertRaisesRegex(ValueError, "Overlength supervised example"):
            build_supervised_dataset(cfg_ok, tok, source_map={"rmsnorm_2048": TRITON_BASELINES["rmsnorm_2048"]}, max_tokens=64)

    def test_zero_epochs_is_clean_noop_without_mutating_config_or_cache(self) -> None:
        cfg = types.SimpleNamespace(
            train_ops=["rmsnorm_2048"],
            heldout_ops=["rmsnorm_4096"],
            backend="triton",
            sft_epochs=0,
        )
        fake_model = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True), training=False)
        fake_opt = types.SimpleNamespace(param_groups=[{"lr": 1e-5}])
        tracker = _TrackingSourceMap(dict(TRITON_BASELINES), forbidden_keys={"rmsnorm_2048", "rmsnorm_4096"})

        metrics = run_supervised_warmup(
            fake_model,
            _TinyCharTokenizer(),
            cfg,
            fake_opt,
            device="cpu",
            trainable_params=[],
            source_map=tracker,
        )
        self.assertFalse(metrics["ran"])
        self.assertEqual(metrics["optimizer_step_count"], 0)
        self.assertEqual(metrics["weight_l2_delta"], 0.0)
        self.assertTrue(fake_model.config.use_cache)
        self.assertEqual(fake_opt.param_groups[0]["lr"], 1e-5)
        self.assertEqual(tracker.accessed_keys, [])

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in this sandbox; runs automatically on Torch CPU/GPU hosts.")
    def test_real_tiny_torch_warmup_masks_prompt_tokens_restores_lr_and_preserves_cache_flag(self) -> None:
        torch.manual_seed(7)

        class TinyCausalLM(torch.nn.Module):
            def __init__(self, vocab: int = 128, hidden: int = 16) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(use_cache=True)
                self.emb = torch.nn.Embedding(vocab, hidden)
                self.drop = torch.nn.Dropout(p=0.5)
                self.head = torch.nn.Linear(hidden, vocab)
                self.seen_use_cache: list[bool | None] = []

            def forward(
                self,
                input_ids: Any,
                use_cache: bool | None = None,
                logits_to_keep: int = 0,
            ) -> Any:
                self.seen_use_cache.append(use_cache)
                h = self.drop(self.emb(input_ids))
                if logits_to_keep and logits_to_keep > 0:
                    h = h[:, -logits_to_keep:, :]
                return types.SimpleNamespace(logits=self.head(h))

        # Explicit gradient-level prompt masking check: token 125 only appears at prompt[0]
        from rl_trainer.gpu_train_grpo import compute_selected_token_logps

        mask_model = TinyCausalLM()
        mask_model.eval()
        p_ids = [125, 10, 11, 12]
        c_ids = [20, 21, 22, 2]
        mask_model.zero_grad()
        loss_mask = -compute_selected_token_logps(mask_model, p_ids, c_ids, "cpu", use_cache=False).mean()
        loss_mask.backward()
        self.assertTrue(torch.all(mask_model.emb.weight.grad[125] == 0.0))
        self.assertGreater(float(mask_model.emb.weight.grad[20].abs().sum().item()), 0.0)

        # 1. Verify strict prompt masking: changing a prompt-only token embedding gets zero gradient
        # when completion logits depend only on positions [p_len - 1 .. p_len + c_len - 2].
        # (In this token-wise embedding model, prompt token 0 never feeds completion logits.)
        model = TinyCausalLM()
        tok = _TinyCharTokenizer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        with tempfile.TemporaryDirectory() as tmp:
            out_jsonl = Path(tmp) / "sft.jsonl"
            cfg = types.SimpleNamespace(
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                backend="triton",
                seed=42,
                max_grad_norm=1.0,
                sft_epochs=2,
                sft_learning_rate=5e-3,
                sft_max_tokens=4096,
                sft_output_jsonl=str(out_jsonl),
            )
            tracker = _TrackingSourceMap(
                {"rmsnorm_2048": TRITON_BASELINES["rmsnorm_2048"]},
                forbidden_keys={"rmsnorm_4096"},
            )
            metrics = run_supervised_warmup(
                model,
                tok,
                cfg,
                optimizer,
                device="cpu",
                trainable_params=list(model.parameters()),
                source_map=tracker,
            )

            self.assertTrue(metrics["ran"])
            self.assertEqual(metrics["optimizer_step_count"], 2)
            self.assertGreater(metrics["weight_l2_delta"], 0.0)
            self.assertNotEqual(metrics["weight_sha256_before"], metrics["weight_sha256_after"])
            # Outer config.use_cache untouched, while every forward received use_cache=False
            self.assertTrue(model.config.use_cache)
            self.assertEqual(model.seen_use_cache, [False, False])
            # Original optimizer LR restored cleanly
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5, places=12)
            # Dropout was disabled during warmup
            self.assertFalse(model.drop.training)
            # JSONL written with finite records
            lines = [json.loads(ln) for ln in out_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["op_name"], "rmsnorm_2048")


if __name__ == "__main__":
    unittest.main()
