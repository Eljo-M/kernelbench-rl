import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kernels.triton_baselines import TRITON_BASELINES
from rl_trainer.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from rl_trainer.gpu_train_grpo import (
    SampledCandidate,
    compute_selected_token_logps,
    compute_selected_token_logps_numpy,
    perform_verified_grpo_step,
)
from rl_trainer.prompting import build_prompt, describe_operator_contract, format_prompt_for_tokenizer

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _FakeChatTokenizer:
    chat_template = "chatml"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<|sys|>{messages[0]['content']}<|usr|>{messages[1]['content']}<|asst|>"


class TestLiveRunAuditFixes(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "PyTorch required")
    def test_full_parameter_adafactor_update_and_checkpoint_resume(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(16, 8)
                self.head = nn.Linear(8, 16)
            def forward(self, ids, logits_to_keep=None):
                return SimpleNamespace(logits=self.head(self.emb(ids)))
        model = Tiny()
        params = list(model.parameters())
        optimizer = torch.optim.Adafactor(params, lr=0.01, foreach=False)
        lp = lambda m, p, c: compute_selected_token_logps(m, p, c, "cpu")
        with torch.no_grad():
            old = lp(model, [1, 2], [3, 4]).tolist()
        sample = SampledCandidate([1, 2], [3, 4], "", "", old, old)
        metrics = perform_verified_grpo_step(policy=model, trainable_params=params,
            optimizer=optimizer, samples=[sample], advantages=[1.0], logp_fn=lp,
            clip_eps=0.2, kl_coef=0.04, max_grad_norm=1.0, device="cpu")
        self.assertGreater(metrics["weight_l2_delta"], 0)
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as td:
            state = CheckpointState(1, 1, 42, "full", 1, 1, optimizer_step_count=1)
            save_checkpoint(Path(td), state, model=model, optimizer=optimizer)
            with torch.no_grad():
                for p in params: p.zero_()
            load_checkpoint(td, expected_config_hash="full", model=model, optimizer=optimizer)
            self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items()))

    def test_contract_derived_prompt_contains_exact_signature_shapes_and_dtypes(self):
        for op_name in TRITON_BASELINES:
            info = describe_operator_contract(op_name)
            prompt = build_prompt(op_name, backend="triton")
            self.assertIn(info["signature"], prompt)
            self.assertIn(str(info["output_shape"]), prompt)
            self.assertIn('input_precision="ieee"', prompt)

        # Quantized INT8 op explicitly warns to keep qweight integer
        i8_prompt = build_prompt("int8_weight_only_gemm_512", backend="triton")
        self.assertIn("def kernel_forward(x, qweight, scales):", i8_prompt)
        self.assertIn("KEEP INTEGER", i8_prompt)
        self.assertIn("int8", i8_prompt)

        # Scalar default preserved verbatim (eps=1e-6)
        rms_prompt = build_prompt("rmsnorm_2048", backend="triton")
        self.assertIn("def kernel_forward(x, weight, eps=1e-06):", rms_prompt)

    def test_chat_template_applied_consistently_and_recorded(self):
        raw = build_prompt("rmsnorm_2048", backend="triton")
        fmt_chat, applied_chat, sha_chat = format_prompt_for_tokenizer(raw, _FakeChatTokenizer())
        self.assertTrue(applied_chat)
        self.assertIn("<|asst|>", fmt_chat)
        self.assertEqual(len(sha_chat), 64)

        fmt_plain, applied_plain, sha_plain = format_prompt_for_tokenizer(raw, None)
        self.assertFalse(applied_plain)
        self.assertEqual(fmt_plain, raw)
        self.assertNotEqual(sha_chat, sha_plain)

    def test_chunked_selected_logps_matches_full_log_softmax_exactly(self):
        rng = np.random.default_rng(123)
        logits = rng.standard_normal((75, 257)).astype(np.float64) * 3.0
        targets = rng.integers(0, 257, size=(75,), dtype=np.int64)

        chunked = compute_selected_token_logps_numpy(logits, targets, chunk_size=16)
        m = np.max(logits, axis=-1, keepdims=True)
        full_log_softmax = (logits - m) - np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))
        expected = full_log_softmax[np.arange(75), targets]
        self.assertTrue(np.allclose(chunked, expected, rtol=1e-12, atol=1e-12))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in this sandbox; runs automatically on Torch CPU/GPU hosts.")
    def test_real_tiny_torch_backward_logits_to_keep_weight_delta_and_qlora_resume(self):
        torch.manual_seed(7)
        vocab = 32
        dim = 16

        class _TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab, dim)
                self.base_proj = nn.Linear(dim, vocab, bias=False)
                self.lora_a = nn.Parameter(torch.randn(dim, 4) * 0.05)
                self.lora_b = nn.Parameter(torch.zeros(4, vocab))
                self.use_adapter = True

            def forward(self, input_ids, logits_to_keep=None):
                h = self.emb(input_ids)
                if logits_to_keep is not None:
                    h = h[:, -logits_to_keep:, :]
                out = self.base_proj(h)
                if self.use_adapter:
                    out = out + (h @ self.lora_a) @ self.lora_b
                return SimpleNamespace(logits=out)

            @contextlib.contextmanager
            def disable_adapter(self):
                prev = self.use_adapter
                self.use_adapter = False
                try:
                    yield
                finally:
                    self.use_adapter = prev

            def save_pretrained(self, out_dir: str):
                p = Path(out_dir)
                p.mkdir(parents=True, exist_ok=True)
                torch.save({"lora_a": self.lora_a.detach().cpu(), "lora_b": self.lora_b.detach().cpu()}, p / "adapter_state.pt")

            def load_adapter_in_place(self, adapter_dir: str):
                sd = torch.load(Path(adapter_dir) / "adapter_state.pt", map_location="cpu", weights_only=True)
                with torch.no_grad():
                    self.lora_a.copy_(sd["lora_a"])
                    self.lora_b.copy_(sd["lora_b"])

        # Freeze base weights like 7B QLoRA; only lora_a and lora_b are trainable
        model = _TinyLM()
        for p in model.emb.parameters():
            p.requires_grad_(False)
        for p in model.base_proj.parameters():
            p.requires_grad_(False)

        trainable = [model.lora_a, model.lora_b]
        orig_param_ids = [id(p) for p in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=1e-2)

        prompt_ids = [1, 2, 3]
        comp_ids = [4, 5, 6, 7]

        # Capture original frozen base reference output before any updates
        with torch.no_grad(), model.disable_adapter():
            ref_lp_before = compute_selected_token_logps(model, prompt_ids, comp_ids, "cpu", chunk_size=2).clone()
        with torch.no_grad():
            old_lp = compute_selected_token_logps(model, prompt_ids, comp_ids, "cpu", chunk_size=2).tolist()

        smp = SampledCandidate(prompt_ids, comp_ids, "x", "x", old_lp, ref_lp_before.tolist())
        metrics = perform_verified_grpo_step(
            policy=model,
            trainable_params=trainable,
            optimizer=optimizer,
            samples=[smp],
            advantages=[1.0],
            logp_fn=lambda m, p_ids, c_ids: compute_selected_token_logps(m, p_ids, c_ids, "cpu", chunk_size=2),
            clip_eps=0.2,
            kl_coef=0.04,
            max_grad_norm=1.0,
            device="cpu",
        )
        self.assertTrue(metrics["policy_updated"])
        self.assertGreater(metrics["weight_l2_delta"], 0.0)
        self.assertEqual(len(metrics["trainable_weights_sha256"]), 64)

        # Save adapter checkpoint, perturb adapter weights, resume in-place, verify param IDs & frozen ref
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td) / "ckpt"
            st = CheckpointState(1, 1, 42, "h1", 1.0, 2, optimizer_step_count=1, training_mode="qlora_7b_nf4")
            save_checkpoint(ckpt_dir, st, model=model, optimizer=optimizer, adapter_only=True)
            saved_a = model.lora_a.detach().clone()

            with torch.no_grad():
                model.lora_a.zero_()
            loaded = load_checkpoint(ckpt_dir, expected_config_hash="h1", model=model, optimizer=optimizer)
            self.assertEqual(loaded.optimizer_step_count, 1)
            self.assertEqual([id(p) for p in trainable], orig_param_ids)
            self.assertTrue(torch.allclose(model.lora_a, saved_a))

            # Frozen base reference via disable_adapter() must remain bit-identical
            with torch.no_grad(), model.disable_adapter():
                ref_lp_after = compute_selected_token_logps(model, prompt_ids, comp_ids, "cpu", chunk_size=2)
            self.assertTrue(torch.equal(ref_lp_before, ref_lp_after))


if __name__ == "__main__":
    unittest.main()
