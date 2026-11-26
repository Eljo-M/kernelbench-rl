import contextlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.gpu_train_grpo import compute_selected_token_logps
from rl_trainer.prompting import sha256_text
from rl_trainer.recover_initial_batch import (
    RECOVERY_METHOD,
    RECOVERY_SCHEDULE,
    recover_initial_batch,
    validate_initial_batch_traces,
    verify_stored_logps_against_policy,
)

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

VALID_RMSNORM_NP = (
    "import numpy as np\n"
    "def kernel_forward(x, w, eps=1e-6):\n"
    "    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)) * w\n"
)
BAD_RMSNORM_NP = "import numpy as np\ndef kernel_forward(x, w, eps=1e-6): return x * 0.0\n"


def _make_8group_16sample_rows(cfg: ExperimentConfig, old_lps=None, ref_lps=None) -> list[dict]:
    rows = []
    olp = old_lps or [-1.1, -0.9, -0.4]
    rlp = ref_lps or [-1.1, -0.9, -0.4]
    for step in range(1, 9):
        op_name = cfg.train_ops[(step - 1) % len(cfg.train_ops)]
        prompt_ids = [1, 2, step + 10]
        for g_idx in range(2):
            src = VALID_RMSNORM_NP if g_idx == 0 else BAD_RMSNORM_NP
            rows.append(
                {
                    "run_id": f"step{step:05d}-g{g_idx}",
                    "op_name": op_name,
                    "step": step,
                    "group_index": g_idx,
                    "seed": cfg.seed + step,
                    "policy_updated": False,
                    "optimizer_step_count": 0,
                    "config_hash": cfg.config_hash(),
                    "model_name_or_path": cfg.model_name_or_path,
                    "model_revision": cfg.model_revision,
                    "training_mode": cfg.training_mode,
                    "prompt_token_ids": prompt_ids,
                    "completion_token_ids": [4, 5, 2],
                    "old_token_logps": list(olp),
                    "reference_token_logps": list(rlp),
                    "raw_completion": src,
                    "candidate_source": src,
                    "candidate_sha256": sha256_text(src),
                }
            )
    return rows


class TestRecoverInitialBatch(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "PyTorch required")
    def test_probability_verification_uses_no_grad_and_finite_tolerance(self):
        from rl_trainer.gpu_train_grpo import SampledCandidate
        sample = SampledCandidate([1], [2, 3], "", "", [-1., -1.], [-1., -1.])
        def score(model, p, c):
            self.assertFalse(torch.is_grad_enabled())
            return -torch.ones(len(c), requires_grad=True) * 1
        verify_stored_logps_against_policy(policy=object(), samples=[sample], logp_fn=score,
                                          verify_ref_with_disable_adapter=False)
        for tol in (float("inf"), float("nan"), 0, -1):
            with self.assertRaises(ValueError):
                verify_stored_logps_against_policy(policy=object(), samples=[sample], logp_fn=score,
                                                  logp_verify_atol=tol)

    def test_missing_identity_fields_and_wrong_training_cursor_are_rejected(self):
        cfg = ExperimentConfig.preset_t4_7b_qlora(backend="numpy_dev", dtype="fp32")
        rows = _make_8group_16sample_rows(cfg)
        for field in ("config_hash", "model_revision", "model_name_or_path", "candidate_sha256", "optimizer_step_count"):
            bad = [dict(r) for r in rows]
            bad[0].pop(field)
            with self.assertRaises(ValueError):
                validate_initial_batch_traces(bad, cfg)
        bad = [dict(r) for r in rows]
        bad[0]["op_name"] = "rmsnorm_4096"
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(bad, cfg)

    def test_validation_rejects_updated_mismatched_duplicate_nonfinite_and_corrupt_rows(self):
        cfg = ExperimentConfig.preset_t4_7b_qlora(
            model_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
            backend="numpy_dev",
            dtype="fp32",
        )
        base_rows = _make_8group_16sample_rows(cfg)
        groups = validate_initial_batch_traces(base_rows, cfg)
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(len(g) == 2 for g in groups))

        # 1. policy_updated=True rejected
        bad = [dict(r) for r in base_rows]
        bad[3]["policy_updated"] = True
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(bad, cfg)

        # 2. optimizer_step_count > 0 rejected
        bad = [dict(r) for r in base_rows]
        bad[0]["optimizer_step_count"] = 1
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(bad, cfg)

        # 3. Mismatched config_hash / revision / model rejected
        for field, val in [("config_hash", "deadbeef"), ("model_revision", "wrong_rev"), ("model_name_or_path", "other/model")]:
            bad = [dict(r) for r in base_rows]
            bad[1][field] = val
            with self.assertRaises(ValueError):
                validate_initial_batch_traces(bad, cfg)

        # 4. Duplicate run_id rejected
        bad = [dict(r) for r in base_rows]
        bad[1]["run_id"] = bad[0]["run_id"]
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(bad, cfg)

        # 5. Source SHA-256 mismatch rejected
        bad = [dict(r) for r in base_rows]
        bad[0]["candidate_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(bad, cfg)

        # 6. Non-finite logp or mismatched length rejected
        bad = [dict(r) for r in base_rows]
        bad[0]["old_token_logps"] = [-1.0, float("nan"), -0.5]
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(bad, cfg)

        bad = [dict(r) for r in base_rows]
        bad[0]["old_token_logps"] = [-1.0]
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(bad, cfg)

        # 7. Incomplete group rejected
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(base_rows[:-1], cfg)

        # 8. Mismatched prompt within same group rejected
        bad = [dict(r) for r in base_rows]
        bad[1]["prompt_token_ids"] = [9, 9, 9]
        with self.assertRaises(ValueError):
            validate_initial_batch_traces(bad, cfg)

    def test_stale_policy_rejected_before_any_update(self):
        from rl_trainer.gpu_train_grpo import SampledCandidate

        smp = SampledCandidate([1, 2], [4, 5, 2], "x", "x", [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0])

        class _DummyPolicy:
            pass

        # Matching policy passes
        old_err, _ = verify_stored_logps_against_policy(
            policy=_DummyPolicy(),
            samples=[smp],
            logp_fn=lambda m, p, c: [-1.001, -0.999, -1.0],
            logp_verify_atol=5e-2,
            verify_ref_with_disable_adapter=False,
        )
        self.assertLess(old_err, 5e-2)

        # Stale/drifted policy raises ValueError
        with self.assertRaises(ValueError):
            verify_stored_logps_against_policy(
                policy=_DummyPolicy(),
                samples=[smp],
                logp_fn=lambda m, p, c: [-1.5, -1.0, -1.0],
                logp_verify_atol=5e-2,
                verify_ref_with_disable_adapter=False,
            )

    def test_refuses_non_empty_out_dir(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            out_dir = td_p / "out"
            out_dir.mkdir()
            (out_dir / "existing.txt").write_text("keep", encoding="utf-8")
            cfg_path = td_p / "config.json"
            in_jsonl = td_p / "train.jsonl"
            cfg = ExperimentConfig(backend="numpy_dev", dtype="fp32", train_ops=["rmsnorm_2048"], heldout_ops=["rmsnorm_4096"])
            cfg.to_json(cfg_path)
            in_jsonl.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                recover_initial_batch(input_jsonl=in_jsonl, config_path=cfg_path, out_dir=out_dir)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in this sandbox; runs automatically on Torch CPU/GPU hosts.")
    def test_real_tiny_torch_8group_16sample_single_step_recovery(self):
        torch.manual_seed(42)
        vocab, dim = 32, 16

        class _TinyPeftLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab, dim)
                self.base = nn.Linear(dim, vocab, bias=False)
                self.lora_a = nn.Parameter(torch.randn(dim, 4) * 0.05)
                self.lora_b = nn.Parameter(torch.zeros(4, vocab))
                self.use_adapter = True
                self.config = SimpleNamespace(use_cache=True)

            def forward(self, input_ids, logits_to_keep=None):
                h = self.emb(input_ids)
                if logits_to_keep is not None:
                    h = h[:, -logits_to_keep:, :]
                out = self.base(h)
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

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            cfg = ExperimentConfig.preset_t4_7b_qlora(
                model_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
                backend="numpy_dev",
                dtype="fp32",
                train_ops=["rmsnorm_2048"],
                heldout_ops=["rmsnorm_4096"],
                learning_rate=1e-2,
            )
            cfg_path = td_p / "config.json"
            cfg.to_json(cfg_path)

            # Build initial model and record exact initial old_token_logps + reference_token_logps
            torch.manual_seed(cfg.seed)
            init_model = _TinyPeftLM()
            for p in init_model.emb.parameters():
                p.requires_grad_(False)
            for p in init_model.base.parameters():
                p.requires_grad_(False)

            rows = _make_8group_16sample_rows(cfg)
            with torch.no_grad():
                for r in rows:
                    lp = compute_selected_token_logps(init_model, r["prompt_token_ids"], r["completion_token_ids"], "cpu").tolist()
                    r["old_token_logps"] = lp
                    r["reference_token_logps"] = lp

            in_jsonl = td_p / "train.jsonl"
            in_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            out_dir = td_p / "recovered"

            def _loader(c):
                torch.manual_seed(c.seed)
                m = _TinyPeftLM()
                for p in m.emb.parameters():
                    p.requires_grad_(False)
                for p in m.base.parameters():
                    p.requires_grad_(False)
                tr = [m.lora_a, m.lora_b]
                opt = torch.optim.AdamW(tr, lr=c.learning_rate)
                return m, tr, opt, (lambda mod, p_ids, c_ids: compute_selected_token_logps(mod, p_ids, c_ids, "cpu")), "cpu"

            manifest = recover_initial_batch(
                input_jsonl=in_jsonl,
                config_path=cfg_path,
                out_dir=out_dir,
                model_loader_fn=_loader,
                is_fixture=True,
            )
            self.assertEqual(manifest["method"], RECOVERY_METHOD)
            self.assertEqual(manifest["recovery_schedule"], RECOVERY_SCHEDULE)
            self.assertEqual(manifest["num_groups"], 8)
            self.assertEqual(manifest["total_rollouts"], 16)
            self.assertTrue(manifest["policy_updated"])
            self.assertEqual(manifest["optimizer_step_count"], 1)
            self.assertGreater(manifest["weight_l2_delta"], 0.0)
            self.assertEqual(len(manifest["manifest_sha256"]), 64)
            self.assertTrue((out_dir / "config.json").exists())
            self.assertTrue((out_dir / "checkpoints" / "step_00008" / "checkpoint_state.json").exists())


if __name__ == "__main__":
    unittest.main()
