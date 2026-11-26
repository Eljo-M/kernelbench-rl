import json
import importlib.util
import re
from pathlib import Path
import tempfile
import unittest

from rl_trainer.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer.evaluate_heldout import evaluate_heldout_methods
from rl_trainer.gpu_train_grpo import SampledCandidate, perform_verified_grpo_step, rollout_and_grade_on_policy, run_gpu_training_loop
from rl_trainer.prompting import build_prompt


class RetryProtocolTests(unittest.TestCase):
    def test_old_run_hashes_preserved_and_active_protocol_hashed(self):
        root = Path(__file__).resolve().parents[1]
        for path, expected in (("laptop_fullparam", "b06d4b72437f6cb2"),
                               ("colab_t4/qlora7b_fresh", "5ec1b65de79cdea6")):
            cfg = ExperimentConfig.from_json(root/"experiments"/path/"config.json")
            self.assertEqual(cfg.config_hash(), expected)
            cfg.prompt_version = "contract_v2"
            changed = cfg.config_hash()
            self.assertNotEqual(changed, expected)
            cfg.sft_epochs = 4
            self.assertNotEqual(cfg.config_hash(), changed)
            changed = cfg.config_hash()
            cfg.training_reward_mode = "correctness_gated"
            self.assertNotEqual(cfg.config_hash(), changed)

    def test_prompt_states_dtype_host_device_and_enforced_policy(self):
        prompt = build_prompt("int8_weight_only_gemm_512", prompt_version="contract_v2", float_dtype="fp16")
        self.assertIn("torch.float16", prompt)
        self.assertIn("KEEP INTEGER", prompt)
        self.assertIn("ordinary Python", prompt)
        self.assertIn("No NumPy imports", prompt)
        self.assertIn("import triton.language as tl", prompt)
        self.assertIn("do not pass .data_ptr()", prompt)
        # No handwritten solution for the requested operation is embedded.
        self.assertEqual(re.findall(r"@triton.jit\ndef (\w+)", prompt), ["_copy"])

    def test_numpy_specification_includes_recursive_mathematical_helpers(self):
        silu = build_prompt("fused_bias_silu_4096", prompt_version="contract_v2")
        self.assertIn("def _stable_sigmoid", silu)
        attention = build_prompt("scaled_dot_product_attn_128", prompt_version="contract_v2")
        self.assertIn("def _ref_causal_masked_softmax", attention)
        self.assertLess(attention.index("def _ref_causal_masked_softmax"), attention.index("def _ref_sdpa"))

    def test_all_invalid_group_cannot_update_from_failure_class_differences(self):
        cfg = ExperimentConfig(training_reward_mode="correctness_gated")
        def sample(prompt, n, seed):
            return [SampledCandidate([1], [2], str(i), str(i)) for i in range(n)]
        def fail(op, source, **kwargs):
            return dict(status="CHEAT" if source == "0" else "COMPILE_ERROR",
                        compiled=False, correct=False, cheat_detected=source == "0",
                        reward=-1.0 if source == "0" else 0.0)
        rows, _, advantages = rollout_and_grade_on_policy("rmsnorm_2048", cfg, step=1, sample_fn=sample, grader_fn=fail)
        self.assertEqual(advantages, [0.0, 0.0])
        self.assertEqual([r["reward"] for r in rows], [0.0, 0.0])
        self.assertEqual([r["verifier_reward"] for r in rows], [-1.0, 0.0])
        def one_pass(op, source, **kwargs):
            if source == "1":
                return dict(status="PASS", compiled=True, correct=True, cheat_detected=False, reward=1.1)
            return fail(op, source, **kwargs)
        _, _, advantages = rollout_and_grade_on_policy("rmsnorm_2048", cfg, step=1, sample_fn=sample, grader_fn=one_pass)
        self.assertGreater(advantages[1], 0)
        self.assertLess(advantages[0], 0)

    def test_inspected_regressions_never_enter_fresh_denominator(self):
        cfg = ExperimentConfig(backend="numpy_dev", heldout_ops=["l2_normalize_2048"],
            development_eval_ops=["rmsnorm_4096"], heldout_best_of_k=2)
        def grader(op, source, **kwargs):
            return dict(op_name=op, status="PASS" if op == "rmsnorm_4096" else "COMPILE_ERROR",
                compiled=op == "rmsnorm_4096", correct=op == "rmsnorm_4096",
                reward=1.0 if op == "rmsnorm_4096" else 0.0,
                method=kwargs["method"], speedup_vs_eager=0, speedup_vs_compile=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"rows.jsonl"
            result = evaluate_heldout_methods(cfg, base_generate_fn=lambda p,n,s: ["pass"]*n,
                trained_generate_fn=lambda p,n,s: ["pass"]*n, grader_fn=grader, out_jsonl=path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        for method, summary in result.items():
            self.assertEqual(summary["ops"], 1)
            self.assertEqual(summary["pass_rate"], 0)
        dev = [r for r in rows if r["op_name"] == "rmsnorm_4096"]
        self.assertEqual(len(dev), 2)  # No best-of-K search on inspected regressions.
        self.assertTrue(all(r["eval_phase"] == "development_test" and not r["heldout"] for r in dev))
        self.assertTrue(all(r["op_name"] == "l2_normalize_2048" for r in rows if r["eval_phase"] == "heldout_test"))

    def test_sft_checkpoint_resume_preserves_phase_count_without_repeating_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = ExperimentConfig(backend="numpy_dev", sft_epochs=4, max_steps=8,
                checkpoint_dir=str(root/"checkpoint"), output_jsonl=str(root/"train.jsonl"))
            state = CheckpointState(8,8,42,cfg.config_hash(),0,16,sft_optimizer_steps=32,is_fixture=True)
            save_checkpoint(Path(cfg.checkpoint_dir), state, save_rng=False)
            loaded = load_checkpoint(cfg.checkpoint_dir, restore_rng=False)
            self.assertEqual(loaded.sft_optimizer_steps, 32)
            cfg.resume_from = cfg.checkpoint_dir
            def forbidden(*args):
                self.fail("A completed resume must not sample or repeat warmup")
            state = run_gpu_training_loop(cfg, sample_fn=forbidden, is_fixture=True)
            self.assertEqual(state.sft_optimizer_steps, 32)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is optional")
    def test_logp_guard_aborts_before_optimizer_mutation(self):
        import torch
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        model = torch.nn.ParameterList([parameter])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        sample = SampledCandidate([1], [2], "", "", old_token_logps=[-1.0], ref_token_logps=[-1.0])
        with self.assertRaisesRegex(ValueError, "mismatch before update"):
            perform_verified_grpo_step(policy=model, trainable_params=list(model.parameters()),
                optimizer=optimizer, samples=[sample], advantages=[1.0],
                logp_fn=lambda m,p,c: parameter.expand(1), clip_eps=0.2, kl_coef=0.04,
                max_grad_norm=1.0, device="cpu", old_logp_tolerance=0.001)
        self.assertEqual(parameter.item(), 0.0)
        self.assertFalse(optimizer.state)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is optional")
    def test_overflowing_gradient_norm_never_runs_an_optimizer_step(self):
        import torch
        parameter = torch.nn.Parameter(torch.zeros(2))
        model = torch.nn.ParameterList([parameter])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        sample = SampledCandidate([1], [2,3], "", "", old_token_logps=[0.,0.], ref_token_logps=[0.,0.])
        result = perform_verified_grpo_step(policy=model, trainable_params=list(model.parameters()),
            optimizer=optimizer, samples=[sample], advantages=[1.0],
            logp_fn=lambda m,p,c: parameter * 1e30, clip_eps=0.2, kl_coef=0.04,
            max_grad_norm=1.0, device="cpu", old_logp_tolerance=0.001)
        self.assertFalse(result["policy_updated"])
        self.assertTrue(result["skipped_non_finite_grad"])
        self.assertFalse(optimizer.state)
        self.assertTrue(torch.all(parameter == 0))
        from rl_trainer.cpu_master_adamw import CpuMasterAdamW
        master = CpuMasterAdamW(model.parameters(), lr=0.1)
        parameter.grad = torch.full_like(parameter, 1e30)
        updated, delta = master.copy_grads_and_step(1.0)
        self.assertFalse(updated)
        self.assertEqual(delta, 0.)
        self.assertFalse(master.optimizer.state)


if __name__ == "__main__":
    unittest.main()
