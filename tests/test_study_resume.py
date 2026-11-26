"""Filesystem study transitions with mocked models/runtime; no hardware jobs."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from rl_trainer.checkpointing import CheckpointState, load_checkpoint, reconcile_resume_log, save_checkpoint
from rl_trainer.experiment_config import ExperimentConfig
from rl_trainer import run_study as study


class StudyResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root/"run"
        self.cfg = ExperimentConfig(max_steps=2, sft_epochs=1,
            checkpoint_dir=str(self.root/"checkpoints"), output_jsonl=str(self.out/"train.jsonl"))
        self.config_path = self.out/"config.json"
        self.cfg.to_json(self.config_path)
        cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "mocked GPU",
            max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
            empty_cache=lambda: None, reset_peak_memory_stats=lambda: None)
        torch = SimpleNamespace(cuda=cuda, __version__="mock", version=SimpleNamespace(cuda="mock"),
                                set_num_threads=lambda _: None)
        self.runtime = SimpleNamespace(torch=torch, triton=SimpleNamespace(__version__="mock"),
            transformers=SimpleNamespace(__version__="mock"), full_train=mock.Mock(side_effect=self.train),
            qlora_train=mock.Mock(side_effect=AssertionError("wrong trainer")),
            evaluate=mock.Mock(side_effect=self.evaluate), generate=mock.Mock(side_effect=self.generate),
            grade=mock.Mock(side_effect=self.grade))
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(study, "_runtime_dependencies", return_value=self.runtime).start()
        self.git = mock.patch.object(study.subprocess, "check_output", side_effect=self.git_output).start()

    @staticmethod
    def git_output(command, **kwargs):
        return "a"*40 if "rev-parse" in command else ""

    def state(self, cfg, step, sha):
        return CheckpointState(step=step, op_cursor=step, seed=cfg.seed, config_hash=cfg.config_hash(),
            best_mean_reward=0, total_rollouts=step*cfg.group_size, optimizer_step_count=0,
            trainable_weights_sha256=sha, sft_optimizer_steps=len(cfg.train_ops))

    def sft_checkpoint(self, cfg):
        target = Path(cfg.checkpoint_dir)/"sft_only"
        if not (target/"latest.json").exists():
            save_checkpoint(target, self.state(cfg, 0, "sft-sha"), save_rng=False)

    def train(self, cfg):
        self.sft_checkpoint(cfg)
        target = Path(cfg.checkpoint_dir)
        if (target/"latest.json").exists():
            return load_checkpoint(target, expected_config_hash=cfg.config_hash(), restore_rng=False)
        if cfg.resume_from:
            self.assertEqual(Path(cfg.resume_from), target/"sft_only")
        state = self.state(cfg, cfg.max_steps, "trained-sha")
        save_checkpoint(target, state, save_rng=False)
        return state

    def evaluate(self, arguments):
        output = Path(arguments[arguments.index("--out") + 1])
        self.assertFalse(output.exists())
        output.write_text('{"method":"base_pass1"}\n', encoding="utf-8")

    def generate(self, cfg, methods, **kwargs):
        cache = kwargs["cache_path"]
        if not cache.exists():
            cache.write_text("saved-generation-cache\n", encoding="utf-8")
        return {"sft_only_checkpoint": []}, True

    def grade(self, cfg, cached, **kwargs):
        self.assertFalse(kwargs["out_jsonl"].exists())
        kwargs["out_jsonl"].write_text('{"method":"sft_only_checkpoint"}\n', encoding="utf-8")
        return {"sft_only_checkpoint": {"pass_rate": 0}}

    def run_study(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            study.run_study(self.config_path, **kwargs)

    def test_opt_in_resume_skips_all_completed_matching_stages(self):
        self.run_study()
        counts = [self.runtime.full_train.call_count, self.runtime.evaluate.call_count,
                  self.runtime.generate.call_count, self.runtime.grade.call_count]
        evidence = {p.name: p.read_bytes() for p in self.out.glob("*.json")}
        with self.assertRaises(FileExistsError):
            self.run_study()
        self.run_study(resume_study=True)
        self.assertEqual(counts, [self.runtime.full_train.call_count, self.runtime.evaluate.call_count,
                                 self.runtime.generate.call_count, self.runtime.grade.call_count])
        self.assertEqual(evidence, {p.name: p.read_bytes() for p in self.out.glob("*.json")})

    def test_resume_refuses_stale_source_config_model_or_revision(self):
        self.run_study()
        path = self.out/"environment.json"
        original = json.loads(path.read_text())
        calls = self.runtime.full_train.call_count
        for key in ("source_commit", "config_hash", "model_id", "model_revision", "training_schedule"):
            stale = dict(original, **{key: "stale"})
            path.write_text(json.dumps(stale), encoding="utf-8")
            evidence = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.run_study(resume_study=True)
            self.assertEqual(path.read_bytes(), evidence)
        self.assertEqual(self.runtime.full_train.call_count, calls)

    def test_interrupted_training_resumes_completed_sft_without_repeating_it(self):
        def interrupted(cfg):
            self.sft_checkpoint(cfg)
            raise RuntimeError("interrupted after durable SFT")
        self.runtime.full_train.side_effect = interrupted
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_study()
        sft_meta = Path(self.cfg.checkpoint_dir)/"sft_only"/"step_00000"/"checkpoint_state.json"
        saved_sft = sft_meta.read_bytes()
        self.runtime.full_train.side_effect = self.train
        self.run_study(resume_study=True)
        resumed_cfg = self.runtime.full_train.call_args_list[1].args[0]
        # The orchestrator uses the same mutable config for the final no-op resume
        # proof; the durable SFT itself must remain byte-for-byte untouched.
        self.assertEqual(sft_meta.read_bytes(), saved_sft)
        self.assertEqual(resumed_cfg.sft_epochs, 1)
        previous = list(self.out.glob("environment.json.attempt-*"))
        self.assertEqual(len(previous), 1)
        self.assertIn("interrupted after durable SFT", previous[0].read_text())

    def test_interrupted_sft_without_checkpoint_cannot_silently_restart(self):
        self.runtime.full_train.side_effect = RuntimeError("SFT interrupted before checkpoint")
        with self.assertRaises(RuntimeError):
            self.run_study()
        evidence = (self.out/"environment.json").read_bytes()
        self.runtime.full_train.reset_mock()
        with self.assertRaisesRegex(ValueError, "No durable checkpoint"):
            self.run_study(resume_study=True)
        self.runtime.full_train.assert_not_called()
        self.assertEqual((self.out/"environment.json").read_bytes(), evidence)

    def test_interrupted_grpo_uses_main_checkpoint_and_preserves_partial_log(self):
        def interrupted(cfg):
            self.sft_checkpoint(cfg)
            save_checkpoint(Path(cfg.checkpoint_dir), self.state(cfg, 1, "step-one"), save_rng=False)
            Path(cfg.output_jsonl).write_text('{"step":1}\n{"step":2}\n', encoding="utf-8")
            raise RuntimeError("interrupted GRPO")
        self.runtime.full_train.side_effect = interrupted
        with self.assertRaises(RuntimeError):
            self.run_study()

        def resumed(cfg):
            self.assertEqual(Path(cfg.resume_from), Path(cfg.checkpoint_dir))
            state = load_checkpoint(cfg.resume_from, expected_config_hash=cfg.config_hash(), restore_rng=False)
            if state.step < cfg.max_steps:
                reconcile_resume_log(Path(cfg.output_jsonl), state.step)
                state = self.state(cfg, cfg.max_steps, "final-after-resume")
                save_checkpoint(Path(cfg.checkpoint_dir), state, save_rng=False)
            return state
        self.runtime.full_train.side_effect = resumed
        self.run_study(resume_study=True)
        self.assertEqual(Path(self.cfg.output_jsonl).read_text(), '{"step":1}\n')
        archives = list(self.out.glob("train.jsonl.uncheckpointed-*"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), '{"step":1}\n{"step":2}\n')

    def test_interrupted_sft_control_resumes_its_cache_without_retraining(self):
        def interrupted(cfg, cached, **kwargs):
            kwargs["out_jsonl"].write_text("partial-control-output\n", encoding="utf-8")
            raise RuntimeError("SFT control interrupted")
        self.runtime.grade.side_effect = interrupted
        with self.assertRaises(RuntimeError):
            self.run_study()
        train_calls = self.runtime.full_train.call_count
        cache = self.out/"sft_only.generations.jsonl"
        original_cache = cache.read_bytes()
        self.runtime.grade.side_effect = self.grade
        self.run_study(resume_study=True)
        self.assertEqual(self.runtime.full_train.call_count, train_calls)
        self.runtime.evaluate.assert_called_once()
        self.assertEqual(cache.read_bytes(), original_cache)
        archives = list(self.out.glob("sft_only.jsonl.attempt-*"))
        self.assertEqual(archives[0].read_text(), "partial-control-output\n")

    def test_interrupted_evaluation_archives_output_and_keeps_generation_cache(self):
        cache = self.out/"heldout.base.generations.jsonl"
        def interrupted(arguments):
            cache.write_text("cached-completion\n", encoding="utf-8")
            (self.out/"heldout.jsonl").write_text("partial-prior-output\n", encoding="utf-8")
            raise RuntimeError("evaluation interrupted")
        self.runtime.evaluate.side_effect = interrupted
        with self.assertRaises(RuntimeError):
            self.run_study()
        train_calls = self.runtime.full_train.call_count
        self.runtime.evaluate.side_effect = self.evaluate
        self.run_study(resume_study=True)
        self.assertEqual(self.runtime.full_train.call_count, train_calls)
        self.assertEqual(cache.read_text(), "cached-completion\n")
        output_history = list(self.out.glob("heldout.jsonl.attempt-*"))
        self.assertEqual(len(output_history), 1)
        self.assertEqual(output_history[0].read_text(), "partial-prior-output\n")
        manifest_history = list(self.out.glob("heldout_environment.json.attempt-*"))
        self.assertIn("evaluation interrupted", manifest_history[0].read_text())

    def test_completed_output_and_checkpoint_counts_are_identity_checked(self):
        self.run_study()
        (self.out/"heldout.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output changed"):
            self.run_study(resume_study=True)
        self.runtime.evaluate.assert_called_once()
        env_path = self.out/"environment.json"
        env = json.loads(env_path.read_text())
        env["sft_optimizer_steps"] += 1
        env_path.write_text(json.dumps(env), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "completed training checkpoint"):
            self.run_study(resume_study=True)

    def test_resume_without_manifest_and_dirty_source_refuse_before_writes(self):
        with self.assertRaises(FileNotFoundError):
            self.run_study(resume_study=True)
        self.git.side_effect = lambda command, **kwargs: "dirty.py" if "diff" in command else self.git_output(command)
        with self.assertRaisesRegex(RuntimeError, "Commit reviewed source"):
            self.run_study()
        self.assertFalse((self.out/"environment.json").exists())


if __name__ == "__main__":
    unittest.main()
