"""Synthetic production-shaped fixtures live only inside temporary test directories."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from eval.compare_heldout import HeldoutValidationError, load_and_validate_run
from eval.compare_retry import compare_retry_runs, load_retry_run
from rl_trainer.experiment_config import ExperimentConfig
from kernels.op_registry import OP_REGISTRY

FRESH = ["l2_normalize_2048", "fused_bias_silu_4096", "matmul_residual_add_512"]
DEVELOPMENT = ["rmsnorm_4096", "int8_weight_only_gemm_512", "scaled_dot_product_attn_128"]
METHODS = ("base_pass1", "base_best_of_8", "trained_checkpoint")
SFT = "sft_only_checkpoint"


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_fixture(name="laptop", *, model="Test/SmallCoder", gpu="Test GPU A", dtype="fp32", mode="full_param_0.49b"):
    cfg = ExperimentConfig(model_name_or_path=model, model_revision="a" * 40,
        dtype=dtype, training_mode=mode, seed=142, heldout_ops=FRESH,
        development_eval_ops=DEVELOPMENT, prompt_version="contract_v2",
        training_reward_mode="correctness_gated", sft_epochs=4, max_steps=8)
    cfg.validate()
    main_env = dict(run_name=name, model_id=model, model_revision=cfg.model_revision,
        training_mode=mode, backend="triton", dtype=dtype, gpu=gpu,
        config_hash=cfg.config_hash(), source_commit="c" * 40,
        checkpoint_step=8, checkpoint_weights_sha256="d" * 64,
        sft_optimizer_steps=32, optimizer_steps=4, generation_engine="hf_eager")
    sft_env = {**main_env, "checkpoint_step": 0, "checkpoint_weights_sha256": "e" * 64,
               "checkpoint_identity": "checkpoint:0:" + "e" * 64}
    protocol = dict(registered_utc="2026-11-26T00:00:00+00:00", fresh_heldout_ops=FRESH,
        development_regressions=DEVELOPMENT, config_hashes={name: cfg.config_hash()},
        run_schedules={name: {"max_steps": cfg.max_steps, "sft_epochs": cfg.sft_epochs, "group_size": cfg.group_size}},
        inference_budgets=dict(base_pass1=1, base_best_of_8_fresh_only=8,
                              sft_only_checkpoint=1, trained_checkpoint=1))

    def row(method, op, sample=0, selection=False):
        index = (FRESH + DEVELOPMENT).index(op)
        dev = op in DEVELOPMENT
        # Inspected operators PASS while every fresh operator fails. The
        # deliberate contrast catches denominator contamination.
        passed = dev
        seed = cfg.seed + cfg.heldout_seed_offset + index
        if selection:
            seed += cfg.selection_seed_offset
        source = f"def kernel_forward(x):\n    return x\n# {op} sample {sample}"
        identity = ("checkpoint:0:" + "e" * 64 if method == SFT else
                    "checkpoint:8:" + "d" * 64 if method == "trained_checkpoint" else
                    "base:" + cfg.model_revision)
        return dict(run_id=f"{method}-{op}-{'selection' if selection else 'test'}-{sample}",
            method=method, op_name=op, level=OP_REGISTRY[op].level, backend="triton", status="PASS" if passed else "COMPILE_ERROR",
            compiled=passed, correct=passed, cheat_detected=False, max_abs_err=0,
            eager_ms=2.0 if passed else 0, compile_ms=1.0 if passed else 0,
            candidate_ms=1.0 if passed else 0, speedup_vs_eager=2.0 if passed else 0,
            speedup_vs_compile=1.0 if passed else 0, compile_available=passed,
            reward=1.0 if passed else 0, seed=seed, independent_seed=seed+cfg.independent_seed_offset,
            dtype=dtype, gpu_model=gpu, model_name_or_path=model, model_revision=cfg.model_revision,
            training_mode=mode, config_hash=cfg.config_hash(), eval_phase="selection_val" if selection else "development_test" if dev else "heldout_test",
            split="development" if dev else "heldout", heldout=not dev,
            evaluation_cohort="previously_inspected_development" if dev else "fresh_heldout",
            selection_method="validation_seed_regrade" if method == "base_best_of_8" else "single_sample",
            sample_k=sample, budget_k=8 if method == "base_best_of_8" else 1,
            policy_updated=False, fixture=False, candidate_source=source, candidate_sha256=_sha(source),
            prompt_sha256=_sha("same frozen prompt " + op), sampling_config=cfg.sampling_config_dict(),
            generation_engine="hf_eager", checkpoint_identity=identity)

    main_rows = [row(method, op) for method in ("base_pass1", "trained_checkpoint") for op in FRESH + DEVELOPMENT]
    for op in FRESH:
        main_rows.extend(row("base_best_of_8", op, sample=k, selection=True) for k in range(8))
        main_rows.append(row("base_best_of_8", op))
    sft_rows = [row(SFT, op) for op in FRESH + DEVELOPMENT]
    return dict(cfg=asdict(cfg), main_env=main_env, sft_env=sft_env, protocol=protocol,
                main_rows=main_rows, sft_rows=sft_rows)


def write_fixture(root, fixture, name="laptop"):
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    for filename, value in (("config.json", fixture["cfg"]), ("heldout_environment.json", fixture["main_env"]),
                            ("sft_only_environment.json", fixture["sft_env"])):
        (run / filename).write_text(json.dumps(value), encoding="utf-8")
    for filename, rows in (("heldout.jsonl", fixture["main_rows"]), ("sft_only.jsonl", fixture["sft_rows"])):
        (run / filename).write_text("\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n", encoding="utf-8")
    (root / "protocol.json").write_text(json.dumps(fixture["protocol"]), encoding="utf-8")
    return run


class TestRetryComparison(unittest.TestCase):
    def test_existing_comparator_excludes_all_six_passing_development_rows(self):
        with tempfile.TemporaryDirectory() as td:
            run = write_fixture(Path(td), make_fixture())
            report = load_and_validate_run(run)
            self.assertEqual(report["excluded_row_counts"]["selection_val"], 24)
            self.assertEqual(report["excluded_row_counts"]["other_eval_phase"], 6)
            for method in METHODS:
                self.assertEqual(report["methods"][method]["total_ops"], 3)
                self.assertEqual(report["methods"][method]["pass_count"], 0)

    def test_full_retry_inventory_keeps_sft_and_development_separate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = write_fixture(root, make_fixture())
            report = compare_retry_runs([run], root / "report")
            result = report["runs"][0]
            self.assertFalse(report["pooled_across_runs"])
            self.assertFalse(report["rl_only_causal_claim"])
            self.assertFalse(report["historical_paired_same_protocol"])
            self.assertEqual(result["inventory"], dict(main_rows=39, main_selection_rows=24,
                main_fresh_final_rows=9, main_development_final_rows=6, sft_only_rows=6,
                sft_fresh_final_rows=3, sft_development_final_rows=3))
            for method in METHODS + (SFT,):
                self.assertEqual(result["fresh_heldout"]["methods"][method]["pass_count"], 0)
                self.assertEqual(result["fresh_heldout"]["methods"][method]["total_ops"], 3)
            for method in ("base_pass1", "trained_checkpoint", SFT):
                self.assertEqual(result["inspected_development"]["methods"][method]["pass_count"], 3)
            self.assertFalse(result["inspected_development"]["methods"]["base_best_of_8"]["available"])
            markdown = (root / "report" / "retry_comparison.md").read_text(encoding="utf-8")
            self.assertIn("cannot be attributed to RL alone", markdown)
            self.assertIn("different prompts", markdown)
            self.assertIn("unavailable (0 correct)", markdown)
            json.loads((root / "report" / "retry_comparison.json").read_text())

    def test_every_fresh_method_operator_identity_and_seed_is_required(self):
        original = make_fixture()
        indices = [i for i, row in enumerate(original["main_rows"]) if row["eval_phase"] == "heldout_test"]
        self.assertEqual(len(indices), 9)
        with tempfile.TemporaryDirectory() as td:
            for index in indices:
                with self.subTest(index=index):
                    broken = deepcopy(original)
                    broken["main_rows"][index]["seed"] += 1
                    run = write_fixture(Path(td), broken)
                    with self.assertRaisesRegex(HeldoutValidationError, "[Ss]eed mismatch"):
                        load_retry_run(run)
            broken = deepcopy(original)
            del broken["main_rows"][indices[0]]
            with self.assertRaisesRegex(HeldoutValidationError, "Incomplete"):
                load_retry_run(write_fixture(Path(td), broken))
            broken = deepcopy(original)
            broken["main_rows"].append(deepcopy(broken["main_rows"][indices[0]]))
            with self.assertRaisesRegex(HeldoutValidationError, "Duplicate"):
                load_retry_run(write_fixture(Path(td), broken))

    def test_sft_only_and_hybrid_success_counts_are_independent(self):
        fixture = make_fixture()
        passing_rows = fixture["sft_rows"][:2] + [fixture["main_rows"][6]]
        for row in passing_rows:
            row.update(status="PASS", compiled=True, correct=True, eager_ms=2.0,
                       compile_ms=0.0, compile_available=False, candidate_ms=1.0,
                       speedup_vs_eager=2.0, speedup_vs_compile=0.0, reward=1.0)
        with tempfile.TemporaryDirectory() as td:
            report = load_retry_run(write_fixture(Path(td), fixture))
        fresh = report["fresh_heldout"]["methods"]
        self.assertEqual(fresh["base_pass1"]["pass_count"], 0)
        self.assertEqual(fresh[SFT]["pass_count"], 2)
        self.assertEqual(fresh["trained_checkpoint"]["pass_count"], 1)
        self.assertEqual(fresh[SFT]["geomean_speedup_vs_eager_correct_only"], 2)
        self.assertEqual(fresh[SFT]["fast_vs_compile_denominator"], 1)
        self.assertEqual(fresh[SFT]["compile_unavailable_passing_count"], 2)

    def test_both_final_cohorts_reject_identity_and_checkpoint_mutations(self):
        original = make_fixture()
        with tempfile.TemporaryDirectory() as td:
            for rows_key, index in (("main_rows", 0), ("main_rows", 3), ("sft_rows", 0), ("sft_rows", 3)):
                for field, value in (("config_hash", "wrong"), ("model_name_or_path", "Other/Model"),
                                     ("model_revision", "other-revision"), ("dtype", "bf16"),
                                     ("gpu_model", "Other real GPU"), ("checkpoint_identity", "checkpoint:99:"+"f"*64),
                                     ("independent_seed", 1), ("policy_updated", True),
                                     ("sampling_config", {"temperature": .7})):
                    with self.subTest(rows=rows_key, index=index, field=field):
                        broken = deepcopy(original)
                        broken[rows_key][index][field] = value
                        with self.assertRaises(HeldoutValidationError):
                            load_retry_run(write_fixture(Path(td), broken))

    def test_sft_manifest_matches_main_model_hardware_config_and_step_zero(self):
        original = make_fixture()
        with tempfile.TemporaryDirectory() as td:
            for field, value in (("model_id", "Other/Model"), ("model_revision", "other"),
                                 ("dtype", "fp16"), ("gpu", "Other GPU"), ("config_hash", "wrong"),
                                 ("source_commit", "other"), ("checkpoint_step", 4),
                                 ("checkpoint_weights_sha256", "invalid"), ("sft_optimizer_steps", 0)):
                with self.subTest(field=field):
                    broken = deepcopy(original)
                    broken["sft_env"][field] = value
                    with self.assertRaises(HeldoutValidationError):
                        load_retry_run(write_fixture(Path(td), broken))

    def test_cohort_contradictions_are_errors_not_heldout_promotions(self):
        original = make_fixture()
        with tempfile.TemporaryDirectory() as td:
            for field, value in (("heldout", False), ("heldout", 1), ("split", "development"),
                                 ("evaluation_cohort", "previously_inspected_development"),
                                 ("policy_updated", "false"), ("selection_method", "single_sample"),
                                 ("budget_k", 1), ("sample_k", 8)):
                with self.subTest(field=field):
                    broken = deepcopy(original)
                    index = next(i for i, r in enumerate(broken["main_rows"]) if r["method"] == "base_best_of_8" and r["eval_phase"] == "heldout_test")
                    broken["main_rows"][index][field] = value
                    with self.assertRaises(HeldoutValidationError):
                        load_retry_run(write_fixture(Path(td), broken))

    def test_selection_inventory_and_winner_source_are_verified(self):
        original = make_fixture()
        with tempfile.TemporaryDirectory() as td:
            for change in ("missing", "duplicate", "different_source", "not_maximal", "prompt_changed"):
                with self.subTest(change=change):
                    broken = deepcopy(original)
                    first = next(i for i, r in enumerate(broken["main_rows"]) if r["eval_phase"] == "selection_val")
                    if change == "missing":
                        del broken["main_rows"][first]
                    elif change == "duplicate":
                        duplicate = deepcopy(broken["main_rows"][first]); duplicate["run_id"] += "other"
                        broken["main_rows"].append(duplicate)
                    elif change == "different_source":
                        broken["main_rows"][first]["candidate_source"] = "different"
                        broken["main_rows"][first]["candidate_sha256"] = _sha("different")
                    elif change == "not_maximal":
                        broken["main_rows"][first+1]["reward"] = .1
                    else:
                        broken["main_rows"][first]["prompt_sha256"] = "f" * 64
                    with self.assertRaises(HeldoutValidationError):
                        load_retry_run(write_fixture(Path(td), broken))

    def test_sft_file_requires_all_six_and_only_its_registered_method(self):
        original = make_fixture()
        with tempfile.TemporaryDirectory() as td:
            for change in ("missing", "method", "fixture", "phase", "seed"):
                broken = deepcopy(original)
                if change == "missing":
                    broken["sft_rows"].pop()
                elif change == "method":
                    broken["sft_rows"][0]["method"] = "trained_checkpoint"
                elif change == "fixture":
                    broken["sft_rows"][0]["fixture"] = True
                elif change == "phase":
                    broken["sft_rows"][3]["eval_phase"] = "heldout_test"
                else:
                    broken["sft_rows"][3]["seed"] -= 3
                with self.subTest(change=change), self.assertRaises(HeldoutValidationError):
                    load_retry_run(write_fixture(Path(td), broken))

    def test_preregistered_config_hash_and_inference_budget_must_match(self):
        original = make_fixture()
        with tempfile.TemporaryDirectory() as td:
            for change in ("hash", "budget", "split", "schedule"):
                broken = deepcopy(original)
                if change == "hash":
                    broken["protocol"]["config_hashes"]["laptop"] = "wrong"
                elif change == "budget":
                    broken["protocol"]["inference_budgets"]["sft_only_checkpoint"] = 8
                elif change == "split":
                    broken["protocol"]["fresh_heldout_ops"] = list(reversed(FRESH))
                else:
                    # max_steps intentionally does not affect the resumable
                    # config hash; the frozen schedule must catch this change.
                    broken["cfg"]["max_steps"] = 16
                with self.subTest(change=change), self.assertRaises(HeldoutValidationError):
                    load_retry_run(write_fixture(Path(td), broken))

    def test_empty_generated_source_is_preserved_as_a_failed_attempt(self):
        original = make_fixture()
        original["sft_rows"][0]["candidate_source"] = ""
        original["sft_rows"][0]["candidate_sha256"] = _sha("")
        with tempfile.TemporaryDirectory() as td:
            run = write_fixture(Path(td), original)
            self.assertEqual(load_retry_run(run)["fresh_heldout"]["methods"][SFT]["total_ops"], 3)

    def test_distinct_models_and_hardware_remain_separate_never_pooled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = make_fixture()
            second = make_fixture("t4", model="Test/LargeCoder", gpu="Test GPU B", dtype="fp16", mode="qlora_7b_nf4")
            combined_hashes = {**first["protocol"]["config_hashes"], **second["protocol"]["config_hashes"]}
            first["protocol"]["config_hashes"] = second["protocol"]["config_hashes"] = combined_hashes
            combined_schedules = {**first["protocol"]["run_schedules"], **second["protocol"]["run_schedules"]}
            first["protocol"]["run_schedules"] = second["protocol"]["run_schedules"] = combined_schedules
            run1 = write_fixture(root, first)
            run2 = write_fixture(root, second, "t4")
            report = compare_retry_runs([run1, run2], root/"report")
            self.assertEqual(len(report["runs"]), 2)
            self.assertFalse(report["pooled_across_runs"])
            self.assertNotEqual(report["runs"][0]["environment"]["gpu"], report["runs"][1]["environment"]["gpu"])
            self.assertTrue(all(run["fresh_heldout"]["methods"][SFT]["total_ops"] == 3 for run in report["runs"]))


if __name__ == "__main__":
    unittest.main()
