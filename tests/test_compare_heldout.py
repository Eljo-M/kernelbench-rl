"""Unit and regression tests for eval/compare_heldout.py.

All synthetic fixtures in this file are unit test data only.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from eval.compare_heldout import HeldoutValidationError, compare_runs, main
from rl_trainer.experiment_config import ExperimentConfig

DEFAULT_HELDOUT_OPS = [
    "rmsnorm_4096",
    "int8_weight_only_gemm_512",
    "scaled_dot_product_attn_128",
]

DEFAULT_TRAIN_OPS = [
    "rmsnorm_2048",
    "layernorm_2048",
    "swiglu_4096",
]


def _make_config(
    *,
    model_id: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    revision: str = "rev049b",
    training_mode: str = "full_param_0.49b",
    dtype: str = "fp32",
    seed: int = 42,
    heldout_seed_offset: int = 50_000,
    independent_seed_offset: int = 10_000,
    heldout_best_of_k: int = 8,
) -> dict[str, Any]:
    return {
        "model_name_or_path": model_id,
        "model_revision": revision,
        "training_mode": training_mode,
        "backend": "triton",
        "dtype": dtype,
        "train_ops": list(DEFAULT_TRAIN_OPS),
        "heldout_ops": list(DEFAULT_HELDOUT_OPS),
        "seed": seed,
        "heldout_seed_offset": heldout_seed_offset,
        "selection_seed_offset": 100_000,
        "independent_seed_offset": independent_seed_offset,
        "heldout_best_of_k": heldout_best_of_k,
    }


def _make_env(
    cfg: dict[str, Any],
    *,
    gpu: str = "NVIDIA GeForce RTX 5060 Laptop GPU",
    source_commit: str = "d9ed88bc2be4053608add3073cec91316db79770",
    checkpoint_weights_sha: str = "sha256_weights_049b",
    run_name: str = "laptop_049b_fp32",
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "model_id": cfg["model_name_or_path"],
        "revision": cfg["model_revision"],
        "training_mode": cfg["training_mode"],
        "dtype": cfg["dtype"],
        "gpu": gpu,
        "config_hash": ExperimentConfig(**cfg).config_hash(),
        "source_commit": source_commit,
        "checkpoint_weights_sha": checkpoint_weights_sha,
    }


def _make_row(
    cfg: dict[str, Any],
    env: dict[str, Any],
    *,
    method: str,
    op_idx: int,
    status: str = "PASS",
    speedup_vs_eager: float = 1.25,
    speedup_vs_compile: float = 1.05,
    compile_ms: float = 0.8,
    compile_available: bool = True,
    gpu_model: str | None = None,
    eval_phase: str = "heldout_test",
    selection_method: str | None = None,
    fixture: bool = False,
    sample_k: int = 0,
    run_id_suffix: str = "",
) -> dict[str, Any]:
    op_name = cfg["heldout_ops"][op_idx]
    seed = cfg["seed"] + cfg["heldout_seed_offset"] + op_idx
    if eval_phase == "selection_val":
        seed += cfg["selection_seed_offset"]
    is_pass = status == "PASS"
    sel = selection_method
    if sel is None:
        sel = "validation_seed_regrade" if method.startswith("base_best_of_") else "single_sample"
    return {
        "run_id": f"{method}-{op_name}-{eval_phase}-k{sample_k}{run_id_suffix}",
        "method": method,
        "op_name": op_name,
        "level": 1,
        "backend": cfg["backend"],
        "status": status,
        "compiled": is_pass or status == "NUMERICAL_MISMATCH",
        "correct": is_pass,
        "cheat_detected": status == "CHEAT",
        "max_abs_err": 0.0 if is_pass else 1.0,
        "eager_ms": 1.0 if is_pass else 0.0,
        "compile_ms": speedup_vs_compile / speedup_vs_eager if (is_pass and compile_available) else 0.0,
        "compile_available": compile_available if is_pass else False,
        "candidate_ms": (1.0 / speedup_vs_eager) if (is_pass and speedup_vs_eager > 0) else 0.0,
        "speedup_vs_eager": speedup_vs_eager if is_pass else 0.0,
        "speedup_vs_compile": speedup_vs_compile if (is_pass and compile_available) else 0.0,
        "reward": 1.5 if is_pass else -1.0,
        "seed": seed,
        "independent_seed": seed + cfg["independent_seed_offset"],
        "dtype": cfg["dtype"],
        "gpu_model": env["gpu"] if gpu_model is None else gpu_model,
        "model_name_or_path": cfg["model_name_or_path"],
        "model_revision": cfg["model_revision"],
        "training_mode": cfg["training_mode"],
        "config_hash": env["config_hash"],
        "eval_phase": eval_phase,
        "selection_method": sel,
        "sample_k": sample_k,
        "budget_k": cfg["heldout_best_of_k"] if method.startswith("base_best_of_") else 1,
        "policy_updated": False,
        "fixture": fixture,
    }


def _write_run_dir(run_dir: Path, cfg: dict[str, Any], env: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (run_dir / "heldout_environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "heldout.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, allow_nan=False) + "\n")


class TestCompareHeldout(unittest.TestCase):
    def test_local_model_snapshot_alias_and_recomputed_configuration_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _make_config(model_id="../../work/models/Qwen2.5-Coder-0.5B-Instruct")
            env = _make_env(cfg)
            env["model_id"] = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
            rows = [_make_row(cfg, env, method=m, op_idx=i) for m in
                    ("base_pass1", "base_best_of_8", "trained_checkpoint") for i in range(3)]
            _write_run_dir(root / "run", cfg, env, rows)
            report = compare_runs([root / "run"], root / "out")
            self.assertEqual(report["runs"][0]["environment"]["model_id"], env["model_id"])
            cfg["seed"] += 1
            _write_run_dir(root / "run", cfg, env, rows)
            with self.assertRaisesRegex(HeldoutValidationError, "config_hash mismatch"):
                compare_runs([root / "run"], root / "out")

    def test_pass_rows_require_positive_consistent_measured_timings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _make_config()
            env = _make_env(cfg)
            rows = [_make_row(cfg, env, method=m, op_idx=i) for m in
                    ("base_pass1", "base_best_of_8", "trained_checkpoint") for i in range(3)]
            for field, invalid in (("compiled", False), ("candidate_ms", 0),
                                   ("speedup_vs_eager", 999), ("compile_ms", -1)):
                with self.subTest(field=field):
                    altered = [dict(r) for r in rows]
                    altered[0][field] = invalid
                    _write_run_dir(root / "run", cfg, env, altered)
                    with self.assertRaises(HeldoutValidationError):
                        compare_runs([root / "run"], root / "out")

    def test_valid_two_run_comparison_excludes_val_fixture_oracle_and_preserves_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Run 1: Laptop 0.49B Full-Parameter FP32
            cfg1 = _make_config(
                model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                revision="rev049b",
                training_mode="full_param_0.49b",
                dtype="fp32",
            )
            env1 = _make_env(
                cfg1,
                gpu="NVIDIA GeForce RTX 5060 Laptop GPU",
                run_name="laptop_049b_fp32",
            )
            rows1 = [
                # base_pass1: 1 PASS (compile unavailable), 1 COMPILE_ERROR (gpu unavailable), 1 TIMEOUT
                _make_row(cfg1, env1, method="base_pass1", op_idx=0, status="PASS", speedup_vs_eager=1.2, compile_available=False),
                _make_row(cfg1, env1, method="base_pass1", op_idx=1, status="COMPILE_ERROR", gpu_model="unavailable"),
                _make_row(cfg1, env1, method="base_pass1", op_idx=2, status="TIMEOUT"),
                # base_best_of_8: selection_val rows MUST be excluded (even if passing with huge speedup!)
                _make_row(cfg1, env1, method="base_best_of_8", op_idx=0, status="PASS", speedup_vs_eager=9.9, eval_phase="selection_val", sample_k=0),
                _make_row(cfg1, env1, method="base_best_of_8", op_idx=1, status="PASS", speedup_vs_eager=9.9, eval_phase="selection_val", sample_k=1),
                # fixture and oracle rows MUST also be excluded
                _make_row(cfg1, env1, method="base_best_of_8", op_idx=0, status="PASS", speedup_vs_eager=8.8, fixture=True, run_id_suffix="-fix"),
                _make_row(cfg1, env1, method="base_best_of_8", op_idx=0, status="PASS", speedup_vs_eager=7.7, eval_phase="heldout_test_oracle", selection_method="oracle_same_seed", run_id_suffix="-orc"),
                # Final heldout_test rows for base_best_of_8: 2 PASS (1 fast vs compile, 1 slow vs compile), 1 NUMERICAL_MISMATCH
                _make_row(cfg1, env1, method="base_best_of_8", op_idx=0, status="PASS", speedup_vs_eager=1.5, speedup_vs_compile=1.1, sample_k=3),
                _make_row(cfg1, env1, method="base_best_of_8", op_idx=1, status="PASS", speedup_vs_eager=0.9, speedup_vs_compile=0.8, sample_k=5),
                _make_row(cfg1, env1, method="base_best_of_8", op_idx=2, status="NUMERICAL_MISMATCH", sample_k=2),
                # trained_checkpoint: 2 PASS (1 compile available fast, 1 compile unavailable), 1 RUNTIME_ERROR
                _make_row(cfg1, env1, method="trained_checkpoint", op_idx=0, status="PASS", speedup_vs_eager=1.6, speedup_vs_compile=1.2),
                _make_row(cfg1, env1, method="trained_checkpoint", op_idx=1, status="PASS", speedup_vs_eager=1.1, compile_available=False),
                _make_row(cfg1, env1, method="trained_checkpoint", op_idx=2, status="RUNTIME_ERROR", gpu_model="unavailable"),
            ]
            _write_run_dir(root / "run1", cfg1, env1, rows1)

            # Run 2: Colab T4 7B QLoRA NF4 FP16
            cfg2 = _make_config(
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
                training_mode="qlora_7b_nf4",
                dtype="fp16",
            )
            env2 = _make_env(
                cfg2,
                gpu="Tesla T4",
                checkpoint_weights_sha="sha256_adapter_7b",
                run_name="colab_t4_7b_qlora_fp16",
            )
            rows2 = [
                _make_row(cfg2, env2, method="base_pass1", op_idx=0, status="COMPILE_ERROR", gpu_model="unavailable"),
                _make_row(cfg2, env2, method="base_pass1", op_idx=1, status="COMPILE_ERROR"),
                _make_row(cfg2, env2, method="base_pass1", op_idx=2, status="COMPILE_ERROR"),
                _make_row(cfg2, env2, method="base_best_of_8", op_idx=0, status="PASS", speedup_vs_eager=1.3, speedup_vs_compile=1.05, sample_k=4),
                _make_row(cfg2, env2, method="base_best_of_8", op_idx=1, status="COMPILE_ERROR"),
                _make_row(cfg2, env2, method="base_best_of_8", op_idx=2, status="OOM"),
                _make_row(cfg2, env2, method="trained_checkpoint", op_idx=0, status="PASS", speedup_vs_eager=1.4, speedup_vs_compile=1.1),
                _make_row(cfg2, env2, method="trained_checkpoint", op_idx=1, status="COMPILE_ERROR"),
                _make_row(cfg2, env2, method="trained_checkpoint", op_idx=2, status="COMPILE_ERROR"),
            ]
            _write_run_dir(root / "run2", cfg2, env2, rows2)

            out_dir = root / "out"
            rc = main(["--run-dir", str(root / "run1"), "--run-dir", str(root / "run2"), "--out", str(out_dir)])
            self.assertEqual(rc, 0)

            md_path = out_dir / "comparison.md"
            json_path = out_dir / "comparison.json"
            png_path = out_dir / "comparison.png"
            self.assertTrue(md_path.is_file())
            self.assertTrue(json_path.is_file())
            self.assertTrue(png_path.is_file())
            self.assertGreater(png_path.stat().st_size, 1000)

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertFalse(data["pooled_across_runs"])
            self.assertEqual(len(data["runs"]), 2)

            r1 = data["runs"][0]
            self.assertEqual(r1["environment"]["training_mode_label"], "Full-Parameter 0.49B")
            self.assertEqual(r1["excluded_row_counts"]["selection_val"], 2)
            self.assertEqual(r1["excluded_row_counts"]["fixture"], 1)
            self.assertEqual(r1["excluded_row_counts"]["oracle"], 1)

            # Failures remain in denominator (N=3)
            m_b1 = r1["methods"]["base_pass1"]
            self.assertEqual(m_b1["total_ops"], 3)
            self.assertEqual(m_b1["pass_count"], 1)
            self.assertEqual(m_b1["failure_count"], 2)
            self.assertEqual(m_b1["compile_unavailable_passing_count"], 1)
            # 1 passing op had compile unavailable -> compile-evaluable denominator = 3 - 1 = 2 (2 failures remain)
            self.assertEqual(m_b1["fast_vs_compile_denominator"], 2)
            self.assertEqual(m_b1["fast_vs_compile_count"], 0)
            self.assertAlmostEqual(m_b1["geomean_speedup_vs_eager_correct_only"], 1.2, places=4)
            self.assertIsNone(m_b1["geomean_speedup_vs_compile_correct_only"])

            m_tr = r1["methods"]["trained_checkpoint"]
            self.assertEqual(m_tr["pass_count"], 2)
            self.assertEqual(m_tr["fast_vs_eager_count"], 2)
            self.assertEqual(m_tr["fast_vs_compile_count"], 1)
            self.assertEqual(m_tr["fast_vs_compile_denominator"], 2)
            self.assertEqual(m_tr["compile_unavailable_passing_count"], 1)

            # Paired per-op statuses serialized
            paired = {p["op_name"]: p["paired_pass1_vs_trained"] for p in r1["per_op_paired"]}
            self.assertEqual(paired["rmsnorm_4096"], "both_pass")
            self.assertEqual(paired["int8_weight_only_gemm_512"], "trained_only_pass")
            self.assertEqual(paired["scaled_dot_product_attn_128"], "both_fail")

            # Run 2: 0 passing rows on base_pass1 -> geomean is None (never fake 0.0x)
            r2 = data["runs"][1]
            self.assertEqual(r2["environment"]["training_mode_label"], "7B QLoRA (NF4)")
            self.assertIsNone(r2["methods"]["base_pass1"]["geomean_speedup_vs_eager_correct_only"])

            md = md_path.read_text(encoding="utf-8")
            self.assertIn("Full-Parameter 0.49B", md)
            self.assertIn("7B QLoRA (NF4)", md)
            self.assertIn("N=3", md)
            self.assertIn("different test-time sampling budgets", md)
            self.assertIn("correct-only", md.lower())
            self.assertIn("No statistical significance claims", md)

    def test_validation_leakage_only_selection_val_rows_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _make_config()
            env = _make_env(cfg)
            rows = []
            for method in ("base_pass1", "base_best_of_8", "trained_checkpoint"):
                for op_idx in range(3):
                    # Leak: base_best_of_8 op 0 only has selection_val, no heldout_test row
                    phase = "selection_val" if (method == "base_best_of_8" and op_idx == 0) else "heldout_test"
                    rows.append(_make_row(cfg, env, method=method, op_idx=op_idx, eval_phase=phase))
            _write_run_dir(root / "run", cfg, env, rows)
            with self.assertRaisesRegex(HeldoutValidationError, "Incomplete heldout_test coverage"):
                compare_runs([root / "run"], root / "out")

    def test_duplicate_final_rows_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _make_config()
            env = _make_env(cfg)
            rows = [
                _make_row(cfg, env, method=m, op_idx=i)
                for m in ("base_pass1", "base_best_of_8", "trained_checkpoint")
                for i in range(3)
            ]
            # Duplicate final row for base_pass1 op 0
            rows.append(_make_row(cfg, env, method="base_pass1", op_idx=0, run_id_suffix="-dup"))
            _write_run_dir(root / "run", cfg, env, rows)
            with self.assertRaisesRegex(HeldoutValidationError, "Duplicate final heldout_test row"):
                compare_runs([root / "run"], root / "out")

    def test_mixed_real_gpu_hardware_rejected_even_on_failed_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _make_config()
            env = _make_env(cfg, gpu="Tesla T4")
            rows = [
                _make_row(cfg, env, method=m, op_idx=i)
                for m in ("base_pass1", "base_best_of_8", "trained_checkpoint")
                for i in range(3)
            ]
            # Replace one row with a failed row from a DIFFERENT real GPU
            rows[0] = _make_row(
                cfg,
                env,
                method="base_pass1",
                op_idx=0,
                status="COMPILE_ERROR",
                gpu_model="NVIDIA GeForce RTX 5060 Laptop GPU",
            )
            _write_run_dir(root / "run", cfg, env, rows)
            with self.assertRaisesRegex(HeldoutValidationError, "Mixed real GPU hardware"):
                compare_runs([root / "run"], root / "out")

    def test_mismatched_config_hash_dtype_seed_and_pass_unavailable_gpu_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _make_config()
            env = _make_env(cfg)

            def _base_rows() -> list[dict[str, Any]]:
                return [
                    _make_row(cfg, env, method=m, op_idx=i)
                    for m in ("base_pass1", "base_best_of_8", "trained_checkpoint")
                    for i in range(3)
                ]

            # 1. Mismatched config_hash
            rows_bad_hash = _base_rows()
            rows_bad_hash[0]["config_hash"] = "wrong_hash_0000"
            _write_run_dir(root / "bad_hash", cfg, env, rows_bad_hash)
            with self.assertRaisesRegex(HeldoutValidationError, "config_hash mismatch"):
                compare_runs([root / "bad_hash"], root / "out")

            # 2. Mismatched dtype
            rows_bad_dtype = _base_rows()
            rows_bad_dtype[0]["dtype"] = "fp16"
            _write_run_dir(root / "bad_dtype", cfg, env, rows_bad_dtype)
            with self.assertRaisesRegex(HeldoutValidationError, "Dtype mismatch"):
                compare_runs([root / "bad_dtype"], root / "out")

            # 3. Wrong expected seed for op
            rows_bad_seed = _base_rows()
            rows_bad_seed[1]["seed"] = 999999
            _write_run_dir(root / "bad_seed", cfg, env, rows_bad_seed)
            with self.assertRaisesRegex(HeldoutValidationError, "Seed mismatch"):
                compare_runs([root / "bad_seed"], root / "out")

            # 4. PASS row with gpu_model='unavailable' rejected
            rows_pass_unavail = _base_rows()
            rows_pass_unavail[0]["gpu_model"] = "unavailable"
            _write_run_dir(root / "pass_unavail", cfg, env, rows_pass_unavail)
            with self.assertRaisesRegex(HeldoutValidationError, "GPU mismatch on PASS row"):
                compare_runs([root / "pass_unavail"], root / "out")

            # 5. Oracle-only rows cannot satisfy heldout_test requirement
            rows_oracle = _base_rows()
            rows_oracle[3]["eval_phase"] = "heldout_test"
            rows_oracle[3]["selection_method"] = "oracle_same_seed"
            _write_run_dir(root / "oracle_leak", cfg, env, rows_oracle)
            with self.assertRaisesRegex(HeldoutValidationError, "Incomplete heldout_test coverage"):
                compare_runs([root / "oracle_leak"], root / "out")


if __name__ == "__main__":
    unittest.main()
