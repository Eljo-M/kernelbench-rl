"""Prepare or apply v0.2 documentation from completed, verified study artifacts.

Do not run this script until the owner confirms that all GPU jobs have stopped.
Preparation in this task is limited to syntax checking this file.

After finalization, review without changing the repository::

    python work/update_project_docs.py --diff

Apply only after all GPU work and saved-byte attributes are finalized::

    python work/update_project_docs.py --write --gpu-jobs-complete

Default distribution guidance replaces stale local-only statements and points
to GitHub Releases for versioned archives and SHA-256 manifests. It neither
asserts completed uploads nor leaves a pending-publication statement, so the
source commit/archive can be finalized once before uploads. Website and paper
links identify the intended public locations; hosting success is recorded by
the final publication workflow, not inferred by this offline documentation tool.
``--published-assets`` remains an optional owner attestation in the printed
execution receipt; it is not needed to obtain final, timeless README wording.
Release links target the index, never guessed asset filenames.
This script changes seven Markdown documents and adds one notebook Markdown
cell. It never writes experiment records, source code, attributes or archives.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import difflib
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import sys
from typing import Any
import uuid


WORKSPACE = Path(__file__).resolve().parent.parent
BEGIN = "<!-- BEGIN KERNELBENCH_RL_V02 -->"
END = "<!-- END KERNELBENCH_RL_V02 -->"
NOTEBOOK_NOTE_ID = "kb-v02-provenance-note"
SITE = "https://eljo-m.github.io/kernelbench-rl/"
RELEASES = "https://github.com/Eljo-M/kernelbench-rl/releases"
METHODS = ("base_pass1", "base_best_of_8", "sft_only_checkpoint", "trained_checkpoint")
METHOD_LABELS = ("Base pass@1", "Base best-of-8", "SFT-only pass@1", "Hybrid pass@1")
DOCS = ("README.md", "RESULTS.md", "TRACK_SYNC.md", "experiments/README.md",
        "rl_trainer/README.md", "eval/README.md", "ARCHITECTURE.md")
NOTEBOOK = "notebooks/free_t4.ipynb"
ALLOWED_TARGETS = set(DOCS) | {NOTEBOOK}
STATUSES = {"PASS", "CHEAT", "COMPILE_ERROR", "NUMERICAL_MISMATCH", "TIMEOUT", "OOM", "RUNTIME_ERROR"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key} in {path}")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      object_pairs_hook=no_duplicates, parse_constant=nonfinite)


def read_rows(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    require(all(line.strip() for line in lines), f"Blank record in {path}")
    return [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite {value} in {path}"))) for line in lines]


def integer(value: Any, context: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"Invalid integer {context}: {value!r}")
    return value


def finite(value: Any, context: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid finite value {context}: {value!r}")
    return float(value)


def completed(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    require(isinstance(manifest, dict) and manifest.get("completed_utc") and not manifest.get("error"),
            f"Incomplete or failed stage: {path}")
    require(manifest.get("source_code_dirty") is False, f"Unreviewed source stage: {path}")
    return manifest


def require_hash(value: Any, length: int, context: str) -> str:
    require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None,
            f"Invalid {context}")
    return value


def validate_inputs(repo: Path, metrics_path: Path, manuscript_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    metrics, manuscript = read_json(metrics_path), read_json(manuscript_path)
    require(isinstance(metrics, list) and len(metrics) in (2, 3), "Expected the two v2 tracks and optional completed v3")
    keys = [study_key(metric) for metric in metrics]
    expected_keys = {"v2/laptop", "v2/laptop_7b"} | ({"v3/laptop_7b"} if len(metrics) == 3 else set())
    require(len(set(keys)) == len(keys) and set(keys) == expected_keys, "Duplicate or unexpected study keys")
    require(str(manuscript.get("status", "")).startswith("FINAL") and
            manuscript.get("pending_results", {}).get("required") is False, "Manuscript is not final")
    dt.date.fromisoformat(manuscript.get("publication_date", ""))
    require(manuscript.get("verified_retry_metrics") == metrics, "Manuscript and verified metrics differ")
    displayed = [manuscript.get("title", ""), manuscript.get("abstract", "")]
    displayed += [str(p) for section in manuscript.get("sections", []) for p in section.get("paragraphs", [])]
    displayed += [json.dumps(table) for table in manuscript.get("tables", [])]
    require(re.search(r"\[DRAFT\b|\[PENDING\b|\bTBD\b|results are pending", " ".join(displayed), re.I) is None,
            "Manuscript contains an unresolved result placeholder")
    by_name = {study_key(metric): metric for metric in metrics}
    metrics = [by_name[name] for name in ("v2/laptop", "v2/laptop_7b", "v3/laptop_7b") if name in by_name]
    sys.path.insert(0, str(repo))
    from eval.compare_retry import load_retry_run
    from rl_trainer.experiment_config import ExperimentConfig
    from export_retry import STUDY_COMMIT, registered_v3_study, validate_run
    from finalize_retry_text import comparison_science
    v3_registration = registered_v3_study(repo)
    require((v3_registration is not None) == ("v3/laptop_7b" in by_name),
            "A present v3 study must finish and appear in final metrics; it cannot be silently omitted")
    study_sources = {"v2": set(), "v3": set()}
    for metric in metrics:
        key, q, env = study_key(metric), metric["quality"], metric["environment"]
        version, name = key.split("/")
        run = repo / "experiments" / ("retry_" + version) / name
        cfg = read_json(run / "config.json")
        expected_source = STUDY_COMMIT if version == "v2" else v3_registration["source_commit"]
        protocol_path = run.parent / "protocol.json"
        checked = validate_run(repo, run, expected_source, load_retry_run, ExperimentConfig, protocol_path)
        require(comparison_science(checked["strict"]) == comparison_science(q),
                f"Final metric no longer matches the full strict validator: {key}")
        require("config_snapshot" not in metric or metric["config_snapshot"] == cfg, "Config snapshot changed")
        require(q["environment"]["run_name"] == name, "Quality run identity mismatch")
        mode = "full_param_0.49b" if name == "laptop" else "qlora_7b_nf4"
        require(env["training_mode"] == q["environment"]["training_mode"] == cfg["training_mode"] == mode,
                f"Model track mismatch: {name}")
        require(q["prompt_version"] == cfg["prompt_version"] == "contract_v2" and
                q["training_reward_mode"] == cfg["training_reward_mode"] == "correctness_gated", "Wrong retry protocol")
        expected_epochs, expected_ops, expected_groups = (4, 8, 8) if version == "v2" else (8, 11, 16)
        require(cfg["sft_epochs"] == expected_epochs and
                len(cfg["train_ops"]) == len(set(cfg["train_ops"])) == expected_ops,
                f"Wrong registered supervised schedule: {key}")
        if version == "v3":
            former_fresh = {"l2_normalize_2048", "fused_bias_silu_4096", "matmul_residual_add_512"}
            require(former_fresh.issubset(cfg["train_ops"]) and not former_fresh.intersection(cfg["heldout_ops"] + cfg["development_eval_ops"]),
                    "Former v2 fresh targets must be training-seen in v3")
            require(not cfg.get("resume_from") and cfg["lora_r"] == 16 and cfg["lora_alpha"] == 32 and
                    set(cfg["lora_target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
                    "V3 must retain registered fresh-base rank-16/all-seven initialization")
            require(metric.get("source_provenance") == v3_registration["provenance"], "V3 source provenance differs")
        require(not set(cfg["train_ops"]).intersection(cfg["heldout_ops"] + cfg["development_eval_ops"]),
                "Training/evaluation operator overlap")
        sft_steps = integer(q["sft_optimizer_steps"], "SFT updates", minimum=1)
        grpo_steps = integer(q["grpo_optimizer_steps"], "GRPO updates")
        groups = integer(cfg["max_steps"], "GRPO groups", minimum=1)
        samples = groups * integer(cfg["group_size"], "GRPO group size", minimum=1)
        require(sft_steps == env["sft_optimizer_steps"] == metric["sft_summary"]["optimizer_step_count"] == expected_epochs * expected_ops,
                "Supervised update inventory differs from the completed schedule")
        require(env["steps"] == groups == expected_groups and grpo_steps == env["optimizer_steps"] <= groups,
                "GRPO update/schedule mismatch")
        source = require_hash(env["source_commit"], 40, "study source commit")
        study_sources[version].add(source)
        require(q["environment"]["source_commit"] == source and env["config_hash"] == q["environment"]["config_hash"],
                "Study/quality source or configuration mismatch")
        require(env["gpu"] == q["environment"]["gpu"] and env["dtype"] == q["environment"]["dtype"] == cfg["dtype"],
                "Training/evaluation GPU or precision mismatch")
        require(env["model_revision"] == q["environment"]["revision"] == cfg["model_revision"],
                "Model revision mismatch")
        for filename in ("environment.json", "resume_verified.json", "heldout_environment.json", "sft_only_environment.json"):
            stage = completed(run / filename)
            require(stage["source_commit"] == source and stage["config_hash"] == env["config_hash"],
                    f"Completed stage identity differs: {filename}")
            if filename == "environment.json":
                require(stage == env, "Training environment changed after finalization")
            if filename in ("heldout_environment.json", "sft_only_environment.json"):
                output = "heldout.jsonl" if filename.startswith("heldout") else "sft_only.jsonl"
                require(stage["output_sha256"] == hashlib.sha256((run / output).read_bytes()).hexdigest(),
                        "Evaluation records changed after completion")
        counts = metric.get("training_status_counts")
        require(isinstance(counts, dict) and set(counts).issubset(STATUSES), "Missing/invalid training_status_counts")
        require(all(type(value) is int and value > 0 for value in counts.values()) and sum(counts.values()) == samples,
                "Training status counts do not cover all actual rollouts")
        actual_train = read_rows(run / "train.jsonl")
        require(dict(Counter(row["status"] for row in actual_train)) == counts and len(actual_train) == samples,
                "Training status counts differ from raw records")
        require(metric["training_failures"] == {key: value for key, value in counts.items() if key != "PASS"},
                "Training failures include successful candidates or omit failures")
        for key, ops, available in (("fresh_heldout", cfg["heldout_ops"], METHODS),
                                    ("inspected_development", cfg["development_eval_ops"],
                                     ("base_pass1", "sft_only_checkpoint", "trained_checkpoint"))):
            require(q[key]["ops"] == ops and len(ops) == 3, f"Wrong {key} cohort")
            methods = q[key]["methods"]
            require(set(methods) == set(METHODS), f"Missing method in {key}")
            for method in available:
                stats = methods[method]
                require(stats.get("available") is not False and integer(stats["total_ops"], "denominator") == 3,
                        "Incomplete final operator coverage")
                require(integer(stats["pass_count"], "passes") <= 3, "Pass count exceeds denominator")
                require(stats["budget_k"] == (8 if method == "base_best_of_8" else 1), "Wrong inference budget")
            if key == "inspected_development":
                require(methods["base_best_of_8"].get("available") is False,
                        "Development best-of-eight must remain unavailable")
        inventory = q["inventory"]
        expected = dict(main_rows=39, main_selection_rows=24, main_fresh_final_rows=9,
                        main_development_final_rows=6, sft_only_rows=6,
                        sft_fresh_final_rows=3, sft_development_final_rows=3)
        require(all(integer(inventory[key], key) == value for key, value in expected.items()),
                "Final phase inventory differs from the frozen protocol")
        require(len(read_rows(run / "heldout.jsonl")) == inventory["main_rows"] and
                len(read_rows(run / "sft_only.jsonl")) == inventory["sft_only_rows"], "Raw final row count changed")
        require_hash(env["trainable_weights_sha256"], 64, "final tensor digest")
        require(q["trained_checkpoint_identity"].split(":")[-1] == env["trainable_weights_sha256"], "Final checkpoint mismatch")
        require(q["sft_only_checkpoint_identity"].split(":")[-1] == metric["sft_summary"]["weight_sha256_after"],
                "SFT checkpoint mismatch")
        if grpo_steps == 0:
            require(q["trained_checkpoint_identity"].split(":")[-1] == q["sft_only_checkpoint_identity"].split(":")[-1],
                    "Zero-GRPO checkpoint differs from its SFT-only weights")
        integer(env["peak_allocated_bytes"], "parent CUDA peak")
        finite(env["seconds"], "training seconds")
    require(len(study_sources["v2"]) == 1 and len(study_sources["v3"]) <= 1,
            "Runs within a protocol used different frozen source snapshots")
    regression = repo / "experiments/retry_v2/verifier_regression"
    regression_env = read_json(regression / "environment.json")
    require(regression_env.get("completed_utc") and not regression_env.get("error"), "Updated-worker regression incomplete")
    require(regression_env["source_commit"] in study_sources["v2"], "Regression uses a different v2 study source")
    rows = read_rows(regression / "baselines.jsonl")
    identities = {(row["op_name"], row["dtype"]) for row in rows}
    ops = {row["op_name"] for row in rows}
    require(len(rows) == len(identities) == 60 and len(ops) == 30 and
            identities == {(op, dtype) for op in ops for dtype in ("fp32", "bf16")}, "Regression lacks all 60 distinct checks")
    require(all(row.get("status") == "PASS" and row.get("compiled") is True and row.get("correct") is True and
                row.get("cheat_detected") is False and row.get("seed") == 142 and row.get("independent_seed") == 10142
                for row in rows), "Updated-worker regression contains a failure or mismatched seed")
    return metrics, manuscript, dict(checks=len(rows), passed=len(rows), source_commit=next(iter(study_sources["v2"])))


def study_key(metric: dict[str, Any]) -> str:
    key = metric.get("study_key", metric.get("name", ""))
    # Preserve compatibility with already generated v2-only final artifacts.
    return "v2/" + key if key in {"laptop", "laptop_7b"} else key


def short_label(metric: dict[str, Any]) -> str:
    return metric.get("short_label", {"v2/laptop": "v2 0.49B FP32", "v2/laptop_7b": "v2 7B rank 8",
                                      "v3/laptop_7b": "v3 7B rank 16"}[study_key(metric)])


def relative(doc: str, target: str) -> str:
    return posixpath.relpath(target, posixpath.dirname(doc) or ".")


def link(doc: str, title: str, target: str) -> str:
    return f"[{title}]({relative(doc, target)})"


def cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]])


def fraction(stats: dict[str, Any]) -> str:
    return f"{stats['pass_count']}/{stats['total_ops']}"


def build_block(doc: str, metrics: list[dict[str, Any]], manuscript: dict[str, Any], regression: dict[str, Any],
                guide: str, published: bool) -> str:
    v3_metric = next((metric for metric in metrics if study_key(metric).startswith("v3/")), None)
    paragraphs = ["## v0.2: completed corrected studies",
        f"Publication record: **{manuscript['publication_date']}**. These separately identified v2/v3 protocols use the custom 30-operator registry; they do not claim coverage of the complete official KernelBench benchmark. Original v1 results below remain historical evidence.",
        f"[Project website]({SITE}) · [Paper landing page]({SITE}paper.html) · [Searchable PDF]({SITE}assets/kernelbench-rl-paper.pdf) · [Study report]({SITE}report.html) · [Release index]({RELEASES})",
        "The v2 schedule uses the dtype-aware `contract_v2` prompt, 32 training-only supervised updates, then eight groups of two completions with correctness-gated GRPO. Every non-PASS candidate receives zero training reward; valid PASS speed rewards are preserved rather than replaced by binary values. Equal rewards yield zero relative advantage; even correct groups can have equal or differing speed rewards. The SFT-only checkpoint is saved before GRPO. The original pre-SFT base anchors reference regularization."]
    if v3_metric is not None:
        q, env = v3_metric["quality"], v3_metric["environment"]
        cfg = v3_metric["config_snapshot"]
        paragraphs.append(f"The separate prospective **v3 7B rank 16** study starts from the pinned original base with fresh adapters, not transferred v2 weights or exact resume. Rank {cfg['lora_r']}/alpha {cfg['lora_alpha']} targets q/k/v/o plus gate/up/down projections. It records {q['sft_optimizer_steps']} SFT updates over {len(cfg['train_ops'])} training operators and {env['steps']} GRPO groups; actual GRPO updates are reported below. All three former v2 fresh targets are now training-seen and excluded from v3 fresh/development denominators. New prospective fresh operators are `cosine_similarity_1024`, `fused_bias_gelu_4096`, `matmul_bias_silu_512`. Their generated programs were not previously inspected in the recorded repository exposure audit; public fixed references and related families were already exposed. Rank, targets, schedule, seed and fresh tasks differ from v2, so no causal SFT/GRPO, model-capacity or pooled comparison follows.")
    training_rows, phase_lines = [], []
    fresh_rows = {"v2": [], "v3": []}
    for metric in metrics:
        q, env = metric["quality"], metric["environment"]
        counts = ", ".join(f"`{status}`={count}" for status, count in sorted(metric["training_status_counts"].items()))
        training_rows.append([short_label(metric), q["sft_optimizer_steps"], env["steps"],
                              sum(metric["training_status_counts"].values()), q["grpo_optimizer_steps"], counts])
        fresh_rows[study_key(metric).split("/")[0]].append([short_label(metric), *[fraction(q["fresh_heldout"]["methods"][method]) for method in METHODS]])
        inv = q["inventory"]
        phase_lines.append(f"{short_label(metric)}: {inv['main_rows']} main rows = {inv['main_selection_rows']} selection + "
                           f"{inv['main_fresh_final_rows']} fresh final + {inv['main_development_final_rows']} development final; "
                           f"{inv['sft_only_rows']} separate SFT-only rows = {inv['sft_fresh_final_rows']} fresh + {inv['sft_development_final_rows']} development.")
    paragraphs += [table(["Model track", "SFT updates", "GRPO groups", "GRPO candidates", "GRPO updates", "Training statuses"], training_rows)]
    for version in ("v2", "v3"):
        if fresh_rows[version]:
            paragraphs += [version.upper() + " fresh held-out passes (N=3 per method; different fresh operators across protocols):",
                           table(["Model track", *METHOD_LABELS], fresh_rows[version])]
    paragraphs += ["Pass@1 methods use one completion; base best-of-eight uses eight validation-selection samples per fresh operator, followed by independent regrading of its winner. Selection records are excluded from final denominators. Failures remain in pass-rate and fast-versus-eager denominators; correct-only speedups and unavailable compiled comparisons are labeled separately.",
                   "\n".join("- " + line for line in phase_lines)]
    if doc in {"RESULTS.md", "experiments/README.md", "eval/README.md"}:
        development = []
        for metric in metrics:
            methods = metric["quality"]["inspected_development"]["methods"]
            development.append([short_label(metric), fraction(methods["base_pass1"]),
                                fraction(methods["sft_only_checkpoint"]), fraction(methods["trained_checkpoint"]), "Unavailable"])
        paragraphs += ["Previously inspected development passes, kept separate from the fresh cohort:",
                       table(["Model track", "Base pass@1", "SFT-only pass@1", "Hybrid pass@1", "Best-of-eight"], development)]
    paragraphs += ["V2 fresh targets were `l2_normalize_2048`, `fused_bias_silu_4096` and `matmul_residual_add_512`; they are training-seen in the separate v3 follow-up when present. Each prospective trio tests programs within shared training families, not unseen-family out-of-distribution behavior. Previously inspected RMSNorm-4096, INT8 weight-only GEMM and attention remain development regressions. Protocols, model/precision tracks and different fresh cohorts are not pooled. Historical `contract_v1` and later `contract_v2` outcomes are unpaired comparisons. The hybrid includes SFT and establishes no RL-only causal effect; N=3 and one training seed per study do not establish significance or broad generalization.",
        "The laptop 7B studies update LoRA adapters over a frozen NF4 base with an FP16 GPU decoder and fixed FP32 CPU embedding/output computations. They are separate from both the historical free-T4 placement and full-parameter 0.49B FP32 training. FSDP, multi-node training and full-parameter 7B remain outside implemented and validated scope."]
    zero = [short_label(metric) for metric in metrics if metric["quality"]["grpo_optimizer_steps"] == 0]
    if zero:
        paragraphs.append("Zero GRPO updates were recorded for " + "; ".join(zero) + ". Each corresponding final trainable-tensor digest equals its SFT-only digest. Sampled-output or timing differences between those arms are not RL parameter gains; skipped-update zero diagnostics are sentinels.")
    paragraphs += [f"Fixed baselines are curated implementations prepared with coding-assistant support, distinct from programs sampled by the evaluated policies; historical files' 'handwritten' wording does not assert exclusively human authorship. The historical v1 **180/180** campaign and geometric means remain unchanged: FP32 **2.494x eager / 1.089x compiled**, BF16 **5.417x / 1.248x**, with 30 eager and 29 available compiled operators. The separate updated-worker correctness regression passed **{regression['passed']}/{regression['checks']}** operator/precision records at seed 142 and independent seed 10142. It checks all 30 operators in FP32/BF16 and does not re-estimate the historical speedup campaign."]
    if doc in {"README.md", "RESULTS.md", "rl_trainer/README.md", "ARCHITECTURE.md"}:
        memory = [[short_label(metric), f"{metric['environment']['seconds']:.2f}",
                   f"{metric['environment']['peak_allocated_bytes']/2**30:.3f}"] for metric in metrics]
        paragraphs += [table(["Model track", "Training seconds", "Parent peak allocated CUDA GiB"], memory),
            "The CUDA peak is the training parent process's PyTorch allocator peak; it excludes child-worker and other applications' device allocations. A checksum-verified checkpoint reload restores optimizer/RNG state without extra training steps; this is not an independent next-step trajectory replication or a separate optimizer/RNG digest comparison."]
    if doc in {"README.md", "rl_trainer/README.md", "ARCHITECTURE.md", "experiments/README.md"}:
        paragraphs += ["From the repository root, the v0.2 entrypoint is "
                       "`python -m rl_trainer.run_study --config experiments/retry_v2/laptop/config.json` "
                       "(use `experiments/retry_v2/laptop_7b/config.json` for the separate 7B track). "
                       "The reproduction guide gives the complete setup, fixed model revisions and checkpoint restoration layout. The legacy `train_grpo` development/search path remains a separate entrypoint."]
    if doc == "eval/README.md":
        paragraphs += ["From the repository root, regenerate the strict retry comparison with:\n\n```bash\n"
                       "python -m eval.compare_retry --run-dir experiments/retry_v2/laptop "
                       "--run-dir experiments/retry_v2/laptop_7b --out experiments/retry_v2/comparison\n```\n\n"
                       "`compare_retry` verifies config/model revision, GPU/precision, exact row inventories, prompt/source/checkpoint identity, "
                       "preregistered seeds and the validation-selected winner; `development_test` never enters fresh held-out counts."]
        if v3_metric is not None:
            paragraphs.append("Regenerate v3 separately with `python -m eval.compare_retry --run-dir experiments/retry_v3/laptop_7b --protocol experiments/retry_v3/protocol.json --out experiments/retry_v3/comparison`. Preserve the original v2 comparison JSON; a duplicate basename `laptop_7b` is not a unique study identity.")
    paragraphs += [f"Checkpoint tensors and optimizer/RNG files are excluded from Git. For versioned archives and SHA-256 manifests, see [GitHub Releases]({RELEASES}).",
        "The publication snapshot marks `experiments/**`, `docs/**` and `paper/**` as `-text` in `.gitattributes`, preserving exact saved bytes, existing line endings and embedded output/proof hashes in committed blobs and clean clones. Earlier Git snapshots used text normalization and may contain LF-normalized blobs where Windows saved files contained CRLF. Earlier release ZIPs are unchanged; their recorded hashes remain their own identities. This packaging correction does not rerun experiments or retroactively change study commits.",
        "Evidence: " + " · ".join([link(doc, "strict comparison", "experiments/retry_v2/comparison/retry_comparison.md"),
                                   link(doc, "strict JSON", "experiments/retry_v2/comparison/retry_comparison.json"),
                                   link(doc, "frozen protocol", "experiments/retry_v2/protocol.json"),
                                   link(doc, "reproduction guide", guide),
                                   link(doc, "manuscript source", "paper/manuscript.json"),
                                   link(doc, "worker regression", "experiments/retry_v2/verifier_regression"),
                                   link(doc, "offline final report", "docs/report.html")]) + ".",
        "Frozen study source: `" + regression["source_commit"] + "`. The later publication snapshot contains documentation, saved-byte packaging and result assets; its distinct source identity is recorded in release provenance."]
    if v3_metric is not None:
        paragraphs += ["V3 evidence: " + " · ".join([link(doc, "separate strict comparison", "experiments/retry_v3/comparison/retry_comparison.md"),
            link(doc, "separate strict JSON", "experiments/retry_v3/comparison/retry_comparison.json"),
            link(doc, "prospective protocol", "experiments/retry_v3/protocol.json"),
            link(doc, "registered source provenance", "experiments/retry_v3/source_provenance.json")]) +
            ". V3 study source prefix `" + v3_metric["environment"]["source_commit"][:12] +
            "` is 12 characters only; full source/configuration/tensor identities are retained in the linked manifests and manuscript metadata."]
    return BEGIN + "\n\n" + "\n\n".join(paragraphs) + "\n\n" + END


def historical_edits(doc: str, text: str, published: bool) -> str:
    if doc == "RESULTS.md":
        text = re.sub(r"\A# [^\n]+", "# KernelBench-RL results", text, count=1)
    replacements = {
        "README.md": [("## Completed laptop training experiment", "## Historical v1: laptop training experiment"),
                      ("## Completed free-T4 experiment", "## Historical v1: free-T4 experiment"),
                      ("Both the 0.49B full-parameter and free-T4 7B QLoRA experiments completed training", "In historical v1, both the 0.49B full-parameter and free-T4 7B QLoRA experiments completed training")],
        "rl_trainer/README.md": [("## CUDA bridge and free-resource experiments", "## Historical v1: CUDA bridge and free-resource experiments"),
                               ("The train/held-out split\ncontains", "The historical v1 train/held-out split\ncontains")],
        "eval/README.md": [("## Paired held-out model comparison", "## Historical v1: paired held-out model comparison")],
        "ARCHITECTURE.md": [("Kernel rewards penalize", "In the historical v1 shaped-reward mode, kernel rewards penalize"),
                            ("the completed laptop ablation", "the completed historical v1 laptop ablation")],
    }
    for before, after in replacements.get(doc, []):
        text = text.replace(before, after)
    legacy_heading = {"RESULTS.md": "## Historical v1: original free-resource results",
                      "TRACK_SYNC.md": "## Historical v1: coordination record",
                      "experiments/README.md": "## Historical v1: experiment index"}.get(doc)
    if legacy_heading and legacy_heading not in text:
        first_line, remaining = text.split("\n", 1)
        text = first_line + "\n\n" + legacy_heading + "\n" + remaining
    # Distribution guidance is deliberately valid on either side of upload.
    # It does not require a second source commit after publishing the archives.
    text = text.replace("Checkpoints are stored locally and excluded\n  from Git.",
                        "Checkpoint tensors and optimizer/RNG files are excluded from Git.\n"
                        f"  For versioned archives and SHA-256 manifests, see [GitHub Releases]({RELEASES}).")
    text = text.replace("Model/adapter, optimizer and RNG checkpoint files are excluded from Git. They\nremain local or in exported Colab bundles.",
                        "Checkpoint tensors and optimizer/RNG files are excluded from Git.\n"
                        f"For versioned archives and SHA-256 manifests, see [GitHub Releases]({RELEASES}).")
    return text


def upsert_block(text: str, block: str) -> str:
    require(text.count(BEGIN) == text.count(END) and text.count(BEGIN) in (0, 1), "Ambiguous managed section markers")
    if BEGIN in text:
        require(text.index(BEGIN) < text.index(END), "Managed section markers are reversed")
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n*", "", text, count=1, flags=re.S)
    require(re.match(r"# [^\n]+\n", text) is not None, "Document needs a leading Markdown H1")
    first_line, remaining = text.split("\n", 1)
    return first_line + "\n\n" + block + "\n\n" + remaining.lstrip("\n")


def notebook_note(raw: bytes, guide: str) -> bytes:
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    text = raw.decode("utf-8-sig")
    notebook = json.loads(text)
    cells = notebook.get("cells")
    require(isinstance(cells, list), "Notebook has no cells")
    managed = [index for index, entry in enumerate(cells) if entry.get("id") == NOTEBOOK_NOTE_ID]
    require(managed in ([], [0]), "Notebook note must be unique and first")
    source = ("## Version and workflow provenance\n\n"
              "This legacy free-T4 notebook pins the post-study cache-scoring fix "
              "`73a9220c5d3182ccc21f3e4cbcd0ec512aa1d48c`. The historical v1 quality study executed "
              "`e72c42e794643ed4c198937e86427c3640b977c5`; the later cache fix does not retroactively rerun it. "
              "This notebook is not an exact bitwise replay of that historical execution and does not implement the "
              "v0.2 laptop workflow with 32 SFT updates, correctness-gated GRPO and fixed CPU vocabulary computation for 7B. "
              f"Use the [v0.2 reproduction guide](../{guide}) and [frozen protocol](../experiments/retry_v2/protocol.json) for that study. "
              "Existing notebook code cells, validation statements and preflight defaults are preserved below.\n")
    note = dict(cell_type="markdown", id=NOTEBOOK_NOTE_ID, metadata={}, source=source.splitlines(keepends=True))
    encoded = json.dumps(note, indent=1, ensure_ascii=False)
    newline = "\r\n" if "\r\n" in text else "\n"
    match = re.search(r'"cells"\s*:\s*\[', text)
    require(match is not None, "Cannot locate notebook cell array")
    if managed:
        start = match.end()
        while text[start].isspace():
            start += 1
        _, end = json.JSONDecoder().raw_decode(text, start)
        replacement = newline.join([encoded.splitlines()[0], *("  " + line for line in encoded.splitlines()[1:])])
        updated = text[:start] + replacement + text[end:]
        expected = cells[1:]
    else:
        replacement = newline.join("  " + line for line in encoded.splitlines())
        updated = text[:match.end()] + newline + replacement + "," + text[match.end():]
        expected = cells
    result = json.loads(updated)
    require(result["cells"][0] == note and result["cells"][1:] == expected,
            "Notebook content changed outside the top managed Markdown note")
    before_other = {key: value for key, value in notebook.items() if key != "cells"}
    after_other = {key: value for key, value in result.items() if key != "cells"}
    require(before_other == after_other, "Notebook metadata changed")
    return bom + updated.encode("utf-8")


def prepare(repo: Path, metrics: list[dict[str, Any]], manuscript: dict[str, Any], regression: dict[str, Any],
            guide: str, published: bool) -> list[tuple[Path, bytes, bytes]]:
    changes = []
    for doc in DOCS:
        target = repo / doc
        require(target.is_file() and not target.is_symlink(), f"Missing/unsafe documentation target {target}")
        before = target.read_bytes()
        text = before.decode("utf-8-sig").replace("\r\n", "\n")
        # Strip our block before targeted legacy edits, so reruns are idempotent.
        require(text.count(BEGIN) == text.count(END) and text.count(BEGIN) in (0, 1), "Ambiguous existing managed block")
        if BEGIN in text:
            require(text.index(BEGIN) < text.index(END), "Reversed existing managed block")
            text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n*", "", text, count=1, flags=re.S)
        text = historical_edits(doc, text, published)
        after = upsert_block(text, build_block(doc, metrics, manuscript, regression, guide, published)).encode("utf-8")
        if after != before:
            changes.append((target, before, after))
    target = repo / NOTEBOOK
    require(target.is_file() and not target.is_symlink(), "Missing/unsafe legacy notebook")
    before = target.read_bytes()
    after = notebook_note(before, guide)
    if after != before:
        changes.append((target, before, after))
    return changes


def apply_changes(repo: Path, changes: list[tuple[Path, bytes, bytes]]) -> None:
    staged, replaced = [], []
    try:
        for target, before, after in changes:
            require(target.resolve().is_relative_to(repo.resolve()) and target.relative_to(repo).as_posix() in ALLOWED_TARGETS,
                    f"Refusing unexpected documentation target {target}")
            require(target.read_bytes() == before, f"Concurrent edit detected: {target}")
            staging = target.with_name("." + target.name + ".docs-" + uuid.uuid4().hex)
            staging.write_bytes(after)
            staged.append((staging, target, before))
        for staging, target, before in staged:
            require(target.read_bytes() == before, f"Concurrent edit detected: {target}")
            staging.replace(target)
            replaced.append((target, before))
    except BaseException:
        for target, before in reversed(replaced):
            rollback = target.with_name("." + target.name + ".rollback-" + uuid.uuid4().hex)
            rollback.write_bytes(before)
            rollback.replace(target)
        raise
    finally:
        for staging, _, _ in staged:
            staging.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=WORKSPACE / "outputs/kernelbench-rl")
    parser.add_argument("--metrics", type=Path, default=WORKSPACE / "outputs/RETRY_VERIFIED_METRICS.json")
    parser.add_argument("--manuscript", type=Path, default=WORKSPACE / "outputs/KernelBench-RL-Manuscript.json")
    parser.add_argument("--reproduction-guide", default="REPRODUCING_RETRY.md", help="Repository-relative Markdown guide")
    parser.add_argument("--write", action="store_true", help="Apply reviewed updates; default only reports proposed changes")
    parser.add_argument("--diff", action="store_true", help="Print the complete proposed diff")
    parser.add_argument("--gpu-jobs-complete", action="store_true", help="Owner attests that all GPU jobs have stopped")
    parser.add_argument("--published-assets", action="store_true", help="Optional owner attestation in the printed receipt; documentation guidance is timeless either way")
    args = parser.parse_args()
    require(not args.write or args.gpu_jobs_complete, "--write requires explicit --gpu-jobs-complete after all GPU jobs stop")
    repo = args.repo.resolve()
    guide_path = (repo / args.reproduction_guide).resolve()
    require(guide_path.is_relative_to(repo) and guide_path.suffix == ".md", "Guide must be a repository-relative Markdown file")
    guide = guide_path.relative_to(repo).as_posix()
    metrics, manuscript, regression = validate_inputs(repo, args.metrics, args.manuscript)
    if args.write:
        linked_targets=[guide, "paper/manuscript.json", "paper/KernelBench-RL-Paper.pdf", "docs/report.html",
                        "experiments/retry_v2/comparison/retry_comparison.md", "experiments/retry_v2/comparison/retry_comparison.json"]
        if any(study_key(metric).startswith("v3/") for metric in metrics):
            linked_targets += ["experiments/retry_v3/comparison/retry_comparison.md",
                               "experiments/retry_v3/comparison/retry_comparison.json",
                               "experiments/retry_v3/protocol.json", "experiments/retry_v3/source_provenance.json"]
        for target in linked_targets:
            require((repo / target).is_file(), f"Copy final linked artifact before documentation publication: {target}")
        attributes = (repo / ".gitattributes").read_text(encoding="utf-8")
        require(all(re.search(r"(?m)^" + re.escape(pattern) + r"\s+-text\s*$", attributes) for pattern in
                    ("experiments/**", "docs/**", "paper/**")), "Apply saved-byte .gitattributes rules after GPU jobs before --write")
    changes = prepare(repo, metrics, manuscript, regression, guide, args.published_assets)
    if args.diff:
        for target, before, after in changes:
            label = target.relative_to(repo).as_posix()
            print("".join(difflib.unified_diff(before.decode("utf-8-sig").splitlines(keepends=True),
                                             after.decode("utf-8-sig").splitlines(keepends=True),
                                             fromfile="a/" + label, tofile="b/" + label)), end="")
    if args.write:
        apply_changes(repo, changes)
    print(json.dumps(dict(written=args.write, changed=[dict(path=target.relative_to(repo).as_posix(),
                     before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(),
                     bytes=len(after)) for target, before, after in changes],
                     published_assets_attested=args.published_assets,
                     pyproject_proposal_only=dict(version="0.2.0", description="Verified GPU kernel synthesis with training-only SFT, correctness-gated GRPO and auditable evaluation",
                                                  publication_extra=["reportlab>=4.4,<5", "pypdf>=6,<7"])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
