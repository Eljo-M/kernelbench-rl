"""Plot the actual completed 88-example v3 SFT stage, separately from v2.

Run only after sft_summary.json and the durable SFT-only checkpoint exist::

    work/.venv/Scripts/python.exe work/paper-source/plot_followup.py

No model, tokenizer or CUDA runtime is loaded. Partial/stale logs are refused.
The v2 PNG and its metadata/default output are never written by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid
from typing import Any, Sequence

from plot_training import (DEFAULT_REPO, WORKSPACE, MAX_PNG_BYTES, ORDER_RULE,
                           TrainingTrack, _json, load_track, require)


STUDY_KEY = "v3/laptop_7b"
CONFIG_HASH = "0dfb0a14abad3067"
STUDY_SOURCE = "2c174abe268495159c10d538c92d071e7733828d"
EPOCHS, OPERATORS, STEPS = 8, 11, 88
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def read(path: Path) -> Any:
    return _json(path.read_text(encoding="utf-8-sig"), str(path))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_config_hash(cfg: dict[str, Any]) -> str:
    """Exact current ExperimentConfig hash rules, without importing GPU code.

    The saved config contains the complete dataclass fields. A different field
    set/rule cannot silently pass the registered 0dfb identity check.
    """
    keys = dict(cfg)
    supplied = keys.pop("config_hash", None)
    for name in ("max_steps", "checkpoint_every", "checkpoint_dir", "resume_from", "output_jsonl"):
        keys.pop(name)
    if keys.get("prompt_version") == "contract_v1":
        keys.pop("prompt_version")
    if keys.get("training_reward_mode") == "legacy_shaped":
        keys.pop("training_reward_mode")
    if keys.get("sft_epochs") == 0:
        for name in ("sft_epochs", "sft_learning_rate", "sft_max_tokens", "sft_output_jsonl"):
            keys.pop(name)
    if keys.get("onpolicy_logp_tolerance") == .001:
        keys.pop("onpolicy_logp_tolerance")
    if not keys.get("development_eval_ops"):
        keys.pop("development_eval_ops", None)
    if not keys.get("qlora_cpu_offload"):
        keys.pop("qlora_cpu_offload", None)
    result = hashlib.sha256(json.dumps(keys, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()[:16]
    require(supplied is None or supplied == result, "Supplied config_hash contradicts the saved configuration")
    return result


def load_followup(repo: Path) -> tuple[TrainingTrack, dict[str, Any]]:
    root = repo / "experiments/retry_v3"
    run = root / "laptop_7b"
    # The summary is written only after all SFT optimizer steps. Check it before
    # opening the log so an ordinary partial training stage is refused early.
    summary = read(run / "sft_summary.json")
    require(summary.get("ran") is True and summary.get("optimizer_step_count") == STEPS and
            summary.get("total_examples") == STEPS and summary.get("sft_epochs") == EPOCHS,
            "V3 supervised stage is incomplete; require the completed 88-update summary")
    cfg = read(run / "config.json")
    require(frozen_config_hash(cfg) == CONFIG_HASH, "V3 saved config does not match the frozen 0dfb identity")
    require(cfg.get("seed") == 242 and cfg.get("lora_r") == 16 and cfg.get("lora_alpha") == 32 and
            set(cfg.get("lora_target_modules", [])) == TARGETS and not cfg.get("resume_from") and
            cfg.get("qlora_cpu_offload") is True, "Wrong v3 seed/rank/targets/initialization/placement")
    require(cfg.get("max_steps") == 16 and cfg.get("group_size") == 2 and
            cfg.get("sft_learning_rate") == 1e-4, "V3 full schedule/LR differs from preregistration")
    source_path, protocol_path = root / "source_provenance.json", root / "protocol.json"
    provenance, protocol = read(source_path), read(protocol_path)
    require(provenance.get("schema_version") == "kernelbench_study_source_provenance_v1" and
            provenance.get("study_source_commit") == STUDY_SOURCE and
            provenance.get("published_before_training") is True and
            provenance.get("config_hashes") == {"laptop_7b": CONFIG_HASH}, "Wrong registered v3 source/config identity")
    require(provenance.get("protocol_path") == "experiments/retry_v3/protocol.json" and
            provenance.get("protocol_sha256") == sha256(protocol_path), "Frozen v3 protocol bytes changed")
    require(protocol.get("config_hashes", {}).get("laptop_7b") == CONFIG_HASH and
            protocol.get("run_schedules", {}).get("laptop_7b") == dict(max_steps=16, sft_epochs=8, group_size=2),
            "V3 protocol schedule/config mismatch")
    track = load_track(run, STUDY_KEY, "qlora_7b_nf4", "v3 7B rank 16 / all seven projections",
                       expected_epochs=EPOCHS, expected_operators=OPERATORS)
    require(summary.get("records") == track.rows, "SFT summary and durable log disagree")
    require(summary.get("total_completion_tokens") == sum(row["completion_tokens"] for row in track.rows),
            "SFT summary completion-token inventory differs")
    manifest_path = root / "token_feasibility.json"
    manifest = read(manifest_path)
    examples = manifest.get("examples", [])
    require(manifest.get("config_hash") == CONFIG_HASH and len(examples) == OPERATORS and
            {ex.get("op_name") for ex in examples} == set(cfg["train_ops"]), "Wrong pretraining example identity inventory")
    by_op = {ex["op_name"]: ex for ex in examples}
    for row in track.rows:
        expected = by_op[row["op_name"]]
        require(all(row[key] == expected[key] for key in
                    ("prompt_tokens", "completion_tokens", "prompt_sha256", "source_sha256")),
                "Logged prompt/source/token identity differs from the recorded pretraining dataset")
        require(expected["full_tokens"] == row["prompt_tokens"] + row["completion_tokens"] <= cfg["sft_max_tokens"],
                "Supervised full-token count exceeds the frozen limit or differs from preflight")
    checkpoint_path = repo / cfg["checkpoint_dir"] / "sft_only/step_00000/checkpoint_state.json"
    state = read(checkpoint_path)
    require(state.get("config_hash") == CONFIG_HASH and state.get("step") == 0 and
            state.get("optimizer_step_count") == 0 and state.get("sft_optimizer_steps") == STEPS and
            state.get("is_fixture") is False and state.get("trainable_weights_sha256") == summary.get("weight_sha256_after"),
            "Durable SFT-only checkpoint does not match the completed supervised stage")
    return track, dict(source_commit=STUDY_SOURCE, config_hash=CONFIG_HASH,
                       source_provenance_sha256=sha256(source_path), protocol_sha256=sha256(protocol_path),
                       token_feasibility_sha256=sha256(manifest_path), sft_summary_sha256=sha256(run / "sft_summary.json"),
                       sft_checkpoint_metadata_sha256=sha256(checkpoint_path),
                       sft_checkpoint_weights_sha256=state["trainable_weights_sha256"],
                       sft_checkpoint_identity="checkpoint:0:" + state["trainable_weights_sha256"])


def plot_followup(track: TrainingTrack, identities: dict[str, Any], out: Path, metadata: Path, *, dpi: int = 180) -> dict[str, Any]:
    require(out.suffix.lower() == ".png" and out.resolve() != metadata.resolve(), "Use distinct PNG/JSON output paths")
    require(out.name != "sft_training.png" and metadata.name != "sft_training.json",
            "Refusing to overwrite the preserved v2 figure or metadata")
    require(len(track.rows) == STEPS and len(track.epoch_means) == EPOCHS, "Require the validated full 88-step v3 track")
    require(100 <= dpi <= 240, "DPI must be between 100 and 240")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    blue, orange, ink = "#16697a", "#bb6520", "#182c3d"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink, "xtick.color": ink, "ytick.color": ink}):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        fig.subplots_adjust(left=.105, right=.98, top=.74, bottom=.30)
        fig.suptitle("V3 7B rank 16: supervised completion fitting", fontsize=11, fontweight="bold", y=.98)
        fig.text(.5, .905, f"All seven linear projection targets | Seed {track.config['seed']} | SFT LR {track.learning_rate:g}",
                 fontsize=8.5, ha="center", va="center")
        fig.legend([Line2D([], [], color=blue, marker="o", markersize=2.6, linewidth=.8),
                    Line2D([], [], color=orange, linewidth=2.1)],
                   ["Logged example NLL", "Epoch mean (11 examples)"], frameon=False,
                   loc="upper center", bbox_to_anchor=(.5, .865), ncol=2, fontsize=8.5)
        losses = [float(row["loss"]) for row in track.rows]
        ax.plot(range(1, STEPS + 1), losses, color=blue, linewidth=.7, marker="o", markersize=2.2, alpha=.9, zorder=3)
        for epoch, mean in enumerate(track.epoch_means, 1):
            ax.hlines(mean, 1 + (epoch - 1) * OPERATORS, epoch * OPERATORS, color=orange, linewidth=2.1, zorder=4)
        for epoch in range(1, EPOCHS):
            ax.axvline(epoch * OPERATORS + .5, color="#cbd4db", linewidth=.6, linestyle=(0, (3, 3)), zorder=1)
        ax.set(xlim=(.5, STEPS + .5), ylim=(0, max(max(losses) * 1.12, 1e-6)),
               xticks=[1, *range(OPERATORS, STEPS + 1, OPERATORS)],
               xlabel="Supervised example / update index", ylabel="Completion NLL (nats/token)")
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", color="#e4e9ed", linewidth=.6, zorder=0)
        values = [f"{i}: {value:.4f}" if value >= .001 else f"{i}: {value:.2e}"
                  for i, value in enumerate(track.epoch_means, 1)]
        fig.text(.54, .18, "Arithmetic epoch means (nats/token; 11 example means per epoch)",
                 fontsize=8.5, ha="center", va="center")
        fig.text(.54, .12, "    ".join(values[:4]) + "\n" + "    ".join(values[4:]),
                 fontsize=8.5, ha="center", va="center", linespacing=1.4)
        fig.text(.5, .043, "Teacher-forced training losses, measured before each update.\n"
                 "Generated-program validity is evaluated separately.", fontsize=8.5,
                 ha="center", va="center", color="#415769", linespacing=1.3)
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = out.with_name("." + out.stem + ".building-" + uuid.uuid4().hex + ".png")
        try:
            fig.savefig(staging, dpi=dpi, facecolor="white", format="png",
                        pil_kwargs={"optimize": True, "compress_level": 9},
                        metadata={"Title": "V3 supervised completion-token NLL",
                                  "Description": "Actual pre-update training losses; not generated-program quality."})
            require(staging.stat().st_size <= MAX_PNG_BYTES, "Follow-up PNG exceeds 200000 bytes; lower --dpi")
            staging.replace(out)
        finally:
            plt.close(fig)
            staging.unlink(missing_ok=True)
    report = dict(schema_version="kernelbench_sft_followup_plot_v1", study_key=STUDY_KEY,
        figure=str(out.resolve()), bytes=out.stat().st_size, figure_sha256=sha256(out),
        **identities, config_file_sha256=track.config_file_sha256, sft_log_sha256=track.log_sha256,
        model_revision=track.config["model_revision"], seed=track.config["seed"], sft_learning_rate=track.learning_rate,
        adapter_rank=track.config["lora_r"], adapter_alpha=track.config["lora_alpha"],
        adapter_target_modules=track.config["lora_target_modules"], train_ops=track.config["train_ops"],
        epochs=EPOCHS, examples_per_epoch=OPERATORS, completed_steps=STEPS,
        epoch_means_nats_per_token=track.epoch_means, observed_operator_order=[row["op_name"] for row in track.rows],
        ordering_rule=ORDER_RULE, inference_quality_claim=False,
        metric="Logged completion-token mean negative log likelihood, nats/token, before each example update",
        epoch_aggregation="Arithmetic mean of eleven logged per-example means; not token-weighted",
        layout=dict(canvas_inches=[6.5, 4.2], intended_manuscript_width_inches=6.5,
                    minimum_text_size_pt_at_intended_width=8.5, required_max_height_inches_for_full_width=4.2),
        caption="V3 7B rank-16 adapters on all seven decoder linear projections: logged completion-token negative log likelihood during eight supervised epochs over eleven training operators. Points are observed pre-update per-example means; horizontal segments average the eleven example means in each epoch, not individual tokens. Teacher-forced training loss does not establish generated-program validity or held-out quality; these data are separate from the v2 figure.")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    staging = metadata.with_name("." + metadata.name + ".building-" + uuid.uuid4().hex)
    try:
        staging.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.replace(metadata)
    finally:
        staging.unlink(missing_ok=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=WORKSPACE / "outputs/paper-assets/sft_followup.png")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    track, identities = load_followup(args.repo.resolve())
    report = plot_followup(track, identities, args.out, args.metadata or args.out.with_suffix(".json"), dpi=args.dpi)
    print(json.dumps({key: report[key] for key in ("figure", "bytes", "study_key", "completed_steps", "epoch_means_nats_per_token")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
