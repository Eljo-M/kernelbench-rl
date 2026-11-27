"""Plot completed retry SFT logs, never partial or reconstructed training data.

Run only after both supervised stages have finished::

    work/.venv/Scripts/python.exe work/paper-source/plot_training.py \
        --out outputs/paper-assets/sft_training.png

Inputs are the two actual retry ``sft.jsonl`` files and their sibling configs.
The figure shows the logged, pre-update completion-token mean NLL for each
example and arithmetic means of the eight example losses in each epoch. Epoch
means are not token-weighted and are not a fixed-checkpoint validation loss.
Teacher-forced fitting does not establish generated-program validity.

No model, CUDA runtime, repository Python module, or training job is loaded.
Matplotlib is imported only after both logs pass all completeness checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import uuid
from typing import Any, Sequence


def default_layout():
    for ancestor in Path(__file__).resolve().parents:
        if (ancestor/'rl_trainer/experiment_config.py').is_file():
            workspace=ancestor.parent.parent if ancestor.parent.name=='outputs' else ancestor
            return workspace,ancestor
    workspace=Path(__file__).resolve().parents[2]
    return workspace,workspace/'outputs/kernelbench-rl'


WORKSPACE,DEFAULT_REPO = default_layout()
EXPECTED_EPOCHS = 4
EXPECTED_OPERATORS = 8
EXPECTED_STEPS = EXPECTED_EPOCHS * EXPECTED_OPERATORS
MAX_PNG_BYTES = 200_000
ORDER_RULE = "random.Random(config.seed + 1000 * epoch).shuffle(config.train_ops indices)"
TRACKS = (
    ("laptop", "full_param_0.49b", "0.49B full-parameter | FP32"),
    ("laptop_7b", "qlora_7b_nf4", "7B QLoRA | NF4 / FP16 decoder"),
)


class TrainingLogError(ValueError):
    """Incomplete or contradictory supervised-training evidence."""


@dataclass(frozen=True)
class TrainingTrack:
    name: str
    label: str
    run_dir: Path
    config: dict[str, Any]
    rows: list[dict[str, Any]]
    epoch_means: list[float]
    learning_rate: float
    log_sha256: str
    config_file_sha256: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingLogError(message)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text: str, context: str) -> Any:
    def reject(value: str) -> None:
        raise TrainingLogError(f"{context}: nonfinite JSON value {value}")
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=reject)
    except (json.JSONDecodeError, TrainingLogError) as exc:
        raise TrainingLogError(f"{context}: {exc}") from exc


def _number(value: Any, context: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{context}: expected a number")
    number = float(value)
    require(math.isfinite(number) and (number > 0 if positive else number >= 0),
            f"{context}: expected a finite {'positive' if positive else 'nonnegative'} value")
    return number


def load_track(run_dir: Path, name: str, mode: str, label: str, *,
               expected_epochs: int = EXPECTED_EPOCHS,
               expected_operators: int = EXPECTED_OPERATORS) -> TrainingTrack:
    """Validate a complete seeded log; defaults preserve the v2 4x8 protocol."""
    require(type(expected_epochs) is int and expected_epochs > 0 and
            type(expected_operators) is int and expected_operators > 0,
            "Expected epoch/operator counts must be positive integers")
    expected_steps = expected_epochs * expected_operators
    config_path, log_path = run_dir / "config.json", run_dir / "sft.jsonl"
    config_bytes, log_bytes = config_path.read_bytes(), log_path.read_bytes()
    cfg = _json(config_bytes.decode("utf-8-sig"), str(config_path))
    require(isinstance(cfg, dict), f"{config_path}: expected an object")
    require(cfg.get("training_mode") == mode, f"{config_path}: unexpected model track")
    require(cfg.get("prompt_version") == "contract_v2", f"{config_path}: not the retry prompt")
    require(cfg.get("training_reward_mode") == "correctness_gated", f"{config_path}: not the retry protocol")
    require(type(cfg.get("sft_epochs")) is int and cfg["sft_epochs"] == expected_epochs,
            f"{config_path}: expected {expected_epochs} supervised epochs")
    ops = cfg.get("train_ops")
    require(isinstance(ops, list) and len(ops) == expected_operators and
            all(isinstance(op, str) and op for op in ops) and len(set(ops)) == len(ops),
            f"{config_path}: expected {expected_operators} distinct training operators")
    require(not set(ops).intersection(cfg.get("heldout_ops", []) + cfg.get("development_eval_ops", [])),
            f"{config_path}: training/evaluation operator overlap")
    require(type(cfg.get("seed")) is int, f"{config_path}: missing integer seed")
    lr = _number(cfg.get("sft_learning_rate"), f"{config_path}: sft_learning_rate", positive=True)
    require(re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("model_revision", ""))) is not None,
            f"{config_path}: missing pinned upstream model revision")
    expected_dtype = "fp32" if mode == "full_param_0.49b" else "fp16"
    require(cfg.get("dtype") == expected_dtype, f"{config_path}: unexpected model precision")
    require(log_bytes.endswith(b"\n"), f"{log_path}: final row is not durably terminated")
    lines = log_bytes.decode("utf-8-sig").splitlines()
    require(len(lines) == expected_steps and all(line.strip() for line in lines),
            f"{log_path}: expected exactly {expected_steps} completed rows; found {len(lines)}")
    records = [_json(line, f"{log_path}:{i}") for i, line in enumerate(lines, 1)]
    expected_order = []
    for epoch in range(1, expected_epochs + 1):
        order = list(range(len(ops)))
        random.Random(cfg["seed"] + 1000 * epoch).shuffle(order)
        expected_order.extend((epoch, ops[index]) for index in order)
    per_op_identity: dict[str, tuple[Any, ...]] = {}
    for step, (row, (epoch, op)) in enumerate(zip(records, expected_order), 1):
        context = f"{log_path}:{step}"
        require(isinstance(row, dict), f"{context}: expected an object")
        require(type(row.get("step")) is int and row["step"] == step,
                f"{context}: duplicate, missing or out-of-order supervised step")
        require(type(row.get("epoch")) is int and row["epoch"] == epoch and row.get("op_name") == op,
                f"{context}: operator/epoch disagrees with the configured seeded shuffle")
        _number(row.get("loss"), context + " loss")
        # The trainer checked real nonzero gradients before rounding this scalar
        # to six decimals; a valid small norm can therefore be logged as 0.0.
        _number(row.get("grad_norm"), context + " grad_norm")
        active_lr = _number(row.get("learning_rate"), context + " learning_rate", positive=True)
        require(math.isclose(active_lr, lr, rel_tol=1e-12, abs_tol=0),
                f"{context}: logged learning rate disagrees with sft_learning_rate")
        require(all(type(row.get(key)) is int and row[key] > 0 for key in ("prompt_tokens", "completion_tokens")),
                f"{context}: missing positive token counts")
        require(row.get("chat_template_applied") is True, f"{context}: missing chat-format evidence")
        require(all(re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))) is not None
                    for key in ("prompt_sha256", "source_sha256")), f"{context}: missing prompt/source digest")
        require("seed" not in row or row["seed"] == cfg["seed"], f"{context}: contradictory seed")
        identity = tuple(row[key] for key in ("prompt_tokens", "completion_tokens", "prompt_sha256", "source_sha256"))
        require(op not in per_op_identity or per_op_identity[op] == identity,
                f"{context}: operator target or prompt changed across epochs")
        per_op_identity[op] = identity
    require(Counter((row["epoch"], row["op_name"]) for row in records) ==
            Counter((epoch, op) for epoch in range(1, expected_epochs + 1) for op in ops),
            f"{log_path}: incomplete per-epoch operator coverage")
    means = [math.fsum(float(row["loss"]) for row in records if row["epoch"] == epoch) / len(ops)
             for epoch in range(1, expected_epochs + 1)]
    return TrainingTrack(name, label, run_dir.resolve(), cfg, records, means, lr,
                         hashlib.sha256(log_bytes).hexdigest(), hashlib.sha256(config_bytes).hexdigest())


def load_tracks(repo: Path) -> list[TrainingTrack]:
    tracks = [load_track(repo / "experiments/retry_v2" / name, name, mode, label)
              for name, mode, label in TRACKS]
    require(tracks[0].config["train_ops"] == tracks[1].config["train_ops"],
            "The two model tracks do not use the same ordered eight training operators")
    return tracks


def plot_training(tracks: list[TrainingTrack], out: Path, metadata_path: Path, *, dpi: int = 180) -> dict[str, Any]:
    """Render observed points and epoch means, without fitted lines or intervals."""
    require(len(tracks) == 2, "Exactly two validated model tracks are required")
    require(out.suffix.lower() == ".png", "The output must be a PNG")
    require(100 <= dpi <= 240, "DPI must be between 100 and 240")
    require(out.resolve() != metadata_path.resolve(), "PNG and metadata paths must differ")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    blue, orange, ink = "#16697a", "#bb6520", "#182c3d"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": ink, "ytick.color": ink}):
        # Design at the actual print width: every text element is at least 8.5 pt
        # when placed 6.5 inches wide. Do not shrink to a shorter height limit.
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2), sharey=False)
        fig.subplots_adjust(left=.10, right=.985, top=.73, bottom=.30, wspace=.34)
        fig.suptitle("Supervised completion fitting on training examples", fontsize=11,
                     fontweight="bold", y=.98)
        fig.legend([Line2D([], [], color=blue, marker="o", markersize=3, linewidth=.9),
                    Line2D([], [], color=orange, linewidth=2.1)],
                   ["Logged example NLL", "Epoch mean (8 examples)"],
                   loc="upper center", bbox_to_anchor=(.5, .925), frameon=False, ncol=2, fontsize=8.5)
        for ax, track in zip(axes, tracks):
            losses = [float(row["loss"]) for row in track.rows]
            ax.plot(range(1, EXPECTED_STEPS + 1), losses, color=blue, linewidth=.85,
                    marker="o", markersize=2.9, alpha=.9, zorder=3)
            for epoch, mean in enumerate(track.epoch_means, 1):
                ax.hlines(mean, 1 + (epoch - 1) * 8, epoch * 8, color=orange, linewidth=2.1, zorder=4)
            for boundary in (8.5, 16.5, 24.5):
                ax.axvline(boundary, color="#cbd4db", linewidth=.6, linestyle=(0, (3, 3)), zorder=1)
            ax.set_title(track.label + f"\nSeed {track.config['seed']} | SFT LR {track.learning_rate:g}",
                         fontsize=8.5, pad=10, linespacing=1.5)
            ax.set(xlim=(.5, 32.5), ylim=(0, max(losses) * 1.12),
                   xticks=[1, 8, 16, 24, 32], xlabel="Supervised example / update index",
                   ylabel="Completion NLL (nats/token)")
            ax.grid(axis="y", color="#e4e9ed", linewidth=.6, zorder=0)
            ax.tick_params(labelsize=8.5)
            box = ax.get_position()
            means = [f"{epoch}: {mean:.4f}" for epoch, mean in enumerate(track.epoch_means, 1)]
            fig.text((box.x0 + box.x1) / 2, .18, "Epoch means (nats/token)",
                     ha="center", va="center", fontsize=8.5)
            fig.text((box.x0 + box.x1) / 2, .12,
                     "    ".join(means[:2]) + "\n" + "    ".join(means[2:]),
                     ha="center", va="center", fontsize=8.5, linespacing=1.4)
        fig.text(.5, .043, "Teacher-forced training losses, measured before each update.\n"
                 "Generated-program validity is evaluated separately.",
                 ha="center", va="center", fontsize=8.5, color="#415769", linespacing=1.3)
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = out.with_name("." + out.stem + ".building-" + uuid.uuid4().hex + ".png")
        try:
            fig.savefig(staging, dpi=dpi, facecolor="white", format="png",
                        pil_kwargs={"optimize": True, "compress_level": 9},
                        metadata={"Title": "Observed supervised completion-token NLL",
                                  "Description": "Actual pre-update training losses; not generated-program quality."})
            require(staging.stat().st_size <= MAX_PNG_BYTES,
                    f"PNG exceeds {MAX_PNG_BYTES} bytes; use a lower --dpi")
            staging.replace(out)
        finally:
            plt.close(fig)
            staging.unlink(missing_ok=True)
    report = {
        "schema_version": "kernelbench_sft_training_plot_v1",
        "figure": str(out.resolve()), "bytes": out.stat().st_size,
        "figure_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "metric": "Logged completion-token mean negative log likelihood, nats/token, before the example update",
        "epoch_aggregation": "Arithmetic mean of eight logged per-example means; not token-weighted",
        "ordering_rule": ORDER_RULE,
        "inference_quality_claim": False,
        "layout": {"canvas_inches": [6.5, 4.2], "intended_manuscript_width_inches": 6.5,
                   "minimum_text_size_pt_at_intended_width": 8.5,
                   "required_max_height_inches_for_full_width": 4.2},
        "caption": "Completion-token negative log likelihood during four supervised epochs over eight training operators, separately by model track. Points are logged pre-update per-example means; horizontal segments are arithmetic epoch means. Epoch means average eight examples, not individual tokens. Lower teacher-forced training loss does not establish generated-program validity or held-out quality.",
        "tracks": [dict(name=track.name, label=track.label, sft_log=str(track.run_dir / "sft.jsonl"),
                        sft_log_sha256=track.log_sha256, config_file_sha256=track.config_file_sha256,
                        model_revision=track.config["model_revision"], seed=track.config["seed"],
                        sft_learning_rate=track.learning_rate, train_ops=track.config["train_ops"],
                        epochs=EXPECTED_EPOCHS, completed_steps=len(track.rows),
                        epoch_means_nats_per_token=track.epoch_means,
                        observed_operator_order=[row["op_name"] for row in track.rows]) for track in tracks],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    staging_json = metadata_path.with_name("." + metadata_path.name + ".building-" + uuid.uuid4().hex)
    try:
        staging_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        staging_json.replace(metadata_path)
    finally:
        staging_json.unlink(missing_ok=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=WORKSPACE / "outputs/paper-assets/sft_training.png")
    parser.add_argument("--metadata", type=Path, help="Defaults to the PNG path with a .json suffix")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    tracks = load_tracks(args.repo.resolve())
    report = plot_training(tracks, args.out, args.metadata or args.out.with_suffix(".json"), dpi=args.dpi)
    print(json.dumps({"figure": report["figure"], "bytes": report["bytes"],
                      "epoch_means": {track["name"]: track["epoch_means_nats_per_token"] for track in report["tracks"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
