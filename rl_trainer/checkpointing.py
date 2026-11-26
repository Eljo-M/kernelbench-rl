"""Resumable checkpoint manager supporting full-parameter and in-place PEFT adapter resume,
optimizer step tracking, trainable-weight checksums, and full RNG restoration.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class CheckpointState:
    step: int
    op_cursor: int
    seed: int
    config_hash: str
    best_mean_reward: float
    total_rollouts: int
    optimizer_step_count: int = 0
    trainable_weights_sha256: str = ""
    training_mode: str = "full_param_0.49b"
    model_weights_file: str = ""
    adapter_dir: str = ""
    optimizer_state_file: str = ""
    rng_state_file: str = ""
    is_fixture: bool = False


def compute_trainable_weights_sha256(params: Iterable[Any]) -> str:
    """Deterministic SHA-256 over all trainable parameter tensors (for objective update proof)."""
    h = hashlib.sha256()
    count = 0
    for p in params:
        if not getattr(p, "requires_grad", False):
            continue
        arr = p.detach().float().cpu().contiguous().numpy()
        h.update(arr.tobytes())
        count += 1
    return h.hexdigest() if count > 0 else ""


def snapshot_trainable_params(params: Iterable[Any]) -> list[Any]:
    """Clone trainable parameters on CPU for post-step L2 delta verification."""
    return [p.detach().float().cpu().clone() for p in params if getattr(p, "requires_grad", False)]


def compute_weight_l2_delta(before_snaps: list[Any], params: Iterable[Any]) -> float:
    """Compute exact L2 norm of parameter update `||theta_after - theta_before||_2`."""
    after_list = [p.detach().float().cpu() for p in params if getattr(p, "requires_grad", False)]
    if len(before_snaps) != len(after_list) or not before_snaps:
        return 0.0
    sq_sum = 0.0
    for b, a in zip(before_snaps, after_list):
        diff = (a - b).numpy()
        sq_sum += float(np.sum(diff * diff, dtype=np.float64))
    return float(np.sqrt(sq_sum))


def _capture_python_numpy_rng() -> dict[str, Any]:
    py_ver, py_inner, py_gauss = random.getstate()
    np_alg, np_keys, np_pos, np_has_gauss, np_cached_gauss = np.random.get_state()
    return {
        "python": [py_ver, list(py_inner), py_gauss],
        "numpy": [str(np_alg), np_keys.tolist(), int(np_pos), int(np_has_gauss), float(np_cached_gauss)],
    }


def _restore_python_numpy_rng(payload: dict[str, Any]) -> None:
    py_ver, py_inner, py_gauss = payload["python"]
    random.setstate((int(py_ver), tuple(py_inner), py_gauss))
    np_alg, np_keys, np_pos, np_has_gauss, np_cached_gauss = payload["numpy"]
    np.random.set_state(
        (str(np_alg), np.asarray(np_keys, dtype=np.uint32), int(np_pos), int(np_has_gauss), float(np_cached_gauss))
    )


def verify_compatible_run_paths(
    checkpoint_dir: Path,
    output_jsonl: Path,
    expected_config_hash: str,
    *,
    is_resuming: bool,
    overwrite: bool = False,
) -> None:
    """Refuse overwriting an incompatible checkpoint directory or appending to incompatible JSONL."""
    latest_ptr = checkpoint_dir / "latest.json"
    if latest_ptr.exists() and not is_resuming and not overwrite:
        info = json.loads(latest_ptr.read_text(encoding="utf-8"))
        prev_meta = checkpoint_dir / info["latest_step_dir"] / "checkpoint_state.json"
        if prev_meta.exists():
            prev = json.loads(prev_meta.read_text(encoding="utf-8"))
            if prev.get("config_hash") != expected_config_hash:
                raise ValueError(
                    f"Refusing to overwrite incompatible checkpoint_dir {checkpoint_dir}: "
                    f"existing config_hash={prev.get('config_hash')!r} != {expected_config_hash!r}."
                )
        raise ValueError(
            f"Checkpoint directory {checkpoint_dir} already contains a run. "
            "Pass --resume-from to continue or --overwrite to start fresh."
        )

    if output_jsonl.exists():
        lines = [ln.strip() for ln in output_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines and not overwrite:
            if not is_resuming:
                raise ValueError(
                    f"Output JSONL already exists: {output_jsonl}. Pass --resume-from or --overwrite."
                )
            if any(json.loads(line).get("config_hash") != expected_config_hash for line in lines):
                raise ValueError(f"Refusing to append to incompatible JSONL {output_jsonl}.")


def reconcile_resume_log(output_jsonl: Path, checkpoint_step: int) -> Path | None:
    """Archive rollouts after the restored checkpoint before replaying those steps.

    A crash can leave logged steps beyond the last durable weights/optimizer.
    Keeping them in the resumed stream would duplicate steps and mix policies.
    The complete original log is retained before replacing the active file.
    """
    if not output_jsonl.exists():
        return None
    lines = output_jsonl.read_text(encoding="utf-8").splitlines(keepends=True)
    keep = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("Resume logs require a positive integer step on every record")
        if step <= checkpoint_step:
            keep.append(line)
    if len(keep) == sum(bool(line.strip()) for line in lines):
        return None
    archive = output_jsonl.with_name(output_jsonl.name + ".uncheckpointed-" + uuid.uuid4().hex + ".jsonl")
    with archive.open("x", encoding="utf-8", newline="") as fh:
        fh.writelines(lines)
    temporary = output_jsonl.with_name(output_jsonl.name + ".resume-" + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as fh:
        fh.writelines(keep)
    temporary.replace(output_jsonl)
    return archive


def _load_adapter_weights_in_place(model: Any, adapter_path: Path, map_location: str) -> None:
    """Load saved LoRA/PEFT adapter tensors IN-PLACE into an existing trainable model.

    Preserves exact parameter object identity (so an existing AdamW optimizer stays
    bound to the exact tensors), keeps `requires_grad=True` on adapters, and leaves
    the frozen base weights untouched so `disable_adapter()` keeps the original reference.
    """
    try:
        from peft import load_peft_weights, set_peft_model_state_dict  # type: ignore

        weights = load_peft_weights(str(adapter_path), device=map_location)
        set_peft_model_state_dict(model, weights, adapter_name="default")
        return
    except ImportError:
        pass

    # Fallback for custom/mock PEFT-like models or non-PEFT trainable adapter dicts
    if hasattr(model, "load_adapter_in_place"):
        model.load_adapter_in_place(str(adapter_path))
        return
    if hasattr(model, "load_adapter"):
        model.load_adapter(str(adapter_path), adapter_name="default")
        return

    import torch  # type: ignore

    for fname in ("adapter_model.bin", "adapter_state.pt"):
        fpath = adapter_path / fname
        if fpath.exists():
            sd = torch.load(fpath, map_location=map_location, weights_only=True)
            model.load_state_dict(sd, strict=False)
            return
    raise FileNotFoundError(f"No loadable adapter weights found in {adapter_path}")


def save_checkpoint(
    checkpoint_dir: Path,
    state: CheckpointState,
    *,
    model: Any = None,
    optimizer: Any = None,
    adapter_only: bool = False,
    save_rng: bool = True,
) -> Path:
    """Publish a complete checkpoint directory, then atomically replace its pointer."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir = checkpoint_dir / f"step_{state.step:05d}"
    if final_dir.exists():
        raise FileExistsError(f"Checkpoint already exists: {final_dir}; use a new run directory")
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=checkpoint_dir))
    completed = _write_checkpoint_contents(staging, state, model=model, optimizer=optimizer,
                                           adapter_only=adapter_only, save_rng=save_rng)
    completed.replace(final_dir)
    (staging / "latest.json").replace(checkpoint_dir / "latest.json")
    staging.rmdir()
    return final_dir


def _write_checkpoint_contents(
    checkpoint_dir: Path, state: CheckpointState, *, model: Any = None,
    optimizer: Any = None, adapter_only: bool = False, save_rng: bool = True,
) -> Path:
    """Write into staging; incomplete directories are never advertised as latest."""
    step_dir = checkpoint_dir / f"step_{state.step:05d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    model_file = state.model_weights_file
    adapter_subdir = state.adapter_dir
    opt_file = state.optimizer_state_file
    rng_file = state.rng_state_file
    weights_sha = state.trainable_weights_sha256

    if model is not None:
        if hasattr(model, "parameters"):
            try:
                computed = compute_trainable_weights_sha256(model.parameters())
                if computed:
                    weights_sha = computed
            except Exception:
                pass

        if adapter_only:
            if hasattr(model, "save_pretrained"):
                ad_path = step_dir / "adapter"
                model.save_pretrained(str(ad_path))
                adapter_subdir = ad_path.name
            else:
                import torch  # type: ignore

                trainable = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
                ad_file = step_dir / "adapter_state.pt"
                torch.save(trainable, ad_file)
                model_file = ad_file.name
        else:
            import torch  # type: ignore

            model_path = step_dir / "model_state.pt"
            torch.save(model.state_dict(), model_path)
            model_file = model_path.name

    if optimizer is not None:
        import torch  # type: ignore

        opt_path = step_dir / "optimizer_state.pt"
        torch.save(optimizer.state_dict(), opt_path)
        opt_file = opt_path.name

    if save_rng:
        rng_payload = _capture_python_numpy_rng()
        try:
            import torch  # type: ignore

            torch_rng_path = step_dir / "torch_rng.pt"
            torch_rng: dict[str, Any] = {"cpu": torch.get_rng_state()}
            if torch.cuda.is_available():
                torch_rng["cuda"] = torch.cuda.get_rng_state_all()
            torch.save(torch_rng, torch_rng_path)
            rng_payload["torch_rng_file"] = torch_rng_path.name
        except ImportError:
            rng_payload["torch_rng_file"] = ""

        rng_path = step_dir / "rng_state.json"
        rng_tmp = step_dir / "rng_state.json.tmp"
        rng_tmp.write_text(json.dumps(rng_payload, allow_nan=False) + "\n", encoding="utf-8")
        rng_tmp.replace(rng_path)
        rng_file = rng_path.name

    final_state = CheckpointState(
        step=state.step,
        op_cursor=state.op_cursor,
        seed=state.seed,
        config_hash=state.config_hash,
        best_mean_reward=float(state.best_mean_reward),
        total_rollouts=int(state.total_rollouts),
        optimizer_step_count=int(state.optimizer_step_count),
        trainable_weights_sha256=weights_sha,
        training_mode=state.training_mode,
        model_weights_file=model_file,
        adapter_dir=adapter_subdir,
        optimizer_state_file=opt_file,
        rng_state_file=rng_file,
        is_fixture=state.is_fixture,
    )

    meta_path = step_dir / "checkpoint_state.json"
    tmp_path = step_dir / "checkpoint_state.json.tmp"
    tmp_path.write_text(json.dumps(asdict(final_state), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp_path.replace(meta_path)

    latest_ptr = checkpoint_dir / "latest.json"
    latest_tmp = checkpoint_dir / "latest.json.tmp"
    latest_tmp.write_text(
        json.dumps({"latest_step_dir": step_dir.name, "step": state.step}, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    latest_tmp.replace(latest_ptr)
    return step_dir


def resolve_checkpoint_dir(resume_from: str | Path) -> Path:
    p = Path(resume_from)
    if (p / "checkpoint_state.json").exists():
        return p
    latest_ptr = p / "latest.json"
    if latest_ptr.exists():
        info = json.loads(latest_ptr.read_text(encoding="utf-8"))
        target = p / info["latest_step_dir"]
        if (target / "checkpoint_state.json").exists():
            return target
    raise FileNotFoundError(f"No valid checkpoint_state.json found under {p}")


def load_checkpoint(
    resume_from: str | Path,
    *,
    expected_config_hash: str | None = None,
    model: Any = None,
    optimizer: Any = None,
    restore_rng: bool = True,
    map_location: str = "cpu",
) -> CheckpointState:
    """Load checkpoint metadata and restore model/adapters in-place, optimizer, and RNG states."""
    step_dir = resolve_checkpoint_dir(resume_from)
    data = json.loads((step_dir / "checkpoint_state.json").read_text(encoding="utf-8"))
    state = CheckpointState(**data)

    if expected_config_hash is not None and state.config_hash != expected_config_hash:
        raise ValueError(
            f"Checkpoint config_hash mismatch: checkpoint has '{state.config_hash}', "
            f"current config has '{expected_config_hash}'."
        )

    if model is not None:
        if state.adapter_dir:
            _load_adapter_weights_in_place(model, step_dir / state.adapter_dir, map_location=map_location)
        elif state.model_weights_file:
            import torch  # type: ignore

            sd = torch.load(step_dir / state.model_weights_file, map_location=map_location, weights_only=True)
            strict = state.training_mode != "qlora_7b_nf4"
            model.load_state_dict(sd, strict=strict)

    if optimizer is not None and state.optimizer_state_file:
        import torch  # type: ignore

        optimizer.load_state_dict(torch.load(step_dir / state.optimizer_state_file, map_location=map_location, weights_only=True))

    if model is not None and state.trainable_weights_sha256 and hasattr(model, "parameters"):
        actual_sha = compute_trainable_weights_sha256(model.parameters())
        if actual_sha and actual_sha != state.trainable_weights_sha256:
            raise ValueError("Restored trainable weights do not match the checkpoint checksum")

    if restore_rng and state.rng_state_file:
        rng_payload = json.loads((step_dir / state.rng_state_file).read_text(encoding="utf-8"))
        _restore_python_numpy_rng(rng_payload)
        torch_rng_file = rng_payload.get("torch_rng_file", "")
        if torch_rng_file:
            try:
                import torch  # type: ignore

                torch_rng = torch.load(step_dir / torch_rng_file, map_location="cpu", weights_only=True)
                torch.set_rng_state(torch_rng["cpu"])
                if "cuda" in torch_rng and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(torch_rng["cuda"])
            except ImportError:
                pass
    return state
