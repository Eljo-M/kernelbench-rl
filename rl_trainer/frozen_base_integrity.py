"""Byte-exact frozen-base proofs for supervised QLoRA studies and their resumes."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


HASH_VERSION = "frozen_stored_tensors_v1"


def hash_frozen_base(model: Any, *, chunk_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Hash frozen parameters, all registered buffers, and available quant_state.

    Includes tensor names, shapes, dtypes and exact stored bytes without converting
    NF4 parameters to floating point. Tensor copies to CPU are bounded by
    chunk_bytes; memoryview avoids allocating a second giant Python bytes object.
    Noncontiguous tensors are streamed in logical row-major order. Device names
    are excluded so equivalent CPU/CUDA placement does not change the digest.
    """
    import torch

    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
        raise ValueError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    counts = dict(frozen_parameter_tensors=0, frozen_parameter_bytes=0,
                  buffer_tensors=0, buffer_bytes=0,
                  quantization_state_roots=0, quantization_state_tensors=0,
                  quantization_state_bytes=0, quantization_metadata_values=0)

    def marker(value):
        encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    def chunks(tensor, elements):
        if tensor.is_contiguous():
            flat = tensor.reshape(-1)
            for offset in range(0, flat.numel(), elements):
                yield flat[offset:offset + elements]
        elif tensor.ndim <= 1:
            for offset in range(0, tensor.numel(), elements):
                yield tensor[offset:offset + elements]
        else:
            for row in tensor:
                yield from chunks(row, elements)

    def tensor_value(name, tensor, category):
        if tensor.is_meta:
            raise ValueError(f"Cannot verify meta tensor {name}")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"Unsupported frozen tensor storage for {name}: {tensor.layout}")
        value = tensor.detach()
        marker(dict(kind="tensor", name=name, shape=list(value.shape), dtype=str(value.dtype),
                    bytes=value.numel() * value.element_size()))
        elements = max(1, chunk_bytes // value.element_size())
        for piece in chunks(value, elements):
            packed = piece.to(device="cpu").contiguous()
            # PyTorch can call a length-one strided view contiguous while its
            # final stride remains >1; dtype reinterpretation requires stride1.
            if packed.stride(-1) != 1:
                contiguous = torch.empty((packed.numel(),), dtype=packed.dtype, device="cpu")
                contiguous.copy_(packed)
                packed = contiguous
            raw = packed.view(torch.uint8).numpy()
            digest.update(memoryview(raw).cast("B"))
        counts[category + "_tensors"] += 1
        counts[category + "_bytes"] += value.numel() * value.element_size()

    visiting: set[int] = set()

    def quant_value(name, value):
        if isinstance(value, torch.Tensor):
            tensor_value(name, value, "quantization_state")
            return
        if value is None or isinstance(value, (str, bool, int, float, torch.dtype)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Nonfinite quantization metadata {name}")
            marker(dict(kind="quant_metadata", name=name, type=type(value).__name__,
                        value=str(value) if isinstance(value, torch.dtype) else value))
            counts["quantization_metadata_values"] += 1
            return
        if id(value) in visiting:
            raise ValueError(f"Cyclic quantization state at {name}")
        visiting.add(id(value))
        try:
            if isinstance(value, Mapping):
                if not all(isinstance(key, str) for key in value):
                    raise ValueError(f"Quantization state keys must be strings: {name}")
                marker(dict(kind="mapping", name=name, keys=sorted(value)))
                for key in sorted(value):
                    quant_value(name + "." + key, value[key])
            elif isinstance(value, (tuple, list)):
                marker(dict(kind="sequence", name=name, length=len(value)))
                for index, item in enumerate(value):
                    quant_value(f"{name}[{index}]", item)
            elif hasattr(value, "__dict__"):
                # BitsAndBytes QuantState stores absmax/code/offset/state2 and
                # NF4 type/shape/block metadata here, including nested state.
                quant_value(name, vars(value))
            elif hasattr(value, "as_dict"):
                quant_value(name, value.as_dict(packed=False))
            else:
                raise ValueError(f"Unsupported quantization metadata {name}: {type(value).__name__}")
        finally:
            visiting.remove(id(value))

    marker(dict(hash_version=HASH_VERSION, scope="frozen parameters + all buffers + quant_state"))
    quant_roots: list[tuple[str, Any]] = []
    seen_quant_states: set[int] = set()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if parameter.requires_grad:
            continue
        tensor_value("parameter." + name, parameter, "frozen_parameter")
        state = getattr(parameter, "quant_state", None)
        if state is not None and id(state) not in seen_quant_states:
            seen_quant_states.add(id(state))
            quant_roots.append(("parameter." + name + ".quant_state", state))
    for name, buffer in sorted(model.named_buffers(), key=lambda item: item[0]):
        tensor_value("buffer." + name, buffer, "buffer")
    for name, module in sorted(model.named_modules(), key=lambda item: item[0]):
        state = getattr(module, "quant_state", None)
        if state is not None and id(state) not in seen_quant_states:
            seen_quant_states.add(id(state))
            quant_roots.append(("module." + name + ".quant_state", state))
    for name, state in quant_roots:
        counts["quantization_state_roots"] += 1
        quant_value(name, state)
    if not counts["frozen_parameter_tensors"]:
        raise ValueError("Frozen-base proof requires at least one frozen parameter")
    return dict(hash_version=HASH_VERSION, sha256=digest.hexdigest(), **counts,
                tensor_payload_bytes=sum(counts[key] for key in
                    ("frozen_parameter_bytes", "buffer_bytes", "quantization_state_bytes")),
                buffer_scope="all registered buffers, including nonpersistent buffers",
                quantization_scope="available parameter/module quant_state; nested state and metadata included",
                trainable_parameters_excluded=True, tensor_order="sorted names; logical row-major stored dtype")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _write_exclusive(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


class FrozenBaseVerification:
    """Retain an immutable initial digest and preserve original resume evidence."""

    def __init__(self, model, proof_path, *, config_hash, model_identity, is_resuming=False):
        self.model = model
        self.path = Path(proof_path)
        self.initial_path = self.path.with_name("frozen_base_initial.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.model_identity = model_identity
        self.is_resuming = bool(is_resuming)
        self.before = hash_frozen_base(model)
        self.original_proof_bytes = self.path.read_bytes() if self.path.exists() else None
        if self.is_resuming:
            if not self.initial_path.exists():
                raise ValueError("Resume requires the original frozen_base_initial.json evidence")
            initial = _read_json(self.initial_path)
            if initial.get("config_hash") != config_hash or initial.get("model_identity") != model_identity:
                raise ValueError("Resume frozen-base evidence belongs to a different model/configuration")
            if initial.get("before") != self.before:
                raise ValueError("Frozen base differs from the original pre-SFT digest on resume")
            if self.original_proof_bytes is not None:
                original = _read_json(self.path)
                if original.get("config_hash") != config_hash or original.get("model_identity") != model_identity:
                    raise ValueError("Original frozen-base proof identity mismatch")
                if original.get("verified") is not True or original.get("before") != self.before or original.get("after") != self.before:
                    raise ValueError("Original frozen-base proof is invalid or failed")
        else:
            if self.path.exists() or self.initial_path.exists():
                raise ValueError("Refusing to overwrite previous frozen-base evidence; archive the prior study first")
            _write_exclusive(self.initial_path, dict(
                proof_version=1, phase="before_sft_and_grpo", config_hash=config_hash,
                model_identity=model_identity, before=self.before, is_quality_metric=False,
            ))
        print(json.dumps(dict(event="frozen_base_before_verified", sha256=self.before["sha256"],
                              tensor_payload_bytes=self.before["tensor_payload_bytes"],
                              quantization_state_roots=self.before["quantization_state_roots"],
                              is_resuming=self.is_resuming)), flush=True)

    def finish(self, *, training_completed: bool):
        after = hash_frozen_base(self.model)
        verified = after == self.before
        if self.original_proof_bytes is not None and self.path.read_bytes() != self.original_proof_bytes:
            raise RuntimeError("Original frozen-base proof changed during resume")
        target = self.path
        if self.original_proof_bytes is not None:
            target = self.path.with_name(self.path.stem + ".resume-" + uuid.uuid4().hex + self.path.suffix)
        payload = dict(proof_version=1, config_hash=self.config_hash, model_identity=self.model_identity,
                       before=self.before, after=after, verified=verified, training_completed=bool(training_completed),
                       phase="before_and_after_current_sft_grpo_invocation", resumed=self.is_resuming,
                       is_quality_metric=False, initial_evidence_file=self.initial_path.name,
                       scope="frozen parameter bytes/dtype/shapes, registered buffers, available NF4 quantization state")
        if self.original_proof_bytes is not None:
            payload.update(original_proof_file=self.path.name,
                           original_proof_sha256=hashlib.sha256(self.original_proof_bytes).hexdigest(),
                           original_proof_preserved=True)
        _write_exclusive(target, payload)
        print(json.dumps(dict(event="frozen_base_after_verified", verified=verified,
                              sha256=after["sha256"], proof_path=str(target))), flush=True)
        if not verified:
            raise RuntimeError("Frozen base parameters, buffers or quantization state changed during training")
        return payload
