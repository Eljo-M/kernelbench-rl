"""Bounded experiment configuration for full-parameter 0.49B and 7B QLoRA GRPO runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from kernels.op_registry import OP_REGISTRY
from kernels.torch_references import TORCH_REFERENCES

BackendMode = Literal["numpy_dev", "triton"]
FloatDtype = Literal["fp32", "bf16", "fp16"]
TrainingMode = Literal["full_param_0.49b", "qlora_7b_nf4"]

DEFAULT_TRAIN_OPS: tuple[str, ...] = (
    "rmsnorm_2048",
    "layernorm_2048",
    "swiglu_4096",
    "fused_add_rmsnorm_2048",
    "fused_softmax_2048",
    "log_softmax_2048",
    "rope_fwd_128",
    "matmul_bias_gelu_512",
)

DEFAULT_HELDOUT_OPS: tuple[str, ...] = (
    "rmsnorm_4096",
    "int8_weight_only_gemm_512",
    "scaled_dot_product_attn_128",
)


@dataclass
class ExperimentConfig:
    """Validated configuration with complete config_hash covering optimizer, master weights, KL, clipping, and sampling."""

    model_name_or_path: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    model_revision: str = "main"
    training_mode: TrainingMode = "full_param_0.49b"
    backend: BackendMode = "triton"
    dtype: FloatDtype = "bf16"
    train_ops: list[str] = field(default_factory=lambda: list(DEFAULT_TRAIN_OPS))
    heldout_ops: list[str] = field(default_factory=lambda: list(DEFAULT_HELDOUT_OPS))
    seed: int = 42
    heldout_seed_offset: int = 50_000
    selection_seed_offset: int = 100_000
    independent_seed_offset: int = 10_000
    group_size: int = 2
    heldout_best_of_k: int = 8
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    learning_rate: float = 1e-6
    optimizer_name: str = "adamw"
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0
    max_steps: int = 16
    checkpoint_every: int = 4
    gradient_checkpointing: bool = True
    cpu_master_adamw: bool = True  # FP32 CPU master weights + AdamW for BF16 full-param on 8 GB VRAM
    offload_ref_to_cpu: bool = True
    offload_policy_during_grading: bool = True
    sequential_generation: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    worker_timeout_secs: float = 45.0
    warmup_iters: int = 10
    bench_iters: int = 50
    checkpoint_dir: str = "checkpoints/qwen05b_grpo"
    resume_from: str = ""
    output_jsonl: str = "eval/results/grpo_rollouts.jsonl"
    prompt_version: str = "contract_v1"
    training_reward_mode: str = "legacy_shaped"
    sft_epochs: int = 0
    sft_learning_rate: float = 1e-4
    sft_max_tokens: int = 4096
    sft_output_jsonl: str = ""
    onpolicy_logp_tolerance: float = 0.001
    development_eval_ops: list[str] = field(default_factory=list)
    qlora_cpu_offload: bool = False

    @classmethod
    def preset_t4_7b_qlora(cls, **overrides) -> "ExperimentConfig":
        """Free Colab Tesla T4 (14.56 GB, sm75) preset: 7B QLoRA NF4 with fp16 compute."""
        params = dict(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            training_mode="qlora_7b_nf4",
            backend="triton",
            dtype="fp16",
            group_size=2,
            heldout_best_of_k=8,
            max_new_tokens=512,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            learning_rate=5e-5,
            kl_coef=0.04,
            clip_eps=0.2,
            gradient_checkpointing=True,
            cpu_master_adamw=False,  # LoRA adapters are already small FP32 tensors
            offload_ref_to_cpu=False,
            offload_policy_during_grading=False,
            sequential_generation=True,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            checkpoint_dir="checkpoints/qwen7b_qlora_t4",
            output_jsonl="eval/results/qwen7b_qlora_t4_rollouts.jsonl",
        )
        params.update(overrides)
        cfg = cls(**params)
        cfg.validate()
        return cfg

    def sampling_config_dict(self) -> dict[str, float | int | bool]:
        return {
            "do_sample": True,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "max_new_tokens": int(self.max_new_tokens),
        }

    def validate(self, *, strict_on_policy_sampling: bool = True) -> None:
        if self.prompt_version not in {"contract_v1", "contract_v2"}:
            raise ValueError("Unsupported prompt_version")
        if self.training_reward_mode not in {"legacy_shaped", "correctness_gated"}:
            raise ValueError("Unsupported training_reward_mode")
        if isinstance(self.sft_epochs, bool) or not isinstance(self.sft_epochs, int) or self.sft_epochs < 0:
            raise ValueError("sft_epochs must be a nonnegative integer")
        if not math.isfinite(self.sft_learning_rate) or self.sft_learning_rate <= 0 or self.sft_max_tokens <= 0:
            raise ValueError("Invalid supervised-warmup learning rate or token limit")
        if not math.isfinite(self.onpolicy_logp_tolerance) or self.onpolicy_logp_tolerance <= 0:
            raise ValueError("onpolicy_logp_tolerance must be finite and positive")
        if self.qlora_cpu_offload and self.training_mode != "qlora_7b_nf4":
            raise ValueError("QLoRA CPU placement applies only to qlora_7b_nf4")
        if len(set(self.train_ops)) != len(self.train_ops) or len(set(self.heldout_ops + self.development_eval_ops)) != len(self.heldout_ops + self.development_eval_ops):
            raise ValueError("Operator splits must not contain duplicates")
        if self.optimizer_name not in {"adamw", "adafactor"}:
            raise ValueError("optimizer_name must be adamw or adafactor")
        if self.optimizer_name == "adafactor" and (self.cpu_master_adamw or self.training_mode != "full_param_0.49b"):
            raise ValueError("Adafactor requires full_param_0.49b and cpu_master_adamw=False")
        if self.training_mode not in {"full_param_0.49b", "qlora_7b_nf4"}:
            raise ValueError(f"Unsupported training_mode '{self.training_mode}'.")
        if self.backend not in {"numpy_dev", "triton"}:
            raise ValueError(f"Unsupported backend '{self.backend}'. Expected 'numpy_dev' or 'triton'.")
        if self.dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported dtype '{self.dtype}'. Expected 'fp32', 'bf16', or 'fp16'.")
        if not self.train_ops or not self.heldout_ops:
            raise ValueError("train_ops and heldout_ops must both be non-empty.")

        for op in self.train_ops + self.heldout_ops + self.development_eval_ops:
            if op not in OP_REGISTRY:
                raise ValueError(f"Unknown operator '{op}' not in OP_REGISTRY.")
            if self.backend == "triton" and op not in TORCH_REFERENCES:
                raise ValueError(f"Operator '{op}' has no native PyTorch reference for backend='triton'.")

        overlap = sorted(set(self.train_ops) & set(self.heldout_ops + self.development_eval_ops))
        if overlap:
            raise ValueError(f"Data leakage: train_ops and heldout_ops overlap on {overlap}.")

        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}.")
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for GRPO normalization, got {self.group_size}.")
        if self.heldout_best_of_k < 1:
            raise ValueError(f"heldout_best_of_k must be >= 1, got {self.heldout_best_of_k}.")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}.")
        if not (math.isfinite(self.temperature) and self.temperature > 0.0):
            raise ValueError(f"temperature must be finite > 0, got {self.temperature}.")
        if not (math.isfinite(self.top_p) and 0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}.")
        if strict_on_policy_sampling and (self.temperature != 1.0 or self.top_p != 1.0 or self.top_k != 0):
            raise ValueError(
                "On-policy GRPO requires temperature=1.0, top_p=1.0, top_k=0 so raw log_softmax "
                "matches the actual sampling distribution."
            )
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0.0):
            raise ValueError(f"learning_rate must be finite > 0, got {self.learning_rate}.")
        if not (math.isfinite(self.kl_coef) and self.kl_coef >= 0.0):
            raise ValueError(f"kl_coef must be finite >= 0, got {self.kl_coef}.")
        if not (math.isfinite(self.clip_eps) and 0.0 < self.clip_eps < 1.0):
            raise ValueError(f"clip_eps must be in (0, 1), got {self.clip_eps}.")
        if not (math.isfinite(self.max_grad_norm) and self.max_grad_norm > 0.0):
            raise ValueError(f"max_grad_norm must be finite > 0, got {self.max_grad_norm}.")
        if self.max_steps <= 0 or self.checkpoint_every <= 0:
            raise ValueError("max_steps and checkpoint_every must be > 0.")
        if not (math.isfinite(self.worker_timeout_secs) and self.worker_timeout_secs > 0.0):
            raise ValueError(f"worker_timeout_secs must be finite > 0, got {self.worker_timeout_secs}.")
        if self.warmup_iters < 0 or self.bench_iters <= 0:
            raise ValueError("warmup_iters must be >= 0 and bench_iters must be > 0.")
        if self.training_mode == "qlora_7b_nf4":
            if self.lora_r <= 0 or self.lora_alpha <= 0:
                raise ValueError("lora_r and lora_alpha must be > 0 for QLoRA.")
            if self.lora_dropout != 0.0 and strict_on_policy_sampling:
                raise ValueError("lora_dropout must be 0.0 for deterministic on-policy GRPO logp matching.")
            if not self.lora_target_modules:
                raise ValueError("lora_target_modules must be non-empty for QLoRA.")

    def config_hash(self) -> str:
        """Stable SHA-256 prefix over ALL training, optimizer, master-weight, KL, clipping, and sampling fields."""
        key_fields = asdict(self)
        for key in ("max_steps", "checkpoint_every", "checkpoint_dir", "resume_from", "output_jsonl"):
            key_fields.pop(key)
        # Keep historical, zero-warmup experiment identities stable. New active
        # training phases and prompt/reward versions are always hashed.
        if self.prompt_version == "contract_v1":
            key_fields.pop("prompt_version")
        if self.training_reward_mode == "legacy_shaped":
            key_fields.pop("training_reward_mode")
        if self.sft_epochs == 0:
            for key in ("sft_epochs", "sft_learning_rate", "sft_max_tokens", "sft_output_jsonl"):
                key_fields.pop(key)
        if self.onpolicy_logp_tolerance == 0.001:
            key_fields.pop("onpolicy_logp_tolerance")
        if not self.development_eval_ops:
            key_fields.pop("development_eval_ops")
        if not self.qlora_cpu_offload:
            key_fields.pop("qlora_cpu_offload")
        raw = json.dumps(key_fields, sort_keys=True, allow_nan=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = cls(**data)
        cfg.validate()
        return cfg
