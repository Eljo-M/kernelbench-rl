from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

EvalStatus = Literal[
    "PASS",
    "COMPILE_ERROR",
    "NUMERICAL_MISMATCH",
    "TIMEOUT",
    "OOM",
    "RUNTIME_ERROR",
    "CHEAT",
]
BackendType = Literal["triton", "cuda", "numpy_dev"]


@dataclass(frozen=True)
class KernelEvalResult:
    run_id: str
    op_name: str
    level: int
    backend: BackendType
    status: EvalStatus
    compiled: bool
    correct: bool
    cheat_detected: bool
    max_abs_err: float
    eager_ms: float
    compile_ms: float
    candidate_ms: float
    speedup_vs_eager: float
    speedup_vs_compile: float
    reward: float
    achieved_tflops: float = 0.0
    arithmetic_intensity: float = 0.0
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "KernelEvalResult":
        return cls(**payload)
