"""Contract-derived kernel prompt builder and consistent chat-template formatting.

Derives exact `kernel_forward(...)` signatures, argument order, defaults,
canonical input shapes/dtypes, and reference math directly from
`kernels/op_registry.py` so real coding models generate directly callable
kernels without guessing argument order or scalar defaults.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Any

import numpy as np

from kernels.op_registry import OP_REGISTRY

SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer. Return only valid, self-contained Python code "
    "implementing the exact requested `kernel_forward` signature with no prose outside code fences."
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe_operator_contract(op_name: str) -> dict[str, Any]:
    """Inspect OP_REGISTRY[op_name] to extract exact signature, shapes, dtypes, and reference source."""
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unknown operator '{op_name}'. Available: {sorted(OP_REGISTRY)}")
    spec = OP_REGISTRY[op_name]
    sig = inspect.signature(spec.reference_fn)
    params = list(sig.parameters.values())

    # Generate canonical sample inputs deterministically to document exact shapes and dtypes
    rng = np.random.default_rng(0)
    sample_inputs = spec.make_inputs(rng)
    sample_output = spec.reference_fn(*tuple(a.copy() for a in sample_inputs))

    arg_specs: list[str] = []
    sig_parts: list[str] = []
    for idx, p in enumerate(params):
        if idx < len(sample_inputs):
            arr = sample_inputs[idx]
            kind = "integer tensor (KEEP INTEGER)" if np.issubdtype(arr.dtype, np.integer) else "floating tensor"
            arg_specs.append(f"- `{p.name}`: {kind}, canonical shape `{tuple(arr.shape)}`, dtype `{arr.dtype}`")
            sig_parts.append(p.name)
        else:
            default_repr = repr(p.default) if p.default is not inspect.Parameter.empty else "required"
            arg_specs.append(f"- `{p.name}`: scalar parameter, default `{default_repr}`")
            if p.default is not inspect.Parameter.empty:
                sig_parts.append(f"{p.name}={default_repr}")
            else:
                sig_parts.append(p.name)

    try:
        ref_source = textwrap.dedent(inspect.getsource(spec.reference_fn)).strip()
    except OSError:
        ref_source = ""

    return {
        "op_name": spec.name,
        "level": spec.level,
        "description": spec.description,
        "signature": "def kernel_forward(" + ", ".join(sig_parts) + "):",
        "arg_specs": arg_specs,
        "output_shape": tuple(sample_output.shape),
        "output_dtype": str(sample_output.dtype),
        "rtol": spec.rtol,
        "atol": spec.atol,
        "reference_source": ref_source,
    }


def build_prompt(op_name: str, backend: str = "triton") -> str:
    """Build a detailed, contract-derived prompt with exact signature, shapes, dtypes, and math."""
    info = describe_operator_contract(op_name)
    args_block = "\n".join(info["arg_specs"])
    ref_block = f"\nReference mathematical definition (NumPy):\n```python\n{info['reference_source']}\n```\n" if info["reference_source"] else ""

    if backend == "triton":
        constraints = (
            "Requirements:\n"
            "1. Implement a fused OpenAI Triton `@triton.jit` kernel and expose the exact entrypoint:\n"
            f"   `{info['signature']}`\n"
            "2. Preserve exact argument order and scalar defaults. Handle arbitrary leading batch dimensions and non-power-of-2 shapes.\n"
            "3. Call `.contiguous()` on input tensors, accumulate reductions/dot products in `tl.float32`, "
            'pass `input_precision="ieee"` to any `tl.dot(...)`, and cast output back to the floating input dtype.\n'
            "4. Keep integer quantization/index tensors as integer tensors (never cast them to float on host).\n"
            "5. Do NOT mutate input tensors in-place and do NOT call `torch.nn.functional` or forbidden OS/system imports."
        )
    else:
        constraints = (
            "Requirements:\n"
            "1. Implement a pure NumPy kernel exposing the exact entrypoint:\n"
            f"   `{info['signature']}`\n"
            "2. Preserve exact argument order and scalar defaults, and handle stress/edge shapes.\n"
            "3. Do NOT mutate input tensors in-place and do NOT use forbidden OS/system imports."
        )

    return (
        f"Task: Implement `{info['op_name']}` (Level {info['level']}): {info['description']}.\n\n"
        f"Exact entrypoint signature:\n`{info['signature']}`\n\n"
        f"Inputs:\n{args_block}\n"
        f"Expected output: canonical shape `{info['output_shape']}`, dtype `{info['output_dtype']}` "
        f"(fp32 tolerance rtol={info['rtol']}, atol={info['atol']}).\n"
        f"{ref_block}\n"
        f"{constraints}\n"
    )


def format_prompt_for_tokenizer(
    prompt: str,
    tokenizer: Any | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, bool, str]:
    """Apply tokenizer chat template consistently across training and held-out evaluation.

    Returns `(formatted_prompt, chat_template_applied, prompt_sha256)`.
    """
    if (
        tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return str(formatted), True, sha256_text(str(formatted))
    return prompt, False, sha256_text(prompt)
