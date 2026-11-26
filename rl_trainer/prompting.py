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


def build_prompt(op_name: str, backend: str = "triton", *,
                 prompt_version: str = "contract_v1", float_dtype: str = "fp32") -> str:
    """Build a detailed, contract-derived prompt with exact signature, shapes, dtypes, and math."""
    if prompt_version == "contract_v2" and backend == "triton":
        return _build_triton_prompt_v2(op_name, float_dtype)
    if prompt_version not in {"contract_v1", "contract_v2"}:
        raise ValueError("Unsupported prompt version")
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


def _build_triton_prompt_v2(op_name: str, float_dtype: str) -> str:
    """State the enforced candidate contract without exposing a target solution.

    The unrelated copy example teaches launch syntax. Operator math is still
    derived only from the public NumPy specification, never baseline sources.
    """
    if float_dtype not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Unsupported floating dtype")
    info = describe_operator_contract(op_name)
    reference_source = reference_with_helpers(OP_REGISTRY[op_name].reference_fn)
    dtype = {"fp32": "torch.float32", "fp16": "torch.float16", "bf16": "torch.bfloat16"}[float_dtype]
    args = "\n".join(s.replace("dtype `float32`", f"dtype `{dtype}`") for s in info["arg_specs"])
    return f'''Write a complete Python module implementing this GPU operation in Triton.
Operation: {info['op_name']}: {info['description']}
Required HOST entrypoint (ordinary Python, never decorate this function with triton.jit):
{info['signature']}
{args}
Output canonical shape: {info['output_shape']}. Floating output must preserve input dtype {dtype}.
Other leading dimensions and non-power-of-two tails are tested. Integer inputs remain integer.

MATHEMATICAL SPECIFICATION ONLY (NumPy notation; do not copy these APIs into your output):
{reference_source}

Implementation contract:
- Begin with: import torch; import triton; import triton.language as tl (on separate lines).
- Define one or more @triton.jit DEVICE functions and launch them using _kernel[grid](tensors, metadata).
- Only the device function computes values: tl.load, tl.arange, tl.sum, tl.max, tl.exp, tl.rsqrt, tl.store.
- Convert loaded values inside JIT with .to(tl.float32); cast stores to output pointer dtype.
- Use tl.dot(a, b, input_precision="ieee") for matrix products. tl.dot requires compatible 2D tiles.
- Host code may allocate with torch.empty/empty_like using device=x.device and dtype=x.dtype, inspect shapes,
  call contiguous/reshape/view, calculate scalar grid sizes, and launch JIT functions. No host tensor math.
- No NumPy imports, torch reductions, torch.nn, torch.softmax, tensor .to/.float/.clone, reference fallback,
  benchmark, print, sample input creation, autotune, classes, or external imports in the returned module.
- There is no tl.meshgrid, tl.warp_id, tl.mma_matrix_shape, triton.current_context or tl.InputPrecision API.
- Use tl.load(ptr + offsets, mask=valid, other=0.0) and tl.store(ptr + offsets, values, mask=valid).
  Pass tensors directly to a launch; do not pass .data_ptr() integers. Mask every tail load and store.
- Honour all scalar parameters and exact argument order/defaults. Write every output element; never mutate inputs.

Unrelated syntax example for copying a vector (implement the requested math, not a copy):
```python
import torch
import triton
import triton.language as tl
@triton.jit
def _copy(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + i, mask=i < N, other=0.0)
    tl.store(Y + i, value, mask=i < N)
def example_copy(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    _copy[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), B=256)
    return y
```
Return exactly one python code fence with a complete module for {info['signature']}.
Do not include the unrelated example or any explanatory prose.
'''


def reference_with_helpers(reference_fn) -> str:
    """Include the mathematical specification's local helper closure.

    Only functions from the reference module are followed. NumPy APIs remain
    specification notation; no candidate or handwritten Triton source is read.
    """
    seen = set()
    sources = []

    def visit(fn):
        if fn in seen:
            return
        seen.add(fn)
        for name in sorted(fn.__code__.co_names):
            helper = fn.__globals__.get(name)
            if inspect.isfunction(helper) and helper.__module__ == reference_fn.__module__:
                visit(helper)
        sources.append(textwrap.dedent(inspect.getsource(fn)).strip())

    visit(reference_fn)
    return "\n\n".join(sources)


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
