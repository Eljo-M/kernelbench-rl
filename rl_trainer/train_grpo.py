"""Execution-feedback search and single-process GRPO for the NumPy development verifier.

The dry run uses fixed fixtures and never updates model weights. The optional
Transformers mode samples and updates the same model. vLLM mode is search only;
it never presents a serving endpoint as an automatically synchronized policy.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from kernels.op_registry import OP_REGISTRY
from rl_trainer.grpo_reward import compute_grpo_advantages
from sandbox_runner.verifier import verify_candidate_source


@dataclass(frozen=True)
class RunConfig:
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    group_size: int = 8
    seed: int = 42
    workers: int = 2
    timeout_secs: float = 30.0
    repair_turns: int = 0
    max_new_tokens: int = 512
    steps: int = 1
    learning_rate: float = 1e-6
    clip: float = 0.2
    beta: float = 0.04

    def validate(self) -> None:
        for name in ("group_size", "workers", "max_new_tokens", "steps"):
            if getattr(self, name) < (2 if name == "group_size" else 1):
                raise ValueError(f"{name} must be at least {2 if name == 'group_size' else 1}")
        if not 0 <= self.repair_turns <= 3:
            raise ValueError("repair_turns must be between zero and three")
        if not math.isfinite(self.timeout_secs) or self.timeout_secs <= 0:
            raise ValueError("timeout_secs must be finite and positive")
        if not 0 < self.clip < 1 or not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Invalid clipping or KL coefficient")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.seed < 0 or self.seed >= 2**31:
            raise ValueError("seed must be in [0, 2**31)")


def build_prompt(op_name: str, target: str = "triton") -> str:
    spec = OP_REGISTRY[op_name]
    if target not in {"triton", "numpy_dev"}:
        raise ValueError("Unsupported generation target")
    inputs = spec.make_inputs(__import__("numpy").random.default_rng(0))
    signature = ", ".join(f"arg{i}: shape={x.shape}, dtype={x.dtype}" for i, x in enumerate(inputs))
    language = "NumPy" if target == "numpy_dev" else "Triton"
    return (
        f"Implement {spec.name}: {spec.description}.\n"
        f"Inputs in argument order: {signature}. Tolerances: rtol={spec.rtol}, atol={spec.atol}.\n"
        f"Return only Python source using {language}, exposing def kernel_forward(...). "
        "Return one array with the same semantics as the named operation. "
        "Do not read files, access network, call subprocesses, import the reference registry, "
        "or delegate to torch.nn.functional."
    )


def extract_source(text: str) -> str:
    """Use a single fenced code block, otherwise treat the full response as source."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if len(blocks) > 1:
        raise ValueError("Expected one code block; received several")
    source = blocks[0] if blocks else text
    if not source.strip():
        raise ValueError("Model returned empty source")
    return source.strip() + "\n"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize the complete group before appending, preventing partial invalid groups.
    data = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()


class Generator(Protocol):
    async def generate(self, prompt: str, count: int, seed: int) -> list[str]: ...


class VLLMGenerator:
    """Concurrent calls to a user-managed vLLM Completions endpoint.

    Endpoint protocol: https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/
    Credentials stay in memory and are never written to rollout/configuration logs.
    """

    def __init__(self, base_url: str, model: str, *, max_tokens=512, request_timeout=120.0, concurrency=2):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use an http(s) base URL without credentials, query, or fragment")
        if concurrency < 1 or max_tokens < 1 or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("Invalid generation limits")
        self.url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.semaphore = asyncio.Semaphore(concurrency)

    def _request(self, prompt: str, seed: int) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "n": 1, "max_tokens": self.max_tokens,
                              "temperature": 1.0, "top_p": 1.0, "seed": seed}).encode()
        headers = {"Content-Type": "application/json"}
        if os.environ.get("VLLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
        request = urllib.request.Request(self.url, data=payload, headers=headers)
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            payload = json.loads(response.read(8 * 1024 * 1024))
        choices = payload.get("choices", [])
        if len(choices) != 1 or not isinstance(choices[0].get("text"), str):
            raise ValueError("Malformed completion response")
        return choices[0]["text"]

    async def generate(self, prompt: str, count: int, seed: int) -> list[str]:
        async def one(index):
            async with self.semaphore:
                return await asyncio.to_thread(self._request, prompt, seed + index)
        return list(await asyncio.gather(*(one(index) for index in range(count))))


def grade_sources_sync(op_name: str, sources: list[str], *, group_id: str, config: RunConfig, turn=0) -> list[dict]:
    """Bound concurrency; each verifier call owns a separately timeout-bounded worker."""
    config.validate()
    def one(item):
        index, source = item
        result = verify_candidate_source(
            op_name, source, run_id=f"{group_id}-candidate-{index}-turn-{turn}",
            seed=config.seed, timeout_secs=config.timeout_secs,
        )
        row = result.to_dict()
        row.update(group_id=group_id, candidate_index=index, repair_turn=turn, seed=config.seed,
                   source_sha256=hashlib.sha256(source.encode()).hexdigest(), source=source,
                   measurement_kind="cpu_development", target="numpy_dev")
        return row

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        rows = list(executor.map(one, enumerate(sources)))
    advantages = compute_grpo_advantages([row["reward"] for row in rows])
    for row, advantage in zip(rows, advantages):
        row["grpo_advantage"] = advantage
    return rows


async def grade_sources(op_name: str, sources: list[str], *, group_id: str, config: RunConfig, turn=0) -> list[dict]:
    """Async adapter for remote generation/search; local-only callers need no event loop."""
    return await asyncio.to_thread(grade_sources_sync, op_name, sources, group_id=group_id, config=config, turn=turn)


async def run_search(generator: Generator, op_names: list[str], out_path: Path, config: RunConfig) -> None:
    """Best-of-group search with optional repair. No policy updates are performed."""
    config.validate()
    _write_json(out_path.with_suffix(".config.json"), {"mode": "search", "policy_updated": False,
                                                     "config": asdict(config), "operators": op_names})
    for group_index, op_name in enumerate(op_names):
        prompt = build_prompt(op_name, "numpy_dev")
        responses = await generator.generate(prompt, config.group_size, config.seed + group_index * config.group_size)
        if len(responses) != config.group_size:
            raise ValueError("Generator returned an incorrect group size")
        sources = []
        for text in responses:
            try:
                sources.append(extract_source(text))
            except ValueError:
                # Empty/malformed model output counts as a failed program, not a dropped trial.
                sources.append(text)
        group_id = f"search-{group_index}-{op_name}"
        rows = await grade_sources(op_name, sources, group_id=group_id, config=config)
        for row in rows:
            row.update(method="best_of_n", model=config.model, group_size=config.group_size, policy_updated=False)
        _append_rows(out_path, rows)
        for turn in range(1, config.repair_turns + 1):
            failed = [index for index, row in enumerate(rows) if not row["correct"]]
            if not failed:
                break
            repair_responses = await asyncio.gather(*(
                generator.generate(prompt + "\nPrevious source:\n" + sources[index] +
                                   "\nVerifier feedback:\n" + rows[index]["status"] + ": " + rows[index].get("error_message", "")[:2000] +
                                   "\nReturn corrected source only.", 1, config.seed + 100000 * turn + index)
                for index in failed
            ))
            for index, response in zip(failed, repair_responses):
                if len(response) != 1:
                    raise ValueError("Repair generator returned an incorrect group size")
                try:
                    sources[index] = extract_source(response[0])
                except ValueError:
                    sources[index] = response[0]
            # Regrade the full group with the same seed so normalized advantages remain comparable.
            rows = await grade_sources(op_name, sources, group_id=group_id, config=config, turn=turn)
            for row in rows:
                row.update(method="best_of_n_repair", model=config.model, group_size=config.group_size, policy_updated=False)
            _append_rows(out_path, rows)


DRY_SOURCES = [
    "import numpy as np\ndef kernel_forward(x, w, eps=1e-6):\n    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)) * w\n",
    "import numpy as np\ndef kernel_forward(x, w, eps=1e-6):\n    return x * 0.0\n",
    "import torch.nn.functional as F\ndef kernel_forward(x, w): return x\n",
    "def kernel_forward(x, w):\n    return (((x\n",
]


def run_dry_loop(out_path: Path) -> None:
    """Four labeled CPU fixtures validate grading/logging; this is not a training run."""
    config = RunConfig(group_size=4)
    rows = grade_sources_sync("rmsnorm_2048", DRY_SOURCES, group_id="dry-rollout", config=config)
    for row in rows:
        row.update(method="dry_run_fixture", fixture=True, policy_updated=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    _write_json(out_path.with_suffix(".config.json"), {"mode": "dry_run_fixture", "policy_updated": False, "config": asdict(config)})
    print(f"CPU fixture verification completed: {len(rows)} rollouts written to {out_path}")


def run_training(op_names: list[str], out_path: Path, checkpoint_dir: Path, config: RunConfig, *, device="cuda") -> None:
    """Single-process full-parameter GRPO, using the same policy for generation/update.

    Optional dependencies: torch and transformers. This path intentionally targets
    NumPy development operators until a GPU verifier is available. It does not
    claim distributed, LoRA, CUDA-kernel training, or validated model improvements.
    """
    config.validate()
    if config.repair_turns:
        raise ValueError("Training repairs are unsupported: repair trajectories need separate conditional policy accounting. Use search mode.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Distributed training is not implemented; launch one process")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError as exc:
        raise RuntimeError("Training requires optional packages torch and transformers") from exc
    from rl_trainer.objective import grpo_loss_torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu with a suitably small local model")
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config.model, trust_remote_code=False, local_files_only=True).to(device)
    reference = AutoModelForCausalLM.from_pretrained(config.model, trust_remote_code=False, local_files_only=True).to(device)
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("Training currently supports decoder-only causal language models")
    reference.eval().requires_grad_(False)
    # eval disables dropout while autograd remains enabled, matching rollout and update probabilities.
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define a pad or EOS token")
    # A new config avoids silently inheriting top-k/repetition/forced-token processors
    # from a model repository, which would invalidate the raw-policy log ratios.
    generation_config = GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        num_return_sequences=config.group_size, max_new_tokens=config.max_new_tokens,
        pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
    )
    _write_json(out_path.with_suffix(".config.json"), {"mode": "grpo_numpy_dev", "config": asdict(config),
                                                     "operators": op_names, "device": device,
                                                     "distributed": False, "model_downloaded": False})

    def token_logps(policy, sequence, start):
        logits = policy(input_ids=sequence, use_cache=False).logits[:, start - 1:-1, :].float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, sequence[:, start:].unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        return selected, entropy

    for step in range(config.steps):
        op_name = op_names[step % len(op_names)]
        prompt = build_prompt(op_name, "numpy_dev")
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = encoded["input_ids"].shape[1]
        if prompt_length < 1:
            raise ValueError("Tokenizer produced an empty prompt")
        with torch.no_grad():
            sequences = model.generate(**encoded, generation_config=generation_config)
        trimmed, sources, old_logps, ref_logps = [], [], [], []
        eos_ids = tokenizer.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        for sequence in sequences:
            tokens = sequence[prompt_length:].tolist()
            end = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
            sequence = sequence[:prompt_length + end].unsqueeze(0)
            if end == 0:
                raise ValueError("Model generated no completion tokens")
            text = tokenizer.decode(sequence[0, prompt_length:], skip_special_tokens=True)
            # Invalid/empty generated text is still graded as a failed program.
            try:
                source = extract_source(text)
            except ValueError:
                source = text
            sources.append(source)
            trimmed.append(sequence)
            with torch.no_grad():
                old_logps.append(token_logps(model, sequence, prompt_length)[0])
                ref_logps.append(token_logps(reference, sequence, prompt_length)[0])
        rows = grade_sources_sync(op_name, sources, group_id=f"train-step-{step}-{op_name}", config=config)
        advantages = compute_grpo_advantages([row["reward"] for row in rows])
        optimizer.zero_grad(set_to_none=True)
        loss_value, entropy_value = 0.0, 0.0
        for sequence, old, ref, advantage in zip(trimmed, old_logps, ref_logps, advantages):
            current, entropy = token_logps(model, sequence, prompt_length)
            loss = grpo_loss_torch(current, old, ref, torch.tensor([advantage], device=device),
                                   torch.ones_like(current), clip=config.clip, beta=config.beta)
            (loss / config.group_size).backward()
            loss_value += float(loss.detach()) / config.group_size
            entropy_value += float(entropy.detach()) / config.group_size
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        for row in rows:
            row.update(method="grpo_numpy_dev", model=config.model, step=step, group_size=config.group_size,
                       policy_updated=True, loss=loss_value, entropy=entropy_value, grad_norm=float(grad_norm))
        _append_rows(out_path, rows)
        checkpoint = checkpoint_dir / f"step-{step + 1:06d}"
        checkpoint.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "step": step + 1,
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, checkpoint / "optimizer.pt")
        _write_json(checkpoint / "run_config.json", {"config": asdict(config), "step": step + 1,
                                                    "measurement_kind": "cpu_development", "target": "numpy_dev"})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=["search", "train"], default="search")
    parser.add_argument("--model", default=RunConfig.model)
    parser.add_argument("--op", action="append", dest="operators")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-secs", type=float, default=30)
    parser.add_argument("--repair-turns", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-url", help="Existing user-managed vLLM base URL including /v1")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--out", default="eval/results/dry_run.jsonl")
    args = parser.parse_args(argv)
    try:
        config = RunConfig(**{key: getattr(args, key) for key in RunConfig.__dataclass_fields__})
        config.validate()
        op_names = args.operators or ["rmsnorm_2048"]
        if not op_names or any(op not in OP_REGISTRY for op in op_names):
            raise ValueError(f"Unknown operator; available: {', '.join(sorted(OP_REGISTRY))}")
        out_path = Path(args.out)
        if args.dry_run:
            run_dry_loop(out_path)
        elif out_path.exists():
            raise ValueError("Output already exists; choose a new --out path to avoid mixing runs")
        elif args.mode == "search":
            if not args.base_url:
                raise ValueError("Search requires an existing vLLM --base-url; use --dry-run for local fixtures")
            generator = VLLMGenerator(args.base_url, config.model, max_tokens=config.max_new_tokens, concurrency=config.workers)
            asyncio.run(run_search(generator, op_names, out_path, config))
        else:
            if args.base_url:
                raise ValueError("Training cannot update a remote serving policy; omit --base-url")
            run_training(op_names, out_path, Path(args.checkpoint_dir), config, device=args.device)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
