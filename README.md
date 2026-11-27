# KernelBench-RL

**Two verified GPU-kernel case studies on an 8 GB laptop.**

KernelBench-RL is an independent engineering project that connects local model
adaptation, generated Triton programs, and execution-based verification. The
current showcase follows two selected successful programs from generation to
their saved correctness evidence.

**2/2 featured kernels verified.** These examples were selected for successful
verification; this count describes the featured case studies, not an estimate
of a model's overall accuracy or a complete benchmark success rate.

[Website](https://eljo-m.github.io/kernelbench-rl/) ·
[Case-study report](https://eljo-m.github.io/kernelbench-rl/paper.html) ·
[Download the PDF](https://eljo-m.github.io/kernelbench-rl/assets/kernelbench-rl-case-studies.pdf) ·
[Releases](https://github.com/Eljo-M/kernelbench-rl/releases)

| Featured case | Computation | Saved verification |
| --- | --- | --- |
| Fused bias + GELU | Add a bias and apply the GELU activation | PASS |
| Matrix multiplication + bias + SiLU | Combine a matrix product, bias and SiLU activation | PASS |

The featured programs are outputs of the recorded Qwen2.5-Coder-7B-Instruct
rank-16 LoRA policy. Both were checked against reference computations using
the registered input contracts. The report links each example to its exact
source, source hash and saved verification record.

## Why it is useful

- A worked example of adapting a 7B coding model under consumer-laptop memory constraints.
- Reusable numerical checks and a bounded worker for inspecting generated GPU programs.
- Saved programs, checkpoints and evidence that make the examples inspectable.

The documented environment is Windows, an NVIDIA RTX 5060 Laptop GPU with 8 GB
VRAM, PyTorch/CUDA and Triton Windows. The frozen quantized 7B base uses LoRA
adapters with CPU vocabulary placement; this is not full-parameter 7B training.
The project used existing hardware and free resources.

These are selected engineering demonstrations. They do not establish broad
generalization, an incremental benefit from reinforcement learning, or a speedup
for an entire real application. KernelBench-RL is independent of the original
[KernelBench benchmark](https://github.com/ScalingIntelligence/KernelBench).

## Inspect and run

The current website contains the two case-study receipts and generated sources.
Its data and presentation can be checked without loading a model or using a GPU:

```bash
python -m pip install -e ".[dev]"
python website/validate_showcase.py --site docs
```

For the underlying implementation, see [kernel verification](kernels/README.md),
[training](rl_trainer/README.md), and [checkpoint restoration](REPRODUCING_RETRY.md).
CUDA execution requires compatible GPU dependencies. The registered operator
contracts, source policy and child-process checks are development tools, not
operating-system containment for adversarial code.

## Report and attribution

Author: **Eljo Mustafaj** · GitHub: **Eljo-M**.

The current report is an independent technical case study. Editable sources are
under `paper/case-studies/`; the original study evidence and publication history
remain preserved separately. AI-assisted implementation and editing are
acknowledged as tools, not listed as authors.
