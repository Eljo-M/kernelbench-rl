# KernelBench-RL architecture

The JSON contract in `contracts/kernel_eval.schema.json` connects evaluation,
reward calculation and reporting. Optional trace metadata carries the seed,
method, candidate source hash, repair turn and fixture provenance.

```text
Generator (fixed CPU fixtures / existing model endpoint / local policy)
  -> candidate group
  -> AST policy + disposable Python workers with timeouts
  -> multi-seed/stress numerical checks and repeated CPU/CUDA timing
  -> KernelEvalResult + provenance
  -> group-relative rewards -> optional policy update
  -> JSONL -> grouped tables, profiler import, plots and offline trace viewer
```

The NumPy development verifier and generated-candidate CUDA verifier have
separate paths. The CUDA baseline validator checks all 30 handwritten entries.
A device being present never changes the label of a CPU measurement. Unavailable
comparison baselines use zero fields and explicit diagnostic metadata, rather
than an estimated timing. Reporting excludes unavailable timing comparisons.

The local training implementation must sample from the same policy it updates.
Search against a vLLM server records no policy update because serving weights are
not automatically synchronized. Multi-turn repair belongs to search; the trainer
does not silently use repaired sequences as original on-policy samples.

Kernel rewards penalize prohibited reference fallbacks, distinguish loading
failure from numerical failure, and reward measured speedup for correct programs.
Rewards are group-normalized before the clipped GRPO objective. The CUDA trainer
retains sampled completion IDs and old/reference probabilities, checks finite
gradients and records actual parameter changes. Full-model and adapter checkpoints
are atomic and preserve optimizer/RNG state. Evaluation generates base and trained
completions sequentially, then grades them after unloading the models. Best-of-K
selection uses a separate validation seed before final test grading. Reporting
checks complete operator/method coverage and keeps hardware/precision runs separate.
A checkpoint alone is not evidence of learning; the completed laptop ablation
showed no held-out quality gain.

The development execution boundary is a separate process with a reduced
environment, conservative NumPy API and timeout. This is not OS-level isolation.
For arbitrary untrusted kernels, supply an external VM/container boundary with
resource and access restrictions before execution. GPU event timing and native
compiler execution require their own suitable CUDA environment.

Track responsibilities and verified/remaining work are recorded in
[TRACK_SYNC.md](TRACK_SYNC.md).
