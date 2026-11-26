# KernelBench-RL architecture

The JSON contract in `contracts/kernel_eval.schema.json` connects evaluation,
reward calculation and reporting. Optional trace metadata carries the seed,
method, candidate source hash, repair turn and fixture provenance.

```text
Generator (fixed CPU fixtures / existing model endpoint / local policy)
  -> candidate group
  -> AST policy + disposable Python workers with timeouts
  -> multi-seed/stress numerical checks and repeated CPU timing
  -> KernelEvalResult + provenance
  -> group-relative rewards -> optional policy update
  -> JSONL -> grouped tables, profiler import, plots and offline trace viewer
```

The NumPy development verifier and CUDA baseline validator have separate paths.
A device being present never changes the label of a CPU measurement. Unavailable
comparison baselines use zero fields and explicit diagnostic metadata, rather
than an estimated timing. Reporting excludes unavailable timing comparisons.

The local training implementation must sample from the same policy it updates.
Search against a vLLM server records no policy update because serving weights are
not automatically synchronized. Multi-turn repair belongs to search; the trainer
does not silently use repaired sequences as original on-policy samples.

Kernel rewards penalize prohibited reference fallbacks, distinguish loading
failure from numerical failure, and reward measured speedup for correct programs.
Rewards are group-normalized before the clipped GRPO objective. A checkpoint alone
is not evidence of learning; held-out operator/seed ablations remain required.

The development execution boundary is a separate process with a reduced
environment, conservative NumPy API and timeout. This is not OS-level isolation.
For arbitrary untrusted kernels, supply an external VM/container boundary with
resource and access restrictions before execution. GPU event timing and native
compiler execution require their own suitable CUDA environment.

Track responsibilities and verified/remaining work are recorded in
[TRACK_SYNC.md](TRACK_SYNC.md).
