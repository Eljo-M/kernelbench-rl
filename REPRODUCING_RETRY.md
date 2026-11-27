# Reproducing and restoring the corrected studies

This guide separates report regeneration, checkpoint restoration and a new
training attempt. Historical results remain under their original directories.
The original corrected v2 studies' executed source is
`cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`; the final publication commit also
contains the later result files, paper and website. A release/source snapshot
and a study-source identity are different provenance fields. The independent v3
7B follow-up has its own executed source, recorded in
`experiments/retry_v3/source_provenance.json` as `study_source_commit`:
`2c174abe268495159c10d538c92d071e7733828d`. Do not
substitute the v3 or publication commit for the original v2 source. That
registration binds the v3 protocol's exact saved SHA-256, configuration hash and
pre-training publication/CI evidence. The final publication may contain all
three completed studies; their results remain separate.

Version 0.2 preserves experiment and publication files byte-for-byte through Git
attributes. Windows Git previously normalized some historical text blobs. The
publication snapshot preserves the saved working-artifact bytes, including
original line endings, so embedded output/proof hashes survive cloning on either
platform. Earlier commits and immutable historical ZIP exports remain intact.

## Tested laptop environment

The corrected runs use Windows, Python 3.12.14, an RTX 5060 Laptop GPU with 8 GB
nominal VRAM, PyTorch 2.11.0+cu128, Transformers 4.57.6, Triton Windows
3.6.0.post26 (runtime 3.6.0), Accelerate 1.16.0, PEFT 0.18.0, bitsandbytes
0.49.0, NumPy 2.3.5 and huggingface-hub 0.36.2. The saved environment files and
configuration hashes define each run. These measurements do not guarantee that
other GPU, operating-system or dependency versions produce identical programs
or timing.

An example installation in a Python 3.12 virtual environment is:

```powershell
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.57.6 triton-windows==3.6.0.post26 accelerate==1.16.0 peft==0.18.0 bitsandbytes==0.49.0 numpy==2.3.5 huggingface-hub==0.36.2
python -m pip install -e ".[dev]"
```

The Triton Windows package is for Windows. Linux uses its own compatible Triton
and PyTorch stack; the original free-T4 notebook preserves a separate historical
Linux protocol. It does not reproduce the corrected laptop CPU-vocabulary
placement experiment. Free Colab GPU quota was exhausted during this retry.
No additional GPU hardware, rented GPU time, paid model API or quota workaround
was used.

## Regenerate reports without a GPU

From the repository root with the CPU dependencies installed:

```powershell
python -m unittest discover -s tests -v
python -m eval.compare_retry --run-dir experiments/retry_v2/laptop --run-dir experiments/retry_v2/laptop_7b --historical-run-dir experiments/laptop_fullparam --historical-run-dir experiments/colab_t4/qlora7b_fresh --out work/recomputed-comparison
python -m eval.compare_retry --protocol experiments/retry_v3/protocol.json --run-dir experiments/retry_v3/laptop_7b --out work/recomputed-v3-comparison
```

The comparator validates source/config/model/checkpoint identity, complete
sampling inventories, validation-selected winners and phase labels. It keeps
three fresh test operators separate from the three previously inspected
development operators. Failures remain in every pass-rate denominator.
Best-of-eight is a larger inference budget than pass@1. The final hybrid
checkpoint includes supervised warmup; it is not an RL-only intervention.

The second comparator command is for a publication containing completed v3
evidence. It uses a different protocol and output directory. It never adds v3
operators or results to the v2 denominators. Before v3 completion, its strict
comparison must fail rather than emit placeholder quality results.

The v2 fresh cohort is `l2_normalize_2048`, `fused_bias_silu_4096` and
`matmul_residual_add_512`. In v3 these three are retired from testing and become
training targets. V3's new prospective cohort is `cosine_similarity_1024`,
`fused_bias_gelu_4096` and `matmul_bias_silu_512`. The preregistered exposure audit
found no earlier model-generated/SFT-target records for this new trio in the
audited repository artifacts. Their public contracts and baseline
implementations already existed; this does not establish pretraining
cleanliness or unseen-family generalization. The development cohort remains
`rmsnorm_4096`, `int8_weight_only_gemm_512` and
`scaled_dot_product_attn_128`, separately labeled in each protocol.

The original 180-record baseline campaign remains the source of its published
speedup values. The additional 60-record strict-worker regression checks the
updated output/timing guards; it does not replace that performance campaign.

## Restore the exact saved checkpoints

Download the source and the desired final or SFT-only checkpoint archives from
the project's GitHub Releases page. Verify their SHA-256 digests against
`RETRY_DELIVERY_MANIFEST.json`, then extract with this layout:

```text
workspace/
  outputs/
    kernelbench-rl/                     source archive
      checkpoints/retry_v2_laptop/...   0.49B checkpoint archive
      checkpoints/retry_v2_laptop_7b/... 7B adapter archive
      checkpoints/retry_v3_laptop_7b/... independent v3 adapter archive
  work/
    models/
      Qwen2.5-Coder-0.5B-Instruct/
      Qwen2.5-Coder-7B-Instruct/
```

Run from `workspace/outputs/kernelbench-rl`. The recorded model paths are
`../../work/models/...`. These paths participate in the configuration hash;
changing them creates a different identity. Retrieve the public, pinned models
to the matching directories with no access token:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    revision="ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
    local_dir="../../work/models/Qwen2.5-Coder-0.5B-Instruct",
    token=False,
)
snapshot_download(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
    local_dir="../../work/models/Qwen2.5-Coder-7B-Instruct",
    token=False,
)
```

The full-parameter archive includes the trained 0.49B model tensors. Its original
base is still needed for the reference policy. The 7B archive includes adapters,
optimizer/RNG state and provenance; the frozen 7B base is obtained from the
upstream model repository. Bundles preserve the original model license and
attribution. Tensor checksums and serialized-file/ZIP digests have different
meanings and must not be interchanged.

For a zero-extra-step restoration on a separate extracted copy, create a new
configuration file with only `resume_from` set to the extracted checkpoint root,
then call the trainer. `resume_from` is intentionally excluded from the config
hash; preserve every other recorded field and the model directory layout:

```python
from pathlib import Path
from rl_trainer.experiment_config import ExperimentConfig

cfg = ExperimentConfig.from_json(Path("experiments/retry_v2/laptop/config.json"))
cfg.resume_from = cfg.checkpoint_dir
cfg.to_json(Path("restoration_config.json"))
```

```powershell
python -m rl_trainer.gpu_train_grpo --config restoration_config.json
```

At the completed eight-group checkpoint this adds no optimizer steps. The
checkpoint manager verifies model identity and restores optimizer and random
state. Select `experiments/retry_v2/laptop_7b/config.json` for the v2 adapter
track. For the independent v3 track, use
`experiments/retry_v3/laptop_7b/config.json` and
`checkpoints/retry_v3_laptop_7b`; its completed checkpoint is group step 16.
The same restoration recipe then adds no training steps. The SFT-only archive
for each study is its own `sft_only/step_00000` control and is separately
evaluated by that study. A model-checksum-verified reload restoring optimizer/RNG
with zero extra steps is not a next-step optimizer/RNG replay proof.

`run_study --resume-study` is stricter than checkpoint restoration: it also binds
the original absolute output paths and source commit. It is intended to continue
the same interrupted local study. Do not rewrite environment manifests to make a
copied study appear to originate on another machine.

## Reproduce the independent v3 schedule

Use a separate clean checkout at the exact v3 `study_source_commit` from
`experiments/retry_v3/source_provenance.json`, retaining that registration from
the publication snapshot for provenance. Its protocol was committed before
training; the registration itself was saved after that commit was published.
Use the same workspace/model directory layout above. The source checkout for a
fresh reproduction must have no existing v3 `environment.json` or v3 checkpoint
state. Preserve the downloaded study evidence and checkpoints in their original
copy.

The v3 configuration hash is `0dfb0a14abad3067`. It uses the pinned original
Qwen2.5-Coder-7B-Instruct revision above, fresh rank-16 adapters with alpha 32 on
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj`.
It does not load a v2 adapter or resume a v2 optimizer. `resume_from` is empty
for the initial invocation. The frozen NF4 base and pre-SFT reference-policy
anchor are retained; this is adapter training with CPU vocabulary offload, not
full-parameter 7B training.

Eight SFT epochs over the eleven ordered training operators produce 88 scheduled
SFT updates. Sixteen GRPO groups of two produce 32 rollouts, with seed 242.
The first five operators receive a second GRPO group because the configured
eleven-operator order cycles. The actual number of GRPO optimizer updates must
come from the completed records; sixteen groups do not imply sixteen updates.
Both non-PASS rewards and equal-reward relative advantages can yield no update.
Base pass@1, validation-selected base best-of-eight, SFT-only pass@1 and final
hybrid pass@1 retain their preregistered budgets and 1,536-token generation cap.

After setting the Windows caches and thread limits shown below, launch from
the isolated source checkout's repository root:

```powershell
python -m rl_trainer.run_study --config experiments/retry_v3/laptop_7b/config.json --threads 4
```

The configuration keeps v3's training/evaluation records under
`experiments/retry_v3/laptop_7b/` and checkpoints under
`checkpoints/retry_v3_laptop_7b/`. The protocol is
`experiments/retry_v3/protocol.json`; `run_study` takes the configuration, while
the strict comparator takes that protocol explicitly. After interruption, only
the original local attempt with a durable checkpoint may use the opt-in form:

```powershell
python -m rl_trainer.run_study --config experiments/retry_v3/laptop_7b/config.json --threads 4 --resume-study
```

That continuation belongs to the same v3 attempt. It does not change its fresh
initialization into a transfer from v2. Never restart partial SFT silently or
change the frozen test cohort to obtain a target pass count. A relocated or
edited configuration defines a new experiment if it changes hashed fields.

For the final v3 publication, retain the original v2 upstream integrity record
and run the public-file checker again after all v3 stages have completed:

```powershell
python tools/publication/verify_model_snapshots.py --verify-local --metadata-input experiments/retry_v2/upstream_expected_metadata.json --output experiments/retry_v3/upstream_model_integrity.json
```

This streams the actual local base shards against the saved official metadata;
it neither loads models nor redownloads the cache. The new proof is separate
from v2's proof and must not replace it. The publication artifact checker and
exporter refuse incomplete v3 stages, changed evaluation/protocol bytes,
missing original-preserving frozen-base resume sidecars, and a missing/stale
post-v3 upstream proof. CPU CI verifies those saved records without repeating
model downloads, shard hashing or GPU experiments. Its explicit v3 gate is:

```powershell
python tools/publication/verify_publication_artifacts.py --repo . --require-v3 --output ../SOURCE_CHECKOUT_VERIFICATION.json
```

Run this only on the clean final publication checkout after the paper/site and
all evidence files have been committed. Use a new output path when preserving
an existing verification record.

## Start a new quality study

Use a clean source checkout, a new output/checkpoint directory and an explicit
protocol. Keep training operators disjoint from both evaluation cohorts. Set
workspace-local Triton/Inductor caches before launching Windows GPU workers.
The two model tracks must run sequentially on an 8 GB GPU.

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:TRITON_CACHE_DIR=(Join-Path (Get-Location) 'work/triton-cache')
$env:TORCHINDUCTOR_CACHE_DIR=(Join-Path (Get-Location) 'work/inductor-cache')
python -m rl_trainer.run_study --config path/to/new-config.json --threads 4
```

The orchestration refuses to overwrite an existing run and requires committed
Python source. A completed study includes training, exact resume verification,
base pass@1, validation-selected base best-of-eight, final-checkpoint pass@1,
separate development regressions and the SFT-only control. Scheduled groups and
actual optimizer updates are reported separately. The original pre-SFT base is
the reference-policy anchor in both tracks. A partial warmup without a durable
SFT checkpoint is preserved as an interrupted attempt rather than silently
repeated.

The source policy and child-process timeout are development guards. They do not
provide operating-system isolation or a formal equivalence proof. The public
website is a static evidence explorer; it does not execute arbitrary GPU code.
