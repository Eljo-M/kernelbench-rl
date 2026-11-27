# Reproducing and restoring the corrected studies

This guide separates report regeneration, checkpoint restoration and a new
training attempt. Historical results remain under their original directories.
The corrected study's executed source is
`cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`; the final publication commit also
contains the later result files, paper and website. A release/source snapshot
and a study-source identity are different provenance fields.

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
```

The comparator validates source/config/model/checkpoint identity, complete
sampling inventories, validation-selected winners and phase labels. It keeps
three fresh test operators separate from the three previously inspected
development operators. Failures remain in every pass-rate denominator.
Best-of-eight is a larger inference budget than pass@1. The final hybrid
checkpoint includes supervised warmup; it is not an RL-only intervention.

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
state. Select `laptop_7b/config.json` for the adapter track. The SFT-only archive
is the step-zero control and is separately evaluated by the saved study.

`run_study --resume-study` is stricter than checkpoint restoration: it also binds
the original absolute output paths and source commit. It is intended to continue
the same interrupted local study. Do not rewrite environment manifests to make a
copied study appear to originate on another machine.

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
