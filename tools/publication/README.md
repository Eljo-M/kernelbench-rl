# Publication and provenance tools

Author: **Eljo Mustafaj**. GitHub identity: **Eljo-M**. No institutional affiliation, journal acceptance or DOI is implied. Publication dates must use the author's actual local calendar date in America/New_York, supplied explicitly to the finalizer; a UTC artifact timestamp is not a publication date.

The version 2 corrected studies executed from frozen source commit `cc4dcc94859c5e0d8658a6224bbbf667a93bdb7e`. A later publication HEAD contains their completed traces, audits and publication helpers. Those commits are distinct provenance fields. Publication tooling must not relabel a study as having executed from a later report/documentation commit. A subsequent training/evaluation protocol must retain its own source, configuration and evidence; these version 2 finalization tools must be extended explicitly before incorporating another study.

## Exact restoration layout

```text
workspace/
  outputs/
    kernelbench-rl/                 repository/source archive; command CWD
      checkpoints/retry_v2_laptop/
      checkpoints/retry_v2_laptop_7b/
  work/
    models/Qwen2.5-Coder-0.5B-Instruct/
    models/Qwen2.5-Coder-7B-Instruct/
```

Keep recorded configurations unchanged: `../../work/models/...` and active `sft_output_jsonl` participate in `config_hash`. Edited paths define a different identity, not an exact resume. The model directories require the public revisions in those configurations. The 7B archives contain adapters, never a redistributed 7B base cache. See the repository reproduction guide for pinned free download and restoration commands.

The exporter detects this layout from `tools/publication/` or the original workspace `work/`. Explicit `--workspace-root`, `--repo`, `--output-root`, `--work-root`, model-directory, paper, report and site overrides are supported. A standalone checkout outside the documented layout defaults its workspace to the repository itself; use explicit layout arguments for exact restoration.

## Regenerate publication content

Run the following only after both studies, controls, resume checks, upstream snapshot hashes and the separate 60-record verifier regression are complete. These tools derive claims from saved evidence; they do not manufacture missing runs.

From the repository root, set `OUTPUT` to the chosen deliverable directory and `STATIC_DIST` to the final site directory (`docs` for GitHub Pages). The manuscript template and primary bibliography are under `paper`.

```text
python paper/plot_training.py --repo . --out OUTPUT/paper-assets/sft_training.png
python tools/publication/finalize_retry_text.py --repo . --site STATIC_DIST --paper-source paper --out OUTPUT --publication-date YYYY-MM-DD --training-figure OUTPUT/paper-assets/sft_training.png --manuscript-template paper/manuscript_template.json
python paper/build_paper.py --manuscript OUTPUT/KernelBench-RL-Manuscript.json --bibliography paper/bibliography.json --baseline-comparison experiments/rtx5060/full30/report/baseline_comparison.json --out OUTPUT/KernelBench-RL-Paper.pdf --final --render-pages
```

`YYYY-MM-DD` is the actual author-local date. Review every rendered PDF page and verify searchable text. Copy that same final PDF to the site, then run `website/configure.py` with the actual public base URL, PDF-relative path and verified download links. The paper title, author, date, abstract, BibTeX, static citation metadata and downloadable PDF must agree. The finalizer binds completed manifests, checkpoint counters/checksums, training records, quality results and figure identities before filling manuscript/site text. A model-checksum-verified reload restores optimizer/RNG state with zero extra steps; it is not an independent optimizer/RNG digest or next-step replay proof.

`verify_model_snapshots.py --metadata-only --check-small-files` captures official public Hugging Face metadata with `files_metadata=True, token=False` and checks only small local files. The separate `--verify-local --metadata-input EXPECTED_JSON` pass streams all safetensors after active studies end. Only that completed pass writes `experiments/retry_v2/upstream_model_integrity.json`; metadata-only evidence is not weight-hash verification.

## Export and publication checks

After the final files and exact-byte Git attributes have been captured in the publication commit, wait for its CI and pass the actual source SHA and run URL:

```text
python tools/publication/export_retry.py --workspace-root WORKSPACE --source-commit FINAL_40_CHARACTER_SHA --ci-url https://github.com/Eljo-M/kernelbench-rl/actions/runs/ACTUAL_RUN_ID --site STATIC_DIST --site-url ACTUAL_PUBLIC_BASE_URL --include-historical
```

`--validate-only` checks readiness without packaging. The full exporter validates both completed model studies, strict fresh/development comparisons, phase/update counters, checkpoint/output identities and the separate verifier regression. It stages versioned source, raw evidence, final/SFT-only checkpoint, website and upstream-license packages; verifies CRCs, traversal, shard splitting and SHA-256; then writes `RETRY_DELIVERY_MANIFEST.json` and `retry-release-assets.json`. Assets are strictly below 2 GiB, or split with explicit reassembly/checksum metadata. `--rebuild` is limited to unchanged files owned by an earlier retry-export manifest. Existing historical exports and `DELIVERY_MANIFEST.json` are never overwritten. A companion upstream-license ZIP supplies original licenses/attribution for unchanged historical model bundles.

The final workflow preserves the full Python test discovery and dedicated real tiny-Torch/Rust tests. Additional CPU checks regenerate strict retry comparisons from committed files, validate `docs`, compile publication scripts and syntax-check JavaScript on Node 22. CI never downloads model weights, hashes the 15 GB snapshots, provisions resources or runs GPU jobs. Paper render/browser verification and live hosting/release verification remain separate checks.

`verify_publication_artifacts.py` adds a CPU-only consistency check of completed stage-output hashes, frozen-base original/resume proof bytes, figure/config/SFT-log hashes, the paper/website PDF identity, and recorded full upstream model-hash evidence against the unchanged copied expected metadata. It reads Git HEAD/status for provenance but never creates a clone unless requested. CI runs this default check into `/tmp` without accessing local model weights.

After the final commit, explicitly run the integration round trip:

```text
python tools/publication/verify_publication_artifacts.py --repo . --expected-commit FINAL_40_CHARACTER_SHA --clone-check --work-root WORKSPACE/work --output WORKSPACE/outputs/SOURCE_CHECKOUT_VERIFICATION.json
```

This requires a clean exact source HEAD, creates a new retained local clone with `core.autocrlf=true`, compares every tracked `experiments`, `docs` and `paper` file against both working bytes and committed blob bytes, then repeats saved-byte checks in the clone. Comparing blobs matters: a CRLF Windows checkout can otherwise hide an LF blob that fails on Linux. Nothing is deleted or altered in the source repository. Add `--verification-file WORKSPACE/outputs/SOURCE_CHECKOUT_VERIFICATION.json` to the final export command to include the matching successful proof as a release asset and manifest reference. Existing verification outputs are preserved; choose a new filename for another attempt.

## Byte preservation boundary

Windows `core.autocrlf=true` previously normalized line endings in some older Git blobs. The saved historical ZIP exports remain byte-for-byte unchanged, and earlier Git commits remain as they were. Do not describe those older normalized blobs as exact copies of every saved working artifact.

The publication attributes mark `experiments/**`, `docs/**` and `paper/**` as `-text`. Attribute/index changes must occur **after** GPU study/regression jobs stop, followed by verification that committed blobs and clean clones preserve the actual saved bytes and embedded output/proof hashes. Stage new files as well: `git add --renormalize` alone updates previously tracked files. The publication snapshot captures those saved bytes, including their existing line endings. Source/proof guards must not be changed during an active study. Enable the full publication CI checks only when the committed publication inputs exist; the training workflow remains applicable during an independent new study.
