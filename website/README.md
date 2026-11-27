# KernelBench-RL selected kernel case studies

The buildless public website is in `docs/`. It presents two selected successful
generated kernels: fused bias-GELU and matmul-bias-SiLU. Its **2/2 featured kernels
verified** headline counts the displayed examples. It is not the complete
held-out pass rate, a new experiment, or an estimate of model accuracy.

The current pages are `index.html`, `paper.html` and `report.html`.
`case-studies.json` binds their selected-case claims to existing v3
`trained_checkpoint` records. Separate `evidence/` receipts and `kernels/`
source files make each featured program inspectable. Timing values preserve
their recorded hardware, precision and measurement method; they do not imply
general production acceleration. All assets are local, with no runtime
packages, external fonts, trackers, accounts or model calls.

The separate case-study paper lives under `paper/case-studies/`. Its public PDF
is `docs/assets/kernelbench-rl-case-studies.pdf`. The author is Eljo Mustafaj,
GitHub Eljo-M, without an institutional affiliation, DOI or peer-review claim.
The canonical public origin is `https://eljo-m.github.io/kernelbench-rl/`.
Offline checks do not establish that a deployment or scholarly index is live.

## Preserved complete study

`website/archive-v0.2/` preserves every original `docs/` Git blob from publication
commit `9e1c103bf4d3afe7cd5078868e9a00b4f1fdba65`, including the complete
three-study tables, evidence JSON, website and searchable 16-page paper.
It is outside the currently served `docs/` directory. The original
`paper/manuscript.json`, `paper/KernelBench-RL-Paper.pdf`, figures, study
records, source history and v0.2.0 release remain unchanged.

This is an immutable evidence mirror. Do not regenerate it in place, edit its
counts, or redirect the full-study checker to the selected subset. CI compares
every mirrored file against its original Git blob and runs the existing full
study metric, output, plot, upstream-provenance and PDF checks against it.
The old result-card helper also reads the mirror when no development `dist/`
copy is present.

The existing `build_data.py`, `configure.py`, finalizer and
`retry-results.schema.json` describe the complete study format. They must not
be run over the new selected-case site. To reproduce the original publication,
use a separate checkout of its exact source and a separate output directory.
`build_data.py` resets complete-study retry data to pending; it is not a
case-study website generator.

## Local checks

Run from the repository root:

```powershell
python website/validate_archive.py --repo .
python website/validate_static.py website/archive-v0.2
node --check website/archive-v0.2/assets/app.js
node website/test_result_cards.cjs
python website/validate_showcase.py --repo . --site docs
node --check docs/assets/app.js
```

The showcase check independently binds the selected PASS records, source
programs, receipt hashes, measurement fields, visible scope and separate paper.
The full study is checked separately with its original scientific semantics:

```powershell
python tools/publication/verify_publication_artifacts.py --repo . --require-v3 --output WORKSPACE/work/full-study-verification.json
```

The checker routes full-study website reads to `website/archive-v0.2/` when the
mirror exists. Its legacy behavior remains available in older checkouts and
isolated test fixtures without a mirror. CI requires the mirror explicitly;
it does not silently fall back when the archival publication is expected.

Serve `docs/` over local HTTP for browser QA. Check desktop/tablet/phone widths,
keyboard navigation and focus, copy controls, local links, citation metadata,
PDF text/search and page rendering. Static checks are not visual or external
link verification. The owner publishes only reviewed source and records exact
public file hashes after deployment.
