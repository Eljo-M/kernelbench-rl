# KernelBench-RL research website

This is a buildless static website. The committed site is in `docs/`; its
maintenance tools are in `website/`. A development copy can instead keep the
site in `website/dist/`, beside those tools. Each site directory contains HTML,
CSS, JavaScript, machine-readable evidence and publication configuration. It has no runtime
packages, external fonts, trackers, account flows or model calls. Its intended
public origin is `https://eljo-m.github.io/kernelbench-rl/`; successful remote
publication is established by the owner's hosting receipt, not an offline build.

## Pages and evidence

- `index.html` in the site directory explains the pipeline, reports fixed baselines separately
  from generated programs, displays one responsive card per recorded study,
  and preserves the original historical results separately. The operator
  explorer and precision/tier filters use saved evidence.
- `paper.html` is the paper landing page for Eljo Mustafaj, with abstract,
  citation metadata, BibTeX and the configured searchable PDF. It invents no
  affiliation, DOI or peer-review status.
- `results.json` contains artifact-derived summaries, thirty custom
  operator contracts and curated fixed sources, historical model results,
  normalized corrected-study records and input-file hashes. Curated baselines
  were prepared with coding-assistant support; their performance is separate
  from programs sampled by the evaluated models.
- `project-config.json` controls the canonical origin, PDF path, verified
  download links and draft/indexing state. Local readiness does not prove that
  files have already been uploaded or indexed.

Study identities are protocol-qualified: `v2/laptop`, `v2/laptop_7b` and
`v3/laptop_7b`. The repeated folder basename `laptop_7b` is not a unique study.
The final three-study publication requires all registered stages to complete
and pass strict validation. An incomplete v3 run must not be represented by
assumed counts, a copied v2 score or a placeholder result card.

Study cards retain actual SFT/GRPO optimizer counts, the four fresh methods and
inference budgets, three development methods, exact workload names and source
identity. Hardware and fuller provenance are in each card's disclosure. Full
interpretation is under “Full study notes and interpretation.” Different v2/v3
fresh cohorts and training configurations are not pooled or a causal comparison.

## Evidence build order

Commands below run from the repository root. The historical builder reads saved
records and registry metadata on CPU. Select its destination explicitly:

```powershell
python website/build_data.py --repo . --out docs/results.json
```

It checks the original 180 unique operator/precision/seed records and rebuilds
fixed-baseline and historical evidence. Geometric means use correct recorded
speedups; compiled comparisons exclude unavailable timings. It runs no model,
CUDA workload or generated-program experiment.

For a separate development site, use `--out website/dist/results.json` instead.
The builder creates evidence JSON; it does not initialize the development site's
HTML/CSS/JavaScript. Start that development directory from a complete site copy.

**`build_data.py` deliberately resets `results.json.retry` to pending.** Run it
before corrected-study integration. If it is run later, reapply the independently
validated normalization before publication; do not hand-copy headline counts.

After the registered studies, controls and regression are complete, the owner
runs `tools/publication/finalize_retry_text.py` from the publication checkout
(or its prepared work counterpart). It consumes strict artifacts and manuscript
source, writes actual normalized study records and updates paper/report text.
It refuses incomplete present v3 evidence, binds plots to logs, preserves the
original v2 comparison bytes, and gives the manuscript and website one verified
metric source.

For final publication, v3 is required alongside both v2 studies; the committed
artifact checker uses `--require-v3`. V3 starts with fresh base/adapters and
promotes former v2 fresh targets into training. Those targets receive no v3
fresh/development credit. Only real completed records add the third card, even
if its score is below v2.

`retry-results.schema.json` documents normalized evidence. Protocol-qualified
keys, short labels, `cohort_ops`, configuration/revision and note fields retain
clear provenance. Validation checks unique keys, workload lists against method
denominators and disjoint fresh/development cohorts. Failures stay in pass
rates; development best-of-eight is unavailable, not an invented zero.

## Paper and publication configuration

The repository manuscript PDF is `paper/KernelBench-RL-Paper.pdf`. Its distinct
public copy is `docs/assets/kernelbench-rl-paper.pdf`, served at
`https://eljo-m.github.io/kernelbench-rl/assets/kernelbench-rl-paper.pdf`.
Copy the verified searchable PDF without changing its bytes; the publication
checker compares repository and website PDF hashes.

```powershell
python website/configure.py --dist docs --public-base-url https://eljo-m.github.io/kernelbench-rl --pdf assets/kernelbench-rl-paper.pdf
```

For the development copy, change `--dist docs` to `--dist website/dist`.
`configure.py` sets canonical/citation URLs, the PDF href and local config. It
refuses missing/non-PDF files and publishes nothing. The paper landing page is
`paper.html`; the explanatory report is `report.html`. For versioned archives
and SHA-256 manifests, see GitHub Releases. Download entries use owner-verified
URLs through `--downloads-json`; never guess release asset names.

Use `--ready` only after final evidence, PDF verification and browser QA. This
updates local draft/indexing metadata and sitemap. The final GitHub Pages
workflow and public checks establish availability; appropriate scholarly
metadata does not guarantee indexing.

## Checks and presentation QA

For the committed site, after actual completed records are integrated:

```powershell
python website/validate_static.py docs
node --check docs/assets/app.js
node website/test_result_cards.cjs
```

The card helper prefers `website/dist/` when that directory exists; otherwise it
reads `docs/`. To check the committed site with its default invocation, run from
a checkout without a local `website/dist/` development copy. Paths are resolved
relative to the helper, so changing the shell's working directory does not change
the selected site. In the prepared work copy, the same helper selects its local
`work/website-source/dist/`.

For a complete development copy at `website/dist/`, use:

```powershell
python website/validate_static.py website/dist
node --check website/dist/assets/app.js
node website/test_result_cards.cjs
```

That CPU-only structural test checks method rows, counts, budgets, workload
names, source identities, rerender behavior and safe text rendering. Its explicit
`--fixture` input is for isolated TEST ONLY data, never research output. Static
checks do not constitute visual browser or external-link verification.

Serve the chosen site directory (`docs/` or development `website/dist/`) over
local HTTP for browser QA; JSON fetches do not support a
plain `file://` preview. Check wide/tablet/phone widths including 320–390 px,
card stacking, long labels, intended internal table/code scrolling and absence
of page-level overflow. Check keyboard focus, pipeline arrow/Home/End navigation,
search/no-results, tier/precision filters, disclosures, clipboard fallback,
citation links and PDF text/search/rendering. Cards use 16 px body text and at
least 14 px labels. The owner verifies final rendering after GPU work releases
the machine.
