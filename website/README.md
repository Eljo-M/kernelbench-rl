# KernelBench-RL website source

The buildless public website lives in [`docs/`](../docs), with its abstract page at [`paper.html`](../docs/paper.html). The canonical publication base is `https://eljo-m.github.io/kernelbench-rl`. GitHub Pages serves `main:/docs`; `.nojekyll` prevents template processing. Hosting is managed separately from the static source.

The site uses local HTML/CSS/JavaScript, system fonts and a small public JSON evidence file. It has no accounts, model endpoints, database, analytics, external font dependency or paid service integration. Operator search, precision/tier filters, source inspection and pipeline tabs run in the browser.

## Evidence and content

`build_data.py` reads the saved final baseline JSONL, operator registry and historical paired-comparison files. It writes 30 operator contracts/sources, actual correctness/speedup summaries, historical model outcomes and artifact SHA-256 provenance into the specified `results.json`. It runs no model or GPU work.

```text
python website/build_data.py --repo . --out docs/results.json
```

This initializes the historical evidence layer and resets the retry section; run the publication finalizer afterward to insert the strictly validated completed retry results. Do not publish an intermediate export in place of the complete evidence. The finalizer keeps fresh and previously inspected development cohorts separate and retains failures in denominators. `retry-results.schema.json` describes the normalized retry interface; schema examples are not experimental results.

The custom 30-operator workload is not the complete original KernelBench dataset or an official KernelBench leaderboard. Fixed/curated baseline performance and evaluated-model outputs remain distinct. Compiled comparisons that were unavailable remain unavailable; no learned speedup, native FP8 capability, distributed training, formal correctness proof or adversarial security isolation is inferred from those records.

## Paper, metadata and download links

The author is Eljo Mustafaj. The abstract page's title, author, actual author-local publication date, abstract and BibTeX are filled from the finalized manuscript. No affiliation, DOI, venue or peer-review status is invented.

Copy the verified searchable PDF to `docs/assets/kernelbench-rl-paper.pdf`, then configure its exact public URL:

```text
python website/configure.py --dist docs --public-base-url https://eljo-m.github.io/kernelbench-rl --pdf assets/kernelbench-rl-paper.pdf --downloads-json VERIFIED_LINKS.json --ready
```

`VERIFIED_LINKS.json` is a list of `{ "label": "...", "url": "https://..." }` entries whose public destinations have been checked. It may point to the project's real Releases index to avoid circular source-commit/asset-name dependencies. Omit `--downloads-json` when there are no additional verified links.

`PUBLIC_BASE_URL` can supply the same HTTPS base including the GitHub project path. The configuration script checks the local PDF, writes static absolute `citation_pdf_url`/abstract/canonical metadata and the PDF href, updates BibTeX and crawler files, and preserves public metadata without requiring JavaScript. `--ready` removes the construction banner and enables indexing; omitting it intentionally retains a guarded unpublished state. Technical Scholar compatibility does not guarantee indexing or ranking.

The default helper output folder is `website/dist`; always pass `--out docs/results.json`, `--dist docs`, or the positional `docs` validator argument when operating on the committed GitHub Pages source. Static files can also be copied to a separate hosting checkout without changing their public canonical metadata.

## Preview and validation

From the repository root:

```text
python -m http.server 4173 --directory docs
python website/validate_static.py docs
node --check docs/assets/app.js
python -m compileall -q website tools/publication paper
```

Serve HTTP rather than opening `file://`, because the application fetches local JSON. The validator checks local assets/anchors, document language, image alt text, a single H1 per page, citation identity, JavaScript element references and evidence coverage. The publication byte verifier additionally binds the PDF, figure, configuration, trace and upstream metadata hashes. These checks do not execute GPU code.

Browser QA covers narrow/wide layouts, page overflow, keyboard focus and tab arrow/Home/End navigation, search/no-results, tier/precision filters, source expansion, clipboard success/fallback, actual retry tables, PDF searchability and public HTTP responses. CSS provides labeled controls, live result counts, responsive layouts, scrollable tables/code, reduced-motion support and a skip link. Browser/hosting receipts are separate from static validation.

Publication scripts and the exact-byte checkout verification procedure are documented under [`tools/publication`](../tools/publication). Do not regenerate hash-bound data or alter its line endings merely to satisfy a check; preserve the original evidence and investigate any mismatch.
