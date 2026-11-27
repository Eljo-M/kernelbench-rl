# Technical report source

`manuscript.json` is the finalized, editable text and result tables.
`bibliography.json` records ten primary references and their verified metadata,
including the KernelLLM software release and the Kevin research preprint.
The public PDF is linked from the repository and the project website's paper
page. It is an independent preprint, with no claimed journal acceptance, DOI
or institutional affiliation.

To reproduce the typeset paper from the repository root, install ReportLab,
pypdf and Pillow, then run:

```powershell
python paper/build_paper.py --manuscript paper/manuscript.json --bibliography paper/bibliography.json --baseline-comparison experiments/rtx5060/full30/report/baseline_comparison.json --out work/KernelBench-RL-Paper.pdf --final --render-pages
```

The published rendering uses Windows Georgia, Calibri and Cambria Math fonts.
The renderer embeds the fonts, verifies glyph coverage, rejects unresolved
result placeholders and limits the final searchable PDF to 5 MB. Use
`--font-dir` for a compatible font directory on another machine. Review every
rendered page after changing text, tables, fonts or figures.

The v2 SFT figure is generated from the actual 32-update logs for each of its
two model tracks:

```powershell
python paper/plot_training.py --repo . --out paper/paper-assets/sft_training.png
```

It checks complete epoch/operator/seed/order/LR inventories. Points are logged
pre-update completion-token negative log likelihood; epoch means average eight
examples and are not token-weighted. Training fit is separate from generated
program validity. Figure metadata binds the configuration and input-log hashes.

The separate v3 figure uses the actual 88 supervised updates from its fresh
7B rank-16 adapter study:

```powershell
python paper/plot_followup.py --repo . --out paper/paper-assets/sft_followup.png
```

Its eight epoch means average eleven example losses each. The plotter verifies
the configured operator order, completion token counts, saved SFT checkpoint,
training-source identities and pre-training protocol/source registration.
Neither loss figure is a held-out accuracy curve. Regenerating a plot uses CPU
dependencies, including Matplotlib and NumPy, and does not run model training.
Keep the published figure and its JSON sidecar together; the sidecar includes
the precise input and PNG hashes.

`manuscript_template.json` is a guarded input template for the publication
finalizer, not an additional paper or a source of new empirical results. The
finalizer fills it only after strict completed-study, checkpoint and verifier
regression validation. It never manufactures missing measurements.

The report separates the two corrected v2 studies from the independent v3
follow-up. Their executed source commits are preserved in experiment metadata;
the later publication commit is a separate identity. All three former v2 fresh
operators become training targets in v3. The v3 prospective evaluation therefore
uses a different, separately registered trio. Preserve these cohort boundaries
when editing the manuscript or comparing results.

The figures, tables, website evidence and report derive from saved run files.
Keep model, precision, inference budget and fresh/development cohorts separate.
An SFT-plus-GRPO checkpoint does not establish an RL-only effect. The model reload
checks the trainable tensor checksum and restores optimizer/RNG state; it is not
an independently replayed next-step trajectory proof.

Google Scholar discovery uses a static public abstract page, a searchable PDF,
ordinary links and citation metadata. Meeting those technical conditions does
not guarantee indexing or a particular discovery time. See the
[official inclusion guidance](https://scholar.google.com/intl/en/scholar/inclusion.html).
