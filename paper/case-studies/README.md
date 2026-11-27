# Two verified GPU-kernel case studies

This is the selected-case technical report for KernelBench-RL v0.3.0, by Eljo
Mustafaj. The two examples were selected after successful verification; the
featured 2/2 count is not a complete-evaluation model success rate.

`manuscript.json` contains the editable prose, `evidence.json` contains the two
saved records and their provenance, `programs/` retains the exact generated
source text, and `bibliography.json` records the primary references.

Render the five-page PDF from these saved inputs with Python and ReportLab:

```bash
python -m pip install reportlab
python render_paper.py --out KernelBench-RL-Case-Studies.pdf
```

The renderer detects common Windows or Linux font families; `--font-dir` can
select a local TrueType font directory. Rendering requires no model, GPU,
network access or new experiment. Output PDF bytes can vary with font and
runtime versions. The released PDF is identified by its saved SHA-256.

`prepare_source.py` is a separate repository evidence-extraction helper. It is
not required for rendering the included saved sources. `qa.json` records visual
and byte checks of the released PDF.
