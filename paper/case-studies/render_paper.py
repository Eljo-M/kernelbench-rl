"""Render the separate five-page selected-case paper with native vector figures."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

HERE = Path(__file__).resolve().parent
PAGE_W, PAGE_H = A4
MARGIN = 48
WIDTH = PAGE_W - 2 * MARGIN
INK = colors.HexColor("#173443")
TEAL = colors.HexColor("#087E82")
GOLD = colors.HexColor("#B57A38")
MUTED = colors.HexColor("#506572")
PALE = colors.HexColor("#EDF5F5")
GRAY = colors.HexColor("#F2F5F7")
LINE = colors.HexColor("#CFDCDF")
WHITE = colors.white


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(text):
    return html.escape(str(text))


class Paper:
    def __init__(self, output, manuscript, evidence, bibliography, source_path, font_dir=None):
        font_sets = [(Path("C:/Windows/Fonts"), ["segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "georgia.ttf", "georgiab.ttf"]),
                     (Path("/usr/share/fonts/truetype/liberation2"), ["LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"]),
                     (Path("/usr/share/fonts/truetype/dejavu"), ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"])]
        if font_dir:
            font_sets = [(font_dir, names) for _, names in font_sets]
        selected = next(((directory, names) for directory, names in font_sets if all((directory / name).is_file() for name in names)), None)
        if selected is None:
            raise FileNotFoundError("Provide --font-dir containing a complete Windows, Liberation2, or DejaVu TTF family")
        directory, filenames = selected
        self.font_files = [str(directory / name) for name in filenames]
        for name, filename in zip(("Body", "Body-Bold", "Body-Italic", "Display", "Display-Bold"), filenames):
            pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
        self.doc = manuscript
        self.evidence = evidence
        self.refs = bibliography["references"]
        self.c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
        self.c.setTitle(manuscript["title"])
        self.c.setAuthor(manuscript["author"]["display_name"])
        self.c.setCreator("KernelBench-RL selected-case renderer / ReportLab")
        self.c.setSubject("KernelBench-RL manuscript SHA-256: " + digest(source_path))
        self.page = 0
        self.y = PAGE_H - 61
        self.blocks = []
        self.styles = dict(
            body=ParagraphStyle("body", fontName="Body", fontSize=10.1, leading=14.4, textColor=INK, spaceAfter=0),
            small=ParagraphStyle("small", fontName="Body", fontSize=9.0, leading=12.6, textColor=MUTED),
            tiny=ParagraphStyle("tiny", fontName="Body", fontSize=8.2, leading=11.2, textColor=MUTED),
            h1=ParagraphStyle("h1", fontName="Display-Bold", fontSize=23.5, leading=28, textColor=INK),
            h2=ParagraphStyle("h2", fontName="Body-Bold", fontSize=11.7, leading=16, textColor=INK),
            kicker=ParagraphStyle("kicker", fontName="Body-Bold", fontSize=8.1, leading=11.5, textColor=TEAL),
            caption=ParagraphStyle("caption", fontName="Body", fontSize=8.6, leading=11.7, textColor=MUTED),
            reference=ParagraphStyle("reference", fontName="Body", fontSize=8.8, leading=12.1, textColor=INK),
            equation=ParagraphStyle("equation", fontName="Body-Bold", fontSize=10.3, leading=16, textColor=INK, alignment=TA_CENTER),
        )

    def block(self, x, bottom, width, height, label):
        if min(x, bottom) < 0 or x + width > PAGE_W + .1 or bottom + height > PAGE_H + .1:
            raise ValueError(f"Off-page block p{self.page}: {label}")
        if bottom < 45:
            raise ValueError(f"Block enters footer p{self.page}: {label} at {bottom:.1f}")
        self.blocks.append(dict(page=self.page, x=x, y=bottom, width=width, height=height, label=label))

    def begin(self, label):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.y = PAGE_H - 61
        c = self.c
        c.setStrokeColor(TEAL)
        c.setLineWidth(1.5)
        c.line(MARGIN, PAGE_H - 34, PAGE_W - MARGIN, PAGE_H - 34)
        c.setFont("Body-Bold", 7.3)
        c.setFillColor(MUTED)
        c.drawString(MARGIN, PAGE_H - 25, "KERNELBENCH-RL  /  SELECTED CASE STUDIES")
        c.setFont("Body", 7.3)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 25, label.upper())
        c.setStrokeColor(LINE)
        c.setLineWidth(.5)
        c.line(MARGIN, 37, PAGE_W - MARGIN, 37)
        c.setFont("Body", 7.3)
        c.setFillColor(MUTED)
        c.drawString(MARGIN, 24, "Eljo Mustafaj  |  November 27, 2026")
        c.drawRightString(PAGE_W - MARGIN, 24, f"{self.page} / 5")

    def p(self, text, style="body", gap=10, x=MARGIN, width=WIDTH, y=None, label=None):
        top = self.y if y is None else y
        para = Paragraph(text, self.styles[style])
        _, height = para.wrap(width, PAGE_H)
        para.drawOn(self.c, x, top - height)
        self.block(x, top - height, width, height, label or style)
        if y is None:
            self.y = top - height - gap
        return height

    def space(self, amount):
        self.y -= amount

    def band(self, text, height=48, style="equation", fill=PALE):
        self.c.setFillColor(fill)
        self.c.roundRect(MARGIN, self.y - height, WIDTH, height, 7, fill=1, stroke=0)
        paragraph = Paragraph(text, self.styles[style])
        _, h = paragraph.wrap(WIDTH - 22, height)
        paragraph.drawOn(self.c, MARGIN + 11, self.y - (height + h) / 2)
        self.block(MARGIN, self.y - height, WIDTH, height, "highlight band")
        self.y -= height + 12

    def arrow(self, x1, y1, x2, y2, color=TEAL):
        c = self.c
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.2)
        c.line(x1, y1, x2, y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 4
        path = c.beginPath()
        path.moveTo(x2, y2)
        path.lineTo(x2 - size * math.cos(angle - .5), y2 - size * math.sin(angle - .5))
        path.lineTo(x2 - size * math.cos(angle + .5), y2 - size * math.sin(angle + .5))
        path.close()
        c.drawPath(path, stroke=0, fill=1)

    def box(self, x, y, width, height, title, subtitle="", fill=PALE):
        c = self.c
        c.setFillColor(fill)
        c.setStrokeColor(LINE)
        c.setLineWidth(.6)
        c.roundRect(x, y, width, height, 6, fill=1, stroke=1)
        c.setFillColor(INK)
        c.setFont("Body-Bold", 9.2)
        c.drawCentredString(x + width / 2, y + height / 2 + (4 if subtitle else -3), title)
        if subtitle:
            c.setFont("Body", 7.7)
            c.setFillColor(MUTED)
            c.drawCentredString(x + width / 2, y + height / 2 - 11, subtitle)

    def pipeline(self):
        height = 82
        y = self.y - height
        w = (WIDTH - 3 * 17) / 4
        labels = [("Contract", "signature + math"), ("Generate", "one saved program"),
                  ("Verify", "references + guards"), ("Record", "PASS + hashes")]
        for index, (title, subtitle) in enumerate(labels):
            x = MARGIN + index * (w + 17)
            self.box(x, y + 19, w, 48, title, subtitle)
            if index < 3:
                self.arrow(x + w + 2, y + 43, x + w + 15, y + 43)
        self.block(MARGIN, y, WIDTH, height, "vector verification pipeline")
        self.y = y - 3
        self.p("Figure 1. " + esc(self.doc["pipeline_caption"]), "caption", gap=13)

    def matrix(self, x, y, w, h, rows=5, cols=7, highlight_row=None):
        c = self.c
        c.setStrokeColor(LINE)
        c.setLineWidth(.45)
        c.setFillColor(GRAY)
        c.rect(x, y, w, h, fill=1, stroke=1)
        if highlight_row is not None:
            c.setFillColor(colors.HexColor("#A7D5D5"))
            c.rect(x, y + highlight_row * h / rows, w, h / rows, fill=1, stroke=0)
        for i in range(1, rows):
            c.line(x, y + h * i / rows, x + w, y + h * i / rows)
        for i in range(1, cols):
            c.line(x + w * i / cols, y, x + w * i / cols, y + h)

    def gelu_figure(self):
        h = 128
        bottom = self.y - h
        x = MARGIN
        c = self.c
        self.matrix(x + 5, bottom + 36, 88, 56, highlight_row=2)
        c.setFont("Body-Bold", 9.1); c.setFillColor(INK)
        c.drawCentredString(x + 49, bottom + 103, "Input rows")
        c.setFont("Body", 8.2); c.setFillColor(MUTED)
        c.drawCentredString(x + 49, bottom + 22, "X + column bias")
        self.arrow(x + 100, bottom + 63, x + 123, bottom + 63)
        self.box(x + 126, bottom + 39, 105, 48, "FP32 add", "z = x + b")
        self.arrow(x + 234, bottom + 63, x + 255, bottom + 63)
        self.box(x + 258, bottom + 39, 104, 48, "GELU", "tanh approximation")
        self.arrow(x + 365, bottom + 63, x + 387, bottom + 63)
        self.matrix(x + 393, bottom + 36, 88, 56, highlight_row=2)
        c.setFont("Body-Bold", 9.1); c.setFillColor(INK)
        c.drawCentredString(x + 437, bottom + 103, "FP16 output")
        c.setFont("Body", 8.2); c.setFillColor(MUTED)
        c.drawCentredString(x + 307, bottom + 20, "One device program per row; masked tails")
        self.block(MARGIN, bottom, WIDTH, h, "vector GELU dataflow")
        self.y = bottom
        self.p("Figure 2. Row-wise fusion in the saved GELU program. The highlighted row is illustrative; tensor sizes are listed below.", "caption", gap=10)

    def matmul_figure(self):
        h = 158
        bottom = self.y - h
        x = MARGIN
        c = self.c
        self.matrix(x + 4, bottom + 66, 68, 61, rows=5, cols=4, highlight_row=2)
        self.matrix(x + 106, bottom + 66, 68, 61, rows=4, cols=5, highlight_row=2)
        c.setFillColor(INK); c.setFont("Body-Bold", 9.0)
        c.drawCentredString(x + 38, bottom + 138, "A tile")
        c.drawCentredString(x + 140, bottom + 138, "B tile")
        c.setFont("Body-Bold", 14); c.drawCentredString(x + 90, bottom + 93, "x")
        self.arrow(x + 180, bottom + 96, x + 214, bottom + 96)
        self.box(x + 218, bottom + 65, 129, 63, "FP32 accumulator", "64 x 64 output tile")
        c.setFont("Body", 8.1); c.setFillColor(MUTED)
        c.drawCentredString(x + 283, bottom + 139, "K loop: blocks of 32")
        self.arrow(x + 282, bottom + 61, x + 282, bottom + 43)
        self.box(x + 218, bottom + 1, 129, 39, "+ bias, then SiLU", "z sigmoid(z)")
        self.arrow(x + 350, bottom + 21, x + 387, bottom + 21)
        self.matrix(x + 393, bottom + 1, 85, 56, rows=5, cols=5, highlight_row=2)
        c.setFont("Body-Bold", 9.0); c.setFillColor(INK)
        c.drawCentredString(x + 435, bottom + 68, "FP16 store")
        c.setFont("Body", 8.1); c.setFillColor(MUTED)
        c.drawString(x + 4, bottom + 30, "Masked M, N and K tails")
        c.drawString(x + 4, bottom + 15, 'tl.dot input_precision="ieee"')
        self.block(MARGIN, bottom, WIDTH, h, "vector tiled matmul dataflow")
        self.y = bottom - 7
        self.p("Figure 3. Tiled reduction followed by a fused bias-and-SiLU epilogue. The accumulator is consumed before the output tile is stored.", "caption", gap=10)

    def timing_table(self, case):
        def cell(text, bold=False):
            style = self.styles["small"]
            return Paragraph(("<b>" + esc(text) + "</b>") if bold else esc(text), style)
        rows = [[cell("Implementation", True), cell("Median microseconds", True), cell("Reference / candidate", True)],
                [cell("Generated Triton program", True), cell(f"{case['candidate_ms'] * 1000:.3f}"), cell("1.000x")],
                [cell("Native PyTorch reference"), cell(f"{case['eager_ms'] * 1000:.3f}"), cell(f"{case['speedup_vs_eager']:.3f}x")],
                [cell("torch.compile reference"), cell(f"{case['compile_ms'] * 1000:.3f}"), cell(f"{case['speedup_vs_compile']:.3f}x")]]
        table = Table(rows, colWidths=[220, 128, WIDTH - 348], rowHeights=[27, 24, 24, 24])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), GRAY),
                                   ("BACKGROUND", (0, 1), (-1, 1), PALE),
                                   ("LINEBELOW", (0, 0), (-1, 0), .6, LINE),
                                   ("LINEBELOW", (0, -1), (-1, -1), .6, LINE),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 9),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        table.wrap(WIDTH, 1000)
        table.drawOn(self.c, MARGIN, self.y - 99)
        self.block(MARGIN, self.y - 99, WIDTH, 99, "timing table")
        self.y -= 110

    def code_hash(self, case):
        c = self.c
        c.setFillColor(GRAY)
        c.roundRect(MARGIN, self.y - 39, WIDTH, 39, 5, fill=1, stroke=0)
        c.setFillColor(MUTED); c.setFont("Body-Bold", 8)
        c.drawString(MARGIN + 10, self.y - 13, "COMPLETE GENERATED SOURCE / SHA-256")
        c.setFillColor(INK); c.setFont("Courier", 8)
        c.drawString(MARGIN + 10, self.y - 28, case["candidate_sha256"])
        self.block(MARGIN, self.y - 39, WIDTH, 39, "complete source hash")
        self.y -= 49

    def cover(self):
        self.begin("Technical case-study paper")
        self.p("TWO PROGRAMS. EXACT ARTIFACTS. BOUNDED CLAIMS.", "kicker", gap=10)
        title = "KernelBench-RL:<br/>Two Verified GPU Kernel<br/>Case Studies on an 8 GB Laptop"
        style = ParagraphStyle("title", fontName="Display-Bold", fontSize=27.5, leading=32, textColor=INK)
        paragraph = Paragraph(title, style)
        _, h = paragraph.wrap(WIDTH, 1000)
        paragraph.drawOn(self.c, MARGIN, self.y - h)
        self.block(MARGIN, self.y - h, WIDTH, h, "exact paper title")
        self.y -= h + 17
        self.p("<b>Eljo Mustafaj</b> &nbsp;&nbsp; | &nbsp;&nbsp; November 27, 2026", "body", gap=17)
        self.band("<b>2/2 selected featured programs verified</b><br/><font size=9.2>Chosen after verification; not a complete-evaluation accuracy estimate.</font>", height=57)
        self.p("ABSTRACT", "kicker", gap=5)
        self.p(esc(self.doc["abstract"]), gap=12)
        self.pipeline()
        self.p(esc(self.doc["opening"]), gap=12)
        c = self.c
        y = self.y - 65
        for i, (title, desc) in enumerate([("01  Bias + GELU", "Broadcast add and row-wise activation"),
                                           ("02  Matmul + bias + SiLU", "Tiled reduction and fused epilogue")]):
            xx = MARGIN + i * (WIDTH / 2 + 6)
            ww = WIDTH / 2 - 6
            c.setFillColor(PALE if i == 0 else GRAY)
            c.roundRect(xx, y, ww, 65, 6, fill=1, stroke=0)
            c.setFillColor(INK); c.setFont("Body-Bold", 10.2)
            c.drawString(xx + 12, y + 41, title)
            c.setFillColor(MUTED); c.setFont("Body", 8.4)
            c.drawString(xx + 12, y + 23, desc)
        self.block(MARGIN, y, WIDTH, 65, "featured case cards")

    def case_page(self, key, case, index):
        data = self.doc[key]
        self.begin("Featured program " + str(index))
        self.p(f"SELECTED CASE {index:02d}  /  VERIFIED FP16 PROGRAM", "kicker", gap=6)
        self.p("Bias plus GELU" if key == "gelu" else "Matmul + bias + SiLU", "h1", gap=8)
        self.p("Saved record: <b>" + esc(case["run_id"]) + "</b>", "tiny", gap=8)
        self.p(esc(data["purpose"]), gap=10)
        equation = "Y = GELU(X + b)" if key == "gelu" else "Y = SiLU(A B + b) = (A B + b) sigmoid(A B + b)"
        if key == "gelu":
            equation += '<br/><font size="8.4">' + esc(data["equation"]) + '</font>'
        self.band(equation, height=51 if key == "gelu" else 38)
        (self.gelu_figure if key == "gelu" else self.matmul_figure)()
        self.p(esc(data["implementation"]), gap=10)
        self.timing_table(case)
        self.p(esc(data["measured_note"]), "caption", gap=10)
        self.p("<b>Correctness evidence.</b> " + esc(data["canonical"]) + " " + esc(data["stress"]) +
               f" Seeds {case['seed']} and {case['independent_seed']}. Recorded maximum absolute error: <b>{case['max_abs_err']:.8g}</b>. Status: <b>PASS</b> under rtol = atol = 0.01.",
               "small", gap=10)
        self.code_hash(case)
        self.p(esc(data["takeaway"]), "small", gap=0)

    def verification(self):
        self.begin("Verification and measurement")
        self.p("WHAT THE SAVED PASS RECORDS ESTABLISH", "kicker", gap=6)
        self.p("From source to checked execution", "h1", gap=12)
        for index, step in enumerate(self.doc["verification_steps"], 1):
            title = Paragraph(esc(step["title"]), self.styles["h2"])
            body = Paragraph(esc(step["body"]), self.styles["small"])
            _, th = title.wrap(WIDTH - 45, 1000)
            _, bh = body.wrap(WIDTH - 45, 1000)
            h = th + bh + 13
            c = self.c
            c.setFillColor(TEAL)
            c.circle(MARGIN + 11, self.y - 10, 11, fill=1, stroke=0)
            c.setFont("Body-Bold", 9); c.setFillColor(WHITE)
            c.drawCentredString(MARGIN + 11, self.y - 13.5, str(index))
            title.drawOn(c, MARGIN + 34, self.y - th)
            body.drawOn(c, MARGIN + 34, self.y - th - 3 - bh)
            self.block(MARGIN, self.y - h, WIDTH, h, "verification step " + str(index))
            self.y -= h + 4
        self.space(4)
        self.p("Timing boundary", "h2", gap=5)
        self.p(esc(self.doc["timing_boundary"]), "small", gap=13)
        rows = [("GPU", self.evidence["gpu"] + " (8 GB class)"),
                ("Arithmetic", "FP16 inputs/output; generated FP32 arithmetic/accumulation"),
                ("Software", f"PyTorch {self.evidence['torch']}; CUDA {self.evidence['cuda']}; Triton {self.evidence['triton']}"),
                ("Record boundary", "Sanitized child process for trusted-candidate development")]
        cells = [[Paragraph("<b>" + esc(a) + "</b>", self.styles["tiny"]), Paragraph(esc(b), self.styles["tiny"])] for a, b in rows]
        table = Table(cells, colWidths=[89, WIDTH - 89])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GRAY),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 9),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        _, h = table.wrap(WIDTH, 1000)
        table.drawOn(self.c, MARGIN, self.y - h)
        self.block(MARGIN, self.y - h, WIDTH, h, "hardware software table")
        self.y -= h + 12
        self.p("These checks cover the registered input cases and saved procedure. They do not constitute a formal proof for every shape, input distribution, or execution environment.", "small", gap=0)

    def provenance(self):
        self.begin("Provenance and references")
        self.p("A TRACEABLE SELECTED-EXAMPLE RECORD", "kicker", gap=6)
        self.p("Checkpoint context and interpretation", "h1", gap=11)
        self.p(esc(self.doc["training_context"]), gap=11)
        self.p(esc(self.doc["provenance_note"]), "small", gap=10)
        self.p("Artifact identities", "h2", gap=7)
        fields = [("Executed study source", self.evidence["source_commit"]),
                  ("Model revision", self.evidence["model_revision"]),
                  ("Training configuration", self.evidence["config_hash"]),
                  ("Final trainable checkpoint SHA-256", self.evidence["checkpoint_weights_sha256"]),
                  ("Original saved record-file SHA-256", self.evidence["cases"][0]["source_record_file_sha256"])]
        c = self.c
        start = self.y
        for title, value in fields:
            c.setFillColor(MUTED); c.setFont("Body-Bold", 8)
            c.drawString(MARGIN, self.y - 9, title)
            c.setFillColor(INK); c.setFont("Courier", 8)
            c.drawString(MARGIN, self.y - 23, value)
            self.y -= 33
        self.block(MARGIN, self.y, WIDTH, start - self.y, "provenance identity ledger")
        self.space(5)
        self.p(esc(self.doc["interpretation"]), "small", gap=12)
        self.p("References", "h2", gap=8)
        short_venues = {"kernelbench2025": "ICML, PMLR 267, 2025.", "triton2019": "MAPL, 2019. doi:10.1145/3315508.3329973.",
                        "qwen25coder2024": "arXiv:2409.12186, 2024.", "qlora2023": "NeurIPS 36, 2023.",
                        "deepseekmath2024": "arXiv:2402.03300, 2024."}
        link_labels = {"kernelbench2025": "PMLR primary record", "triton2019": "Publisher DOI", "qwen25coder2024": "arXiv primary record",
                       "qlora2023": "NeurIPS primary record", "deepseekmath2024": "arXiv primary record"}
        for index, ref in enumerate(self.refs, 1):
            authors = ", ".join(ref["authors"][:3]) + (", et al." if len(ref["authors"]) > 3 else ".")
            ending = "" if ref["title"].endswith(("?", "!", ".")) else "."
            text = f"<b>[{index}]</b> {esc(authors)} <i>{esc(ref['title'])}</i>{ending} {esc(short_venues[ref['id']])} "
            text += f'<link href="{esc(ref["url"])}" color="#087E82">{esc(link_labels[ref["id"]])}</link>.'
            self.p(text, "reference", gap=7)
        self.space(2)
        self.p("Selected-case scope: two chosen positive examples, both verified. No complete-evaluation accuracy, causal training gain, new method, or application-level acceleration is inferred.", "tiny", gap=0)

    def save(self, output, source_path):
        self.cover()
        self.case_page("gelu", self.evidence["cases"][0], 1)
        self.case_page("matmul", self.evidence["cases"][1], 2)
        self.verification()
        self.provenance()
        assert self.page == 5
        self.c.save()
        qa = dict(pages=5, renderer="ReportLab", native_vector_figures=3,
                  font_files=self.font_files,
                  source_manuscript_sha256=digest(source_path), pdf_sha256=digest(output),
                  pdf_bytes=output.stat().st_size, layout_blocks=self.blocks,
                  bounds_check="All content blocks remained on-page and above the footer; visual inspection is separately required.")
        (HERE / "render-metadata.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in qa.items() if key != "layout_blocks"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, help="Optional TTF font directory; common Windows/Linux families are detected")
    args = parser.parse_args()
    manuscript_path = HERE / "manuscript.json"
    manuscript = json.loads(manuscript_path.read_text(encoding="utf-8"))
    evidence_path = HERE / manuscript["evidence_file"]
    assert digest(evidence_path) == manuscript["evidence_file_sha256"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    bibliography = json.loads((HERE / "bibliography.json").read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Paper(args.out, manuscript, evidence, bibliography, manuscript_path, args.font_dir).save(args.out, manuscript_path)


if __name__ == "__main__":
    main()
