"""Render the structured manuscript as a scientific preprint, with draft guards.

Example (layout preview only)::
    python build_paper.py --baseline-comparison ../../outputs/kernelbench-rl/experiments/rtx5060/full30/report/baseline_comparison.json --render-pages

Paths on the command line are relative to the current working directory. Paths
inside figures JSON are relative to that JSON's directory. With no --final,
output is restricted to this script's preview/ directory. A final render requires
resolved status, actual publication_date, pending_results.required=false and no
rendered DRAFT/TBD/placeholders.

Optional top-level manuscript tables (or --tables JSON list):
  {"id":"retry_fresh", "section_id":"corrected_protocol", "after_paragraph":6,
   "caption":"Fresh held-out results, separately by model track.",
   "columns":[{"key":"method","label":"Method","weight":2},
              {"key":"passed","label":"Passed / N","weight":1,"align":"RIGHT"}],
   "rows":[{"method":"base_pass1","passed":"0/3"}], "notes":["Recorded failures remain in N."]}
after_paragraph is 1-based; omit it to insert after the section. Rows may instead
be arrays and columns strings. Tables repeat the header on page breaks. The
example row above illustrates the input schema; it is not inserted automatically.

Optional figures (top-level or --figures JSON list):
  {"id":"baseline_plot","section_id":"original_evidence","after_paragraph":2,
   "path":"path/to/actual.png","caption":"Measured comparison...",
   "max_height_inches":3.5}

--baseline-comparison takes the existing released baseline_comparison.json. It
creates one compact vector figure from those measured records, without running
benchmarks or changing the source data. Baseline bars are fixed implementations,
not generated-model results. Arbitrary TeX is deliberately not interpreted. Known
draft equation IDs are supported, or provide display_text/display_markup explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import datetime as dt
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Sequence
from urllib.parse import urlparse
import uuid

from reportlab import rl_config
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Flowable, Frame, HRFlowable, Image,
    KeepTogether, LongTable, PageBreak, PageTemplate, Paragraph, Spacer, Table,
    TableStyle,
)
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parent
PAGE_W, PAGE_H = letter
MARGIN_X = 66
MARGIN_TOP = 55
MARGIN_BOTTOM = 53
CONTENT_W = PAGE_W - MARGIN_X * 2
INK = colors.HexColor("#1e2833")
MUTED = colors.HexColor("#596777")
BLUE = colors.HexColor("#214d79")
TEAL = colors.HexColor("#297f82")
RULE = colors.HexColor("#ccd3da")
BODY_FONT = "PaperGeorgia"
BODY_BOLD = "PaperGeorgiaBold"
BODY_ITALIC = "PaperGeorgiaItalic"
SANS_FONT = "PaperCalibri"
SANS_BOLD = "PaperCalibriBold"
CODE_FONT = "PaperConsolas"
MATH_FONT = "PaperCambriaMath"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda val: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {val}")))


def clean_text(value: Any) -> str:
    text = str(value)
    # ASCII hyphens avoid nonbreaking/unsupported dash variants in PDF extraction.
    return text.translate(str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-",
                                        "\u2013": "-", "\u2014": "-", "\u2212": "-",
                                        "\u00a0": " "}))


def register_fonts(font_dir: Path | None = None) -> dict[str, str]:
    candidates = [font_dir] if font_dir else []
    candidates += [Path("C:/Windows/Fonts"), ROOT / "fonts"]
    mapping = {BODY_FONT: "georgia.ttf", BODY_BOLD: "georgiab.ttf",
               BODY_ITALIC: "georgiai.ttf", "PaperGeorgiaBoldItalic": "georgiaz.ttf",
               SANS_FONT: "calibri.ttf", SANS_BOLD: "calibrib.ttf", CODE_FONT: "consola.ttf"}
    selected = {}
    for name, filename in mapping.items():
        path = next((folder / filename for folder in candidates if folder and (folder / filename).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing {filename}; pass --font-dir containing the Georgia/Calibri/Consolas TTF files")
        pdfmetrics.registerFont(TTFont(name, str(path)))
        selected[name] = str(path)
    pdfmetrics.registerFontFamily(BODY_FONT, normal=BODY_FONT, bold=BODY_BOLD,
                                 italic=BODY_ITALIC, boldItalic="PaperGeorgiaBoldItalic")
    pdfmetrics.registerFontFamily(SANS_FONT, normal=SANS_FONT, bold=SANS_BOLD,
                                 italic=SANS_FONT, boldItalic=SANS_BOLD)
    pdfmetrics.registerFontFamily(CODE_FONT, normal=CODE_FONT, bold=CODE_FONT,
                                 italic=CODE_FONT, boldItalic=CODE_FONT)
    math_path = next((folder / "cambria.ttc" for folder in candidates if folder and (folder / "cambria.ttc").is_file()),None)
    if math_path is None:
        raise FileNotFoundError("Missing cambria.ttc; the Cambria Math subfont is required for equation glyphs")
    pdfmetrics.registerFont(TTFont(MATH_FONT,str(math_path),subfontIndex=1))
    pdfmetrics.registerFontFamily(MATH_FONT,normal=MATH_FONT,bold=MATH_FONT,italic=MATH_FONT,boldItalic=MATH_FONT)
    selected[MATH_FONT]=str(math_path)
    return selected


def styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName=BODY_FONT, textColor=INK, allowWidows=0, allowOrphans=0,
                splitLongWords=True, spaceShrinkage=.035)
    return {
        "body": ParagraphStyle("PaperBody", fontSize=10.6, leading=15.55,
                               alignment=TA_JUSTIFY, spaceAfter=7.2, **base),
        "heading": ParagraphStyle("PaperHeading", fontSize=13, leading=17.2,
                                  fontName=BODY_BOLD, textColor=INK,
                                  spaceBefore=14, spaceAfter=8, keepWithNext=True),
        "title": ParagraphStyle("PaperTitle", fontSize=23.5, leading=28.2,
                                fontName=BODY_BOLD, textColor=INK, spaceAfter=15),
        "author": ParagraphStyle("PaperAuthor", fontSize=12.4, leading=17,
                                 fontName=BODY_FONT, textColor=INK, spaceAfter=4),
        "metadata": ParagraphStyle("PaperMetadata", fontSize=8.6, leading=12,
                                   fontName=SANS_FONT, textColor=MUTED, spaceAfter=2),
        "abstract_heading": ParagraphStyle("AbstractHeading", fontSize=10.5, leading=14,
                                           fontName=BODY_BOLD, textColor=INK, spaceAfter=5,
                                           keepWithNext=True),
        "abstract": ParagraphStyle("PaperAbstract", fontSize=9.65, leading=14.2,
                                   alignment=TA_JUSTIFY, spaceAfter=12, **base),
        "caption": ParagraphStyle("PaperCaption", fontSize=9.0, leading=12.4,
                                  alignment=TA_LEFT, spaceAfter=6, **base),
        "table_caption": ParagraphStyle("TableCaption", fontSize=9.1, leading=12.6,
                                        alignment=TA_LEFT, keepWithNext=True, spaceAfter=6, **base),
        "note": ParagraphStyle("PaperNote", fontSize=8.1, leading=11.5,
                               fontName=SANS_FONT, textColor=MUTED, spaceAfter=7),
        "cell": ParagraphStyle("PaperCell", fontSize=8.7, leading=12,
                               fontName=BODY_FONT, textColor=INK, splitLongWords=True),
        "cell_header": ParagraphStyle("PaperCellHeader", fontSize=8.5, leading=11.5,
                                      fontName=SANS_BOLD, textColor=INK),
        "reference": ParagraphStyle("PaperReference", fontSize=8.6, leading=12.15,
                                    leftIndent=20, firstLineIndent=-20,
                                    spaceAfter=9, alignment=TA_LEFT, **base),
        "equation": ParagraphStyle("PaperEquation", fontSize=10.1, leading=16,
                                   fontName=MATH_FONT, textColor=INK, alignment=TA_CENTER,
                                   spaceBefore=4, spaceAfter=4),
        "equation_number": ParagraphStyle("EquationNumber", fontSize=9.4, leading=16,
                                          fontName=BODY_FONT, textColor=INK, alignment=TA_RIGHT),
    }


def rich_text(value: Any, citation_numbers: set[int] | None = None) -> str:
    text = escape(clean_text(value))
    text = re.sub(r"`([^`]+)`", lambda m: f'<font name="{CODE_FONT}" size="9.1">{m.group(1)}</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    if citation_numbers:
        def cite(match):
            nums = [int(x.strip()) for x in match.group(1).split(",")]
            if any(n not in citation_numbers for n in nums):
                raise ValueError(f"Unknown numeric citation {match.group(0)}")
            return "[" + ",".join(f'<link href="#ref-{n}" color="#214d79">{n}</link>' for n in nums) + "]"
        text = re.sub(r"\[([0-9]+(?:\s*,\s*[0-9]+)*)\]", cite, text)
    text = re.sub(r"(\[DRAFT:.*?\])", r'<font color="#9c332e">\1</font>', text, flags=re.DOTALL)
    return text.replace("\n", "<br/>")


def valid_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError(f"Only public HTTP(S) bibliography links are supported: {url}")
    return escape(url, quote=True)


class PaperDoc(BaseDocTemplate):
    def __init__(self, filename: str, *, manuscript: dict, draft: bool, manuscript_sha256: str, **kwargs):
        self.manuscript = manuscript
        self.is_draft = draft
        super().__init__(filename, pagesize=letter, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                         topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
                         title=manuscript["title"], author=manuscript["author"]["display_name"],
                         subject="KernelBench-RL manuscript SHA-256: "+manuscript_sha256,
                         creator="ReportLab scientific manuscript renderer", **kwargs)
        frame = Frame(MARGIN_X, MARGIN_BOTTOM, CONTENT_W, PAGE_H-MARGIN_TOP-MARGIN_BOTTOM,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="Paper", frames=[frame], onPage=self.decorate_page))

    def decorate_page(self, canv, doc):
        canv.saveState()
        if self.is_draft:
            canv.saveState()
            canv.setFillColor(colors.HexColor("#dce2e8"))
            try:
                canv.setFillAlpha(.24)
            except AttributeError:
                pass
            canv.translate(PAGE_W/2, PAGE_H/2)
            canv.rotate(43)
            canv.setFont(SANS_BOLD, 72)
            canv.drawCentredString(0, 0, "DRAFT")
            canv.restoreState()
        canv.setStrokeColor(RULE)
        canv.setLineWidth(.45)
        if doc.page > 1:
            canv.setFont(SANS_FONT, 8.5)
            canv.setFillColor(MUTED)
            canv.drawString(MARGIN_X, PAGE_H-31, "KernelBench-RL")
            canv.drawRightString(PAGE_W-MARGIN_X, PAGE_H-31,
                                clean_text(self.manuscript["author"]["display_name"]))
            canv.line(MARGIN_X, PAGE_H-37, PAGE_W-MARGIN_X, PAGE_H-37)
        canv.line(MARGIN_X, 38, PAGE_W-MARGIN_X, 38)
        canv.setFont(SANS_FONT, 8)
        canv.setFillColor(MUTED)
        footer = "DRAFT - results and publication metadata not finalized" if self.is_draft else "Research manuscript"
        canv.drawString(MARGIN_X, 24, footer)
        canv.drawRightString(PAGE_W-MARGIN_X, 24, str(doc.page))
        canv.restoreState()

    def afterFlowable(self, flowable):
        bookmark = getattr(flowable, "bookmark_name", None)
        if bookmark:
            self.canv.bookmarkPage(bookmark)
            self.canv.addOutlineEntry(flowable.getPlainText(), bookmark, level=0, closed=False)


def paragraph(text, style, citations):
    return Paragraph(rich_text(text, citations), style)


def format_reference(ref: dict) -> str:
    number = ref["number"]
    authors = ", ".join(clean_text(a) for a in ref["authors"])
    year = "n.d." if ref.get("type") == "technical_documentation" else str(ref["year"])
    year_stop = "" if year.endswith(".") else "."
    title = clean_text(ref["title"])
    title_stop = "" if title.endswith((".", "?", "!")) else "."
    parts = [f'<a name="ref-{number}"/>[{number}] {escape(authors)}. {year}{year_stop}',
             f'<i>{escape(title)}</i>{title_stop}']
    venue = clean_text(ref.get("venue", ""))
    if venue and ref.get("type") != "technical_documentation":
        volume = clean_text(ref.get("volume", ""))
        issue = clean_text(ref.get("issue", ""))
        pagination = clean_text(ref.get("pages", ""))
        loc = volume + (f"({issue})" if issue else "")
        if pagination:
            loc += (": " if loc else "pp. ") + pagination
        parts.append(escape(venue + (f", {loc}" if loc else "")) + ".")
    if ref.get("type") == "preprint":
        version = ref.get("version", "")
        ident = ref.get("identifier", "") + version
        if ident:
            parts.append(f'<link href="{valid_url(ref["url"])}" color="#214d79">{escape(ident)}</link>.')
    if ref.get("doi"):
        doi = ref["doi"]
        parts.append(f'<link href="{valid_url("https://doi.org/"+doi)}" color="#214d79">doi:{escape(doi)}</link>.')
    elif ref.get("url"):
        label = {"technical_documentation": "Online documentation",
                 "software_release": "Official model release"}.get(ref.get("type"), "Publisher record")
        parts.append(f'<link href="{valid_url(ref["url"])}" color="#214d79">{label}</link>.')
    if ref.get("accessed_date"):
        parts.append("Accessed " + escape(ref["accessed_date"]) + ".")
    return " ".join(parts)


def table_flowables(spec: dict, number: int, st, citations) -> list:
    columns = spec.get("columns", [])
    if not columns or not isinstance(columns, list):
        raise ValueError(f"Table {spec.get('id')} requires columns")
    columns = [dict(key=str(i), label=value, weight=1) if isinstance(value, str) else value
               for i, value in enumerate(columns)]
    weights = [float(col.get("weight", 1)) for col in columns]
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Table column weights must be finite and positive")
    widths = [CONTENT_W*w/sum(weights) for w in weights]
    data = [[paragraph(col["label"], st["cell_header"], citations) for col in columns]]
    for row in spec.get("rows", []):
        values = [row.get(col["key"], "") for col in columns] if isinstance(row, dict) else row
        if len(values) != len(columns):
            raise ValueError(f"Wrong number of cells in table {spec.get('id')}")
        cells = []
        for i, value in enumerate(values):
            align = {"LEFT": TA_LEFT, "RIGHT": TA_RIGHT, "CENTER": TA_CENTER}.get(columns[i].get("align", "LEFT"))
            cell_style = ParagraphStyle(f"cell-{i}", parent=st["cell"], alignment=align)
            cells.append(paragraph(value, cell_style, citations))
        data.append(cells)
    tab = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    commands = [("VALIGN", (0,0), (-1,-1), "TOP"), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#edf1f5")),
                ("LINEABOVE", (0,0), (-1,0), .8, INK), ("LINEBELOW", (0,0), (-1,0), .5, RULE),
                ("LINEBELOW", (0,-1), (-1,-1), .7, INK),
                ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
                ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]
    for i in range(2, len(data), 2):
        commands.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#f8fafb")))
    tab.setStyle(TableStyle(commands))
    flow = [CondPageBreak(115), paragraph(f"Table {number}. {spec['caption']}", st["table_caption"], citations), tab]
    for note in spec.get("notes", []):
        flow.append(paragraph(note, st["note"], citations))
    flow.append(Spacer(1, 10))
    return flow


def baseline_drawing(path: Path) -> tuple[Drawing, str, list[dict]]:
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError("Baseline comparison must be a nonempty JSON list")
    seen = set()
    for record in records:
        key = (record["op_name"], record["dtype"])
        if key in seen:
            raise ValueError(f"Duplicate baseline operator/dtype {key}")
        seen.add(key)
    groups = []
    for dtype in ("fp32", "bf16"):
        rows = [row for row in records if row["dtype"] == dtype]
        if not rows:
            raise ValueError(f"No {dtype} measurements in baseline source")
        for method in ("eager", "compile"):
            values = [row[f"speedup_{method}"] for row in rows if row.get(f"speedup_{method}") is not None]
            if not values or any(not isinstance(v, (int,float)) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Baseline speedups must be positive finite measured values")
            groups.append(dict(dtype=dtype, method=method, n=len(values),
                               speedup=math.exp(math.fsum(math.log(v) for v in values)/len(values))))
    width, height = CONTENT_W, 223
    drawing = Drawing(width, height)
    x0, y0, plot_w, plot_h = 45, 42, width-67, 143
    ymax = math.ceil(max(g["speedup"] for g in groups)*1.16)
    for value in range(ymax+1):
        y = y0 + plot_h*value/ymax
        drawing.add(Line(x0,y,x0+plot_w,y,strokeColor=RULE,strokeWidth=.35))
        drawing.add(String(x0-8,y-3,str(value),fontName=SANS_FONT,fontSize=8.5,textAnchor="end",fillColor=MUTED))
    drawing.add(Line(x0,y0,x0,y0+plot_h,strokeColor=INK,strokeWidth=.65))
    drawing.add(Line(x0,y0,x0+plot_w,y0,strokeColor=INK,strokeWidth=.65))
    unity_y = y0+plot_h/ymax
    drawing.add(Line(x0,unity_y,x0+plot_w,unity_y,strokeColor=MUTED,strokeWidth=.75,strokeDashArray=[3,3]))
    drawing.add(String(x0, height-13, "Geometric mean speedup (x)",fontName=SANS_BOLD,fontSize=9.3,fillColor=INK))
    bar_width = 42
    for group_index, method in enumerate(("eager", "compile")):
        center = x0+plot_w*(.27 if group_index==0 else .77)
        for dtype_index, dtype in enumerate(("fp32", "bf16")):
            item = next(g for g in groups if g["method"]==method and g["dtype"]==dtype)
            x = center + (dtype_index-.5)*55 - bar_width/2
            bar_h = plot_h*item["speedup"]/ymax
            color = BLUE if dtype=="fp32" else TEAL
            drawing.add(Rect(x,y0,bar_width,bar_h,fillColor=color,strokeColor=None))
            drawing.add(String(x+bar_width/2,y0+bar_h+6,f"{item['speedup']:.3f}x",fontName=SANS_BOLD,fontSize=9.0,textAnchor="middle",fillColor=INK))
            drawing.add(String(x+bar_width/2,y0-13,f"n={item['n']}",fontName=SANS_FONT,fontSize=8.1,textAnchor="middle",fillColor=MUTED))
        drawing.add(String(center,10,"Native PyTorch eager" if method=="eager" else "Verified torch.compile",fontName=SANS_FONT,fontSize=9,textAnchor="middle",fillColor=INK))
    for x, label, color in ((width-119,"fp32",BLUE),(width-60,"bf16",TEAL)):
        drawing.add(Rect(x,height-17,9,9,fillColor=color,strokeColor=None))
        drawing.add(String(x+14,height-15,label,fontName=SANS_FONT,fontSize=8.5,fillColor=MUTED))
    caption = ("Fixed Triton baseline comparison on the RTX 5060 Laptop GPU, using the released CUDA-Graph measurements. "
               "Bars give geometric means across operators after within-operator aggregation over three input seeds. "
               "The SSM compiled reference is unavailable (compiled n=29; eager n=30). These are fixed implementations, not learned-policy gains.")
    return drawing, caption, groups


def figure_flowables(spec: dict, number: int, st, citations, base_dir: Path) -> list:
    source = Path(spec["path"])
    source = source if source.is_absolute() else base_dir / source
    if not source.is_file():
        raise FileNotFoundError(f"Figure not found: {source}")
    image = Image(str(source))
    scale = min(CONTENT_W/image.imageWidth,
                float(spec.get("max_height_inches", 4.7))*inch/image.imageHeight)
    image.drawWidth = image.imageWidth*scale
    image.drawHeight = image.imageHeight*scale
    image.hAlign = "CENTER"
    return [CondPageBreak(min(image.drawHeight+75, PAGE_H-MARGIN_TOP-MARGIN_BOTTOM)),
            KeepTogether([image, Spacer(1,7), paragraph(f"Figure {number}. {spec['caption']}", st["caption"], citations)]),
            Spacer(1, 8)]


KNOWN_EQUATIONS = {
    "group_advantage": dict(section_id="system_methods", after_paragraph=6,
        markup="A<sub>i</sub> = (r<sub>i</sub> - mean(r)) / (std(r) + ε)"),
    "completion_nll": dict(section_id="corrected_protocol", after_paragraph=1,
        markup="L<sub>SFT</sub> = -(1/T) Σ<sub>t=1..T</sub> log π<sub>θ</sub>(y<sub>t</sub> | x, y<sub>&lt;t</sub>)"),
    "gated_reward": dict(section_id="corrected_protocol", after_paragraph=2,
        markup="r<sub>i</sub><super>train</super> = I[PASS ∧ compiled ∧ correct ∧ ¬cheat] · r<sub>i</sub><super>verifier</super>"),
}


def final_guard(manuscript, tables, figures, draft):
    if draft:
        return
    if not str(manuscript.get("status", "")).upper().startswith("FINAL"):
        raise ValueError("Final render refused: manuscript status must be FINAL, not DRAFT or pending")
    if manuscript.get("pending_results", {}).get("required", False):
        raise ValueError("Final render refused: pending_results.required is true")
    date = manuscript.get("publication_date")
    try:
        dt.date.fromisoformat(date)
    except (ValueError, TypeError):
        raise ValueError("Final render requires the actual publication_date in YYYY-MM-DD format") from None
    strings = [manuscript["title"], manuscript["abstract"]]
    strings += [str(p) for sec in manuscript["sections"] for p in sec.get("paragraphs", [])]
    strings += [json.dumps(obj) for obj in tables+figures]
    if re.search(r"\[DRAFT\b|\[PENDING\b|\bTBD\b|\[ACTUAL_[^\]]*\]|\[COUNT[^\]]*\]|\[H/N\]|\[S/N\]", " ".join(strings), re.IGNORECASE):
        raise ValueError("Final render refused: rendered content contains unresolved draft/result placeholders")


def check_glyph_coverage(flowables):
    """Fail rather than silently rendering missing mathematical glyph boxes."""
    missing=set()
    def visit(flow):
        if isinstance(flow,Paragraph):
            for frag in flow.frags:
                name=getattr(frag,"fontName",None)
                text=getattr(frag,"text","")
                if name:
                    font=pdfmetrics.getFont(name)
                    widths=getattr(font.face,"charWidths",{})
                    for character in text:
                        if ord(character)>31 and ord(character) not in widths:
                            missing.add((name,f"U+{ord(character):04X}"))
        elif isinstance(flow,Table):
            for row in flow._cellvalues:
                for cell in row:
                    for child in cell if isinstance(cell,(list,tuple)) else [cell]:
                        visit(child)
        elif isinstance(flow,KeepTogether):
            for child in flow._content:
                visit(child)
    for flow in flowables:
        visit(flow)
    if missing:
        raise ValueError(f"Missing embedded-font glyphs: {sorted(missing)}")


def rendered_font_report(pdf):
    """Core fonts may occur in empty drawing defaults; inspect actual text use."""
    used={}
    for page in pdf.pages:
        resources=page.get("/Resources",{}).get_object()
        fonts=resources.get("/Font",{}).get_object()
        current=None
        stack=[]
        for operands,operator in ContentStream(page.get_contents(),pdf).operations:
            if operator==b"q":
                stack.append(current)
            elif operator==b"Q":
                current=stack.pop() if stack else current
            elif operator==b"Tf":
                current=operands[0]
            elif operator in (b"Tj",b"TJ",b"'",b'"') and current is not None:
                texts=operands[0] if operator==b"TJ" else [operands[-1]]
                if not any(isinstance(text,(str,bytes)) and len(text)>0 for text in texts):
                    continue
                font=fonts[current].get_object()
                descriptor=font.get("/FontDescriptor")
                descriptor=descriptor.get_object() if descriptor else {}
                used[str(font.get("/BaseFont"))]=any(key in descriptor for key in ("/FontFile","/FontFile2","/FontFile3"))
    return used


def build_pdf(manuscript_path: Path, bibliography_path: Path, output: Path, *, draft=True,
              tables_path: Path | None=None, figures_path: Path | None=None,
              baseline_path: Path | None=None, font_dir: Path | None=None) -> dict:
    manuscript_bytes = manuscript_path.read_bytes()
    manuscript = json.loads(manuscript_bytes.decode('utf-8'),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite manuscript JSON: '+value)))
    bibliography = read_json(bibliography_path)
    refs = bibliography["references"]
    if [ref["number"] for ref in refs] != list(range(1,len(refs)+1)):
        raise ValueError("Bibliography numbering must be contiguous and ordered from 1")
    if manuscript.get("citation_number_order") and manuscript["citation_number_order"] != [r["id"] for r in refs]:
        raise ValueError("Manuscript citation order and bibliography IDs differ")
    citations = set(ref["number"] for ref in refs)
    tables = list(manuscript.get("tables", []))
    figures = list(manuscript.get("figures", []))
    if tables_path:
        tables.extend(read_json(tables_path))
    if figures_path:
        extra = read_json(figures_path)
        for spec in extra:
            spec["_base_dir"] = str(figures_path.resolve().parent)
        figures.extend(extra)
    final_guard(manuscript, tables, figures, draft)
    if draft and not output.resolve().is_relative_to((ROOT/"preview").resolve()):
        raise ValueError("Draft output is restricted to work/paper-source/preview; use --final only after resolving results")
    fonts = register_fonts(font_dir)
    st = styles()
    story = []
    story.append(paragraph(manuscript["title"], st["title"], citations))
    story.append(paragraph(manuscript["author"]["display_name"], st["author"], citations))
    if manuscript["author"].get("affiliation"):
        story.append(paragraph(manuscript["author"]["affiliation"], st["metadata"], citations))
    if draft:
        story.append(paragraph("Draft for layout review - empirical retry results pending", st["metadata"], citations))
    else:
        story.append(paragraph(manuscript["publication_date"] + " | Research manuscript", st["metadata"], citations))
    if manuscript.get("repository_url"):
        url = manuscript["repository_url"]
        story.append(Paragraph(f'<link href="{valid_url(url)}" color="#214d79">Code and experimental artifacts: {escape(url)}</link>', st["metadata"]))
    story.extend([Spacer(1,14), HRFlowable(width="100%",thickness=.65,color=RULE), Spacer(1,14),
                  paragraph("Abstract", st["abstract_heading"], citations),
                  paragraph(manuscript["abstract"], st["abstract"], citations), Spacer(1,5)])
    attachments = {}
    section_ids = {sec["id"] for sec in manuscript["sections"]}
    for kind, specs in (("table", tables), ("figure", figures)):
        for index, spec in enumerate(specs,1):
            sid = spec.get("section_id")
            if sid not in section_ids:
                raise ValueError(f"Unknown section_id {sid} for {kind}")
            attachments.setdefault((sid, spec.get("after_paragraph")), []).append((kind,index,spec))
    baseline_stats = None
    if baseline_path:
        drawing, caption, baseline_stats = baseline_drawing(baseline_path)
        figure_number = len(figures)+1
        attachments.setdefault(("original_evidence",2), []).append(("baseline",figure_number,dict(drawing=drawing,caption=caption)))
    equations = []
    for equation in manuscript.get("equations", []):
        defaults = KNOWN_EQUATIONS.get(equation.get("id"), {})
        if not defaults and not (equation.get("display_markup") or equation.get("display_text")):
            raise ValueError("Arbitrary TeX is not interpreted; provide display_markup or display_text for equation " + equation.get("id", "unknown"))
        eq = {**defaults, **equation}
        sid = eq.get("section_id")
        if sid not in section_ids:
            raise ValueError(f"Unknown equation section {sid}")
        equations.append(eq)
        attachments.setdefault((sid,eq.get("after_paragraph")), []).append(("equation",0,eq))

    equation_counter = 0
    table_counter = 0
    figure_counter = 0
    table_numbering = {}
    figure_numbering = {}
    consumed = set()
    def insert(key):
        nonlocal equation_counter, table_counter, figure_counter
        consumed.add(key)
        for kind, number, spec in attachments.get(key, []):
            if kind=="table":
                table_counter += 1
                table_numbering[spec.get("id",str(table_counter))] = table_counter
                story.extend(table_flowables(spec,table_counter,st,citations))
            elif kind=="figure":
                figure_counter += 1
                figure_numbering[spec.get("id",str(figure_counter))] = figure_counter
                base = Path(spec.get("_base_dir", manuscript_path.resolve().parent))
                story.extend(figure_flowables(spec,figure_counter,st,citations,base))
            elif kind=="baseline":
                figure_counter += 1
                figure_numbering["released_baseline_summary"] = figure_counter
                story.extend([CondPageBreak(290), KeepTogether([spec["drawing"], Spacer(1,6), paragraph(f"Figure {figure_counter}. {spec['caption']}",st["caption"],citations)]),Spacer(1,9)])
            else:
                equation_counter += 1
                markup = spec.get("display_markup") or spec.get("markup") or escape(clean_text(spec["display_text"]))
                eq_table = Table([[Paragraph(markup,st["equation"]), Paragraph(f"({equation_counter})",st["equation_number"])]],colWidths=[CONTENT_W-30,30])
                eq_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
                story.extend([CondPageBreak(70),eq_table])
                if spec.get("description"):
                    story.append(paragraph(spec["description"],st["note"],citations))
    for sec in manuscript["sections"]:
        heading = paragraph(sec["heading"],st["heading"],citations)
        heading.bookmark_name = "sec-"+sec["id"]
        story.append(heading)
        for index, text in enumerate(sec.get("paragraphs",[]),1):
            story.append(paragraph(text,st["body"],citations))
            insert((sec["id"],index))
        insert((sec["id"],None))
    unused = set(attachments)-consumed
    if unused:
        raise ValueError(f"Attachment positions outside section paragraph range: {unused}")
    story.append(CondPageBreak(180))
    heading = paragraph("References",st["heading"],citations)
    heading.bookmark_name="references"
    story.append(heading)
    for ref in refs:
        story.append(Paragraph(format_reference(ref),st["reference"]))
    check_glyph_coverage(story)
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.with_name("."+output.stem+".building-"+uuid.uuid4().hex+".pdf")
    manuscript_sha256=hashlib.sha256(manuscript_bytes).hexdigest()
    doc = PaperDoc(str(staging),manuscript=manuscript,draft=draft,manuscript_sha256=manuscript_sha256)
    doc.build(story)
    if staging.stat().st_size>5_000_000:
        staging.unlink()
        raise ValueError("Paper exceeds the 5 MB public-hosting limit; reduce oversized raster figures before publication")
    pdf = PdfReader(str(staging))
    page_texts = [page.extract_text() or "" for page in pdf.pages]
    if any(not text.strip() for text in page_texts):
        raise ValueError("PDF contains an empty text page")
    full_text = "\n".join(page_texts)
    for ref in refs:
        if f"[{ref['number']}]" not in full_text:
            raise ValueError(f"Missing reference {ref['number']} after PDF extraction")
    used_fonts=rendered_font_report(pdf)
    if not used_fonts or not all(used_fonts.values()):
        raise ValueError(f"Some rendered text uses an unembedded font: {used_fonts}")
    staging.replace(output)
    report = dict(output=str(output.resolve()),draft=draft,pages=len(pdf.pages),bytes=output.stat().st_size,
                  paper_size="US Letter",embedded_font_sources=fonts,
                  rendered_text_fonts_embedded=used_fonts,
                  page_character_counts=[len(text) for text in page_texts],
                  external_link_count=sum(1 for page in pdf.pages for item in page.get("/Annots",[]) if item.get_object().get("/A",{}).get("/URI")),
                  tables=len(tables),figures=len(figures)+(1 if baseline_path else 0),equations=len(equations),
                  table_numbering=table_numbering,figure_numbering=figure_numbering,
                  baseline_aggregation=baseline_stats,
                  pending_markers_detected=bool(re.search(r"\[DRAFT\b|\bTBD\b",full_text)),
                  input_manuscript_sha256=manuscript_sha256,
                  visual_review_required=True)
    output.with_suffix(".qa.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return report


def render_preview(pdf_path: Path, *, dpi=135) -> dict:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is required for visual QA")
    # Retain earlier renders while preventing stale pages from a longer PDF
    # from entering this render's contact sheet or visual-review manifest.
    folder = pdf_path.parent / (pdf_path.stem+"-pages-"+uuid.uuid4().hex[:10])
    folder.mkdir(parents=True,exist_ok=False)
    prefix = folder/"page"
    subprocess.run([pdftoppm,"-png","-r",str(dpi),str(pdf_path),str(prefix)],check=True,capture_output=True,text=True)
    pages=sorted(folder.glob("page-*.png"))
    if not pages:
        raise RuntimeError("Poppler did not produce any page images")
    if len(pages) != len(PdfReader(str(pdf_path)).pages):
        raise RuntimeError("Rendered page count differs from the current PDF")
    from PIL import Image as PILImage, ImageDraw
    thumb_w, pad, label_h = 260, 16, 24
    source=PILImage.open(pages[0]); thumb_h=round(source.height*thumb_w/source.width); source.close()
    cols=min(4,len(pages)); rows=math.ceil(len(pages)/cols)
    sheet=PILImage.new("RGB",(pad+(thumb_w+pad)*cols,pad+(thumb_h+label_h+pad)*rows),"#e6eaee")
    draw=ImageDraw.Draw(sheet)
    for i,path in enumerate(pages):
        with PILImage.open(path) as page:
            page.thumbnail((thumb_w,thumb_h),PILImage.Resampling.LANCZOS)
            x=pad+(i%cols)*(thumb_w+pad); y=pad+(i//cols)*(thumb_h+label_h+pad)
            sheet.paste(page,(x,y)); draw.text((x,y+thumb_h+5),f"Page {i+1}",fill="#263544")
    contact=pdf_path.parent/(pdf_path.stem+"-contact.png")
    sheet.save(contact)
    return dict(page_images=[str(path.resolve()) for path in pages],contact_sheet=str(contact.resolve()))


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manuscript",type=Path,default=ROOT/"manuscript_draft.json")
    parser.add_argument("--bibliography",type=Path,default=ROOT/"bibliography.json")
    parser.add_argument("--out",type=Path,default=ROOT/"preview"/"manuscript-draft.pdf")
    parser.add_argument("--tables",type=Path)
    parser.add_argument("--figures",type=Path)
    parser.add_argument("--baseline-comparison",type=Path)
    parser.add_argument("--font-dir",type=Path)
    parser.add_argument("--final",action="store_true")
    parser.add_argument("--render-pages",action="store_true")
    args=parser.parse_args(argv)
    report=build_pdf(args.manuscript,args.bibliography,args.out,draft=not args.final,
                     tables_path=args.tables,figures_path=args.figures,baseline_path=args.baseline_comparison,
                     font_dir=args.font_dir)
    if args.render_pages:
        report.update(render_preview(args.out))
        args.out.with_suffix(".qa.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("output","draft","pages","bytes","tables","figures","equations")},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
