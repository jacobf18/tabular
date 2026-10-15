#!/usr/bin/env python3
"""Two-page shareable PDF report: MatrixFormer on the CPS / Kansas / Prop99
causal benchmarks.

Page 1: draw-normalized treated-block RMSE and treatment-effect error, one bar
chart per dataset (same summaries as plot_dwip_spectral_tm128k_bars.py).
Page 2: per-scenario treated-block RMSE table and method notes.

Writes figures/dwip_spectral_tm128k/MatrixFormer_causal_benchmarks.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))
import plot_dwip_spectral_tm128k_bars as bars  # noqa: E402

OUT = bars.FIGURES_DIR / "MatrixFormer_causal_benchmarks.pdf"
NEW = bars.NEW
LABELS = {
    "did": "DiD", "sdid": "SDiD", "sc": "SC", "hyperimpute": "HyperImpute", "knn": "K-NN",
    "softimpute": "SoftImpute", "dwip-v2-2": "MatFormer", "dwip-v2-2-T": "MatFormer (transpose)",
    "dwip-v2-2-avg": "MatFormer (avg)", **bars.MATRIXFORMER_LABELS,
}
MATFORMER = {"dwip-v2-2", "dwip-v2-2-T", "dwip-v2-2-avg"}

FG = colors.HexColor("#16191f")
FG_2 = colors.HexColor("#4b5261")
LINE = colors.HexColor("#dde1e8")
ACCENT = colors.HexColor("#2a6fbb")
BAR_MATFORMER = colors.HexColor("#9dbde6")
BAR_OTHER = colors.HexColor("#c9ced7")

PAGE_W, PAGE_H = landscape(letter)
MARGIN = 36


def nice_ticks(hi: float) -> list[float]:
    for step in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5):
        if hi / step <= 5:
            break
    return [round(i * step, 6) for i in range(int(hi / step + 1e-9) + 1)]


def label_style(method: str) -> tuple[str, colors.Color, colors.Color]:
    """(font, label color, bar color)."""
    if method.startswith(NEW):
        return "Helvetica-Bold", ACCENT, ACCENT
    if method in MATFORMER:
        return "Helvetica-Bold", FG, BAR_MATFORMER
    return "Helvetica", FG_2, BAR_OTHER


def draw_chart(c: canvas.Canvas, x0: float, top: float, width: float, rows, title: str,
               xlabel: str) -> float:
    """Horizontal bars, best first. rows: (method, value, ci halfwidth). Returns bottom y."""
    rows = sorted(rows, key=lambda r: r[1])
    fs, row_h, bar_h = 7.5, 13, 8
    label_w = max(stringWidth(LABELS[m], label_style(m)[0], fs) for m, _, _ in rows) + 6
    px0, pw = x0 + label_w, width - label_w - 8
    hi = max(v + ci for _, v, ci in rows) * 1.05
    ticks = nice_ticks(hi)
    hi = max(hi, ticks[-1])
    sx = lambda v: px0 + pw * min(max(v, 0.0), hi) / hi  # noqa: E731

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(FG)
    c.drawString(x0, top - 9, title)
    y_top = top - 16
    y_bot = y_top - row_h * len(rows)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.5)
    c.setFont("Helvetica", 6.5)
    c.setFillColor(FG_2)
    for t in ticks:
        c.line(sx(t), y_top, sx(t), y_bot)
        c.drawCentredString(sx(t), y_bot - 9, f"{t:g}")
    c.setFont("Helvetica", 7)
    c.drawCentredString(px0 + pw / 2, y_bot - 20, xlabel)
    for i, (m, v, ci) in enumerate(rows):
        yc = y_top - row_h * i - row_h / 2
        font, lab_color, bar_color = label_style(m)
        c.setFont(font, fs)
        c.setFillColor(lab_color)
        c.drawRightString(px0 - 5, yc - 2.6, LABELS[m])
        c.setFillColor(bar_color)
        c.roundRect(px0, yc - bar_h / 2, max(sx(v) - px0, 1.0), bar_h, 1.2, stroke=0, fill=1)
        c.setStrokeColor(FG)
        c.setLineWidth(0.8)
        c.line(sx(v - ci), yc, sx(v + ci), yc)
    c.setStrokeColor(FG_2)
    c.setLineWidth(0.6)
    c.line(px0, y_top, px0, y_bot)
    return y_bot - 24


def chart_rows():
    """{(dataset, metric suffix): (rows, xlabel, title)} from the shared summaries."""
    out = {}
    for name, paths in bars.DATASETS.items():
        for col, suffix, _ in bars.METRICS:
            s, n_draws = bars.normalized_summary(paths, col)
            rows = list(zip(s["method"], s["norm"], 1.96 * s["norm_se"]))
            scen = f"{len(paths)} scenario{'s' if len(paths) > 1 else ''}"
            out[(name, suffix)] = (rows, "normalized score (0 = best, 1 = worst in each draw)",
                                   f"{name}: {scen}, {n_draws:,} draws")
    return out


def scenario_table() -> list[list[str]]:
    cols = [f"{NEW}", f"{NEW}-avg", "dwip-v2-2", "dwip-v2-2-avg", "sdid"]
    header = ["Scenario", "MatrixFormer", "MatrixFormer (avg)", "MatFormer", "MatFormer (avg)",
              "SDiD", "Rank of MatrixFormer (avg)"]
    body = []
    for name, paths in bars.DATASETS.items():
        for p in paths:
            df = pd.read_csv(BENCH_DIR / p / "results/per_draw_metrics.csv")
            means = df[df["method"].isin(bars.METHODS)].groupby("method")["treated_block_rmse"].mean()
            rank = int((means < means[f"{NEW}-avg"]).sum()) + 1
            scen = p.split("/")[-1].replace("kansas_factor_r3_200", "factor r3").replace("_", " ")
            vals = [means[m] for m in cols]
            digits = 3 if max(vals) >= 10 else 4
            body.append([f"{name} · {scen}", *(f"{v:.{digits}f}" for v in vals),
                         f"{rank} of {len(bars.METHODS)}"])
    return [header, *body]


def page_one(c: canvas.Canvas, charts) -> None:
    y = PAGE_H - MARGIN
    c.setFillColor(FG)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(MARGIN, y - 17, "MatrixFormer on the CPS, Kansas and Prop99 causal benchmarks")
    caption = Paragraph(
        "MatrixFormer: new DWIP model (DWIP v2-2 fine-tuned with the teacher-mean recipe, checkpoint step 98k). "
        "3,200 benchmark draws. September 28, 2026.<br/>"
        "MatFormer is dwip-v2-2, the model MatrixFormer was fine-tuned from. (transpose) runs a model on the "
        "transposed matrix; (avg) averages the normal and transposed runs.<br/>"
        "Scores: in each draw, the 12 methods' errors are min-max normalized (0 = best, 1 = worst in that draw), "
        "then averaged over the draws of each scenario and over scenarios with equal weight. Best method at the top.<br/>"
        "Whiskers are 95% CIs from draw-to-draw variation (CPS 1,800 draws, Prop99 1,200, Kansas 200), so they "
        "mean the same thing for all three datasets.",
        ParagraphStyle("caption", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=FG_2))
    _, ch = caption.wrapOn(c, PAGE_W - 2 * MARGIN, PAGE_H)
    caption.drawOn(c, MARGIN, y - 25 - ch)
    ly = y - 25 - ch - 14
    lx = MARGIN
    for text, color in (("MatrixFormer (new)", ACCENT), ("MatFormer", BAR_MATFORMER), ("Baselines", BAR_OTHER)):
        c.setFillColor(color)
        c.roundRect(lx, ly, 9, 9, 1.5, stroke=0, fill=1)
        c.setFillColor(FG)
        c.setFont("Helvetica", 8)
        c.drawString(lx + 13, ly + 2, text)
        lx += 13 + stringWidth(text, "Helvetica", 8) + 18

    top = ly - 10
    gap = 18
    width = (PAGE_W - 2 * MARGIN - 2 * gap) / 3
    for suffix, heading in (("tb_rmse", "Treated-block RMSE (imputation error on the treated block)"),
                            ("tau_error", "Absolute error of the treatment-effect estimate")):
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(FG)
        c.drawString(MARGIN, top - 11, heading)
        bottoms = []
        for k, name in enumerate(bars.DATASETS):
            rows, xlabel, title = charts[(name, suffix)]
            bottoms.append(draw_chart(c, MARGIN + k * (width + gap), top - 18, width, rows, title, xlabel))
        top = min(bottoms) - 6


def page_two(c: canvas.Canvas) -> None:
    y = PAGE_H - MARGIN
    c.setFillColor(FG)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN, y - 13, "Treated-block RMSE by scenario")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(FG_2)
    c.drawString(MARGIN, y - 27, "Mean over draws. Bold marks the lowest of the five model columns. "
                 "Rank is among the 12 methods in the charts (1 = best).")
    data = scenario_table()
    table = Table(data, colWidths=[150, 82, 100, 82, 90, 70, 130], rowHeights=16)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), FG_2),
        ("TEXTCOLOR", (0, 1), (-1, -1), FG),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for r, row in enumerate(data[1:], start=1):
        vals = [float(v) for v in row[1:6]]
        best = 1 + int(np.argmin(vals))
        style += [("FONT", (best, r), (best, r), "Helvetica-Bold", 8.5),
                  ("TEXTCOLOR", (best, r), (best, r), ACCENT)]
    table.setStyle(TableStyle(style))
    _, th = table.wrapOn(c, PAGE_W - 2 * MARGIN, PAGE_H)
    table.drawOn(c, MARGIN, y - 40 - th)

    notes = ParagraphStyle("notes", fontName="Helvetica", fontSize=8.5, leading=12, textColor=FG_2)
    head = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=FG)
    ny = y - 40 - th - 22
    for text, style_ in (
        ("About the benchmarks", head),
        ("CPS: 9 scenarios (hours, log wage and unemployment rate, each with abort-ban, minimum-wage and "
         "open-carry treatment patterns), 200 draws each. Kansas: 1 semi-synthetic factor scenario, 200 draws. "
         "Prop99: ca_simul and ca_stagg (500 draws each) and stocks (200 draws). Each draw hides the treated "
         "block; methods impute it and are scored on treated-block RMSE and on the error of the implied "
         "treatment-effect estimate.", notes),
        ("MatrixFormer checkpoint: tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/"
         "best_validation_checkpoint.pth (step 98,000, selected on synthetic validation only). Inputs are "
         "column-standardized; observed cells are kept; predictions use the mean of the predictive "
         "distribution.", notes),
    ):
        p = Paragraph(text, style_)
        _, ph = p.wrapOn(c, PAGE_W - 2 * MARGIN, PAGE_H)
        p.drawOn(c, MARGIN, ny - ph)
        ny -= ph + 6


def main() -> None:
    charts = chart_rows()
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("MatrixFormer on the CPS, Kansas and Prop99 causal benchmarks")
    page_one(c, charts)
    c.showPage()
    page_two(c)
    c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
