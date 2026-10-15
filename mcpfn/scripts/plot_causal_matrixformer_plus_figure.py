"""Causal normalized-score figure for MatrixFormer+ next to the six non-MatrixFormer core methods.

Reads normalized_scores_orientations.csv (written by evaluate_causal_matrixformer_plus_transpose.py) and
draws it with the paper's plot function, export_causal_results.plot. Only presentation changes: the
chosen MatrixFormer+ orientation is labelled "MatrixFormer", methods are sorted by mean score over the
six panels, California smoking is the first column, and column titles omit design/draw counts. Scores
stay on the paper's seven-core-method scale, so the baseline bars match the paper.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.axes
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "69d7e55fd826e0325c3115d0/scripts"))
import export_causal_results as paper  # noqa: E402


OUTPUT = REPO / "runs/benchmark_evaluations/causal_matrixformer_plus_20261005"
FIGURE = OUTPUT / "causal_normalized_scores_matrixformer_plus_only_sorted_ca_first_notitle"
COLUMNS = ("California smoking", "CPS", "Kansas")
VARIANTS = ["matrixformer-plus", "matrixformer-plus-T", "matrixformer-plus-avg"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURE,
                        help="Output path without extension; .pdf and .png are written.")
    args = parser.parse_args()

    scores = pd.read_csv(OUTPUT / "normalized_scores_orientations.csv")
    methods = scores[scores["shown"]].groupby("method")["score"].mean().sort_values().index.tolist()
    chosen = [m for m in methods if m in VARIANTS]
    if len(chosen) != 1 or methods[0] != chosen[0]:
        raise SystemExit(f"Expected one MatrixFormer+ orientation ranked first, got order {methods}")

    paper.LABELS.update(dict(zip(scores["method"], scores["label"])))
    paper.LABELS[chosen[0]] = "MatrixFormer"
    paper.CANDIDATES.extend(VARIANTS)
    paper.FAMILIES = {family: paper.FAMILIES[family] for family in COLUMNS}
    set_title = matplotlib.axes.Axes.set_title
    matplotlib.axes.Axes.set_title = lambda self, label, *a, **kw: set_title(self, label.split("\n")[0], *a, **kw)

    for ext in ("pdf", "png"):
        paper.plot(scores, args.output.with_suffix(f".{ext}"), methods)
        print(args.output.with_suffix(f".{ext}"))


if __name__ == "__main__":
    main()
