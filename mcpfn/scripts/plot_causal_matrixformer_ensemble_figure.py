"""Causal normalized-score figure with the MatrixFormer ensemble as the only MatrixFormer method.

The per-draw min-max scale is set by the seven methods shown: the MatrixFormer ensemble (MatrixFormer+,
standard orientation) and the six baselines (SDiD, SC, DiD, SoftImpute, K-NN, HyperImpute). The
transposed and averaged ensemble runs are placed on that scale and the shown orientation follows the
paper's rule (lowest mean score over the six panels). The drawing follows the paper's imputation bar
plots (plot_imputation_results.plot_panels): every bar is labelled with its score to three decimals,
the lowest score in each panel is bold, MatrixFormer is blue and the baselines are gray. Methods are
sorted by mean score over the six panels and California smoking is the first column.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "69d7e55fd826e0325c3115d0/scripts"))
import export_causal_results as paper  # noqa: E402


OUTPUT = REPO / "runs/benchmark_evaluations/causal_matrixformer_plus_20261005"
FIGURE = OUTPUT / "causal_normalized_scores_matrixformer_ensemble_core"
METHOD = "matrixformer-plus"
VARIANTS = [METHOD, f"{METHOD}-T", f"{METHOD}-avg"]
BASELINES = list(paper.BASELINES)
COLUMNS = ("California smoking", "CPS", "Kansas")
BLUE, NEUTRAL, LABEL_BLUE, LABEL_GRAY = "#2A6FBB", "#b9bec8", "#1d4f87", "#333333"


def plot(scores, methods, ours, labels, output):
    """Bars with 95% whiskers; every bar is labelled and the lowest score of each panel is bold."""
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5})
    fig, axes = plt.subplots(len(paper.METRICS), len(COLUMNS), figsize=(5.5, 3.6), sharey=True)
    y = np.arange(len(methods))
    shown = scores[scores["method"].isin(methods)]
    maximum = max(1.0, (shown["score"] + shown["ci95"]).max())
    for row, metric in enumerate(paper.METRICS):
        for col, family in enumerate(COLUMNS):
            ax = axes[row, col]
            s = shown[(shown["family"] == family) & (shown["metric"] == metric)].set_index("method").loc[methods]
            best = s["score"].round(3).min()
            ax.barh(y, s["score"], height=0.7, linewidth=0, color=[BLUE if m == ours else NEUTRAL for m in methods])
            ax.errorbar(s["score"], y, xerr=s["ci95"], fmt="none", ecolor="#222222", elinewidth=0.7, capsize=1.5)
            ax.axvline(1, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
            for position, (method, r) in enumerate(s.iterrows()):
                ax.text(r["score"] + r["ci95"] + maximum * 0.04, position, f"{r['score']:.3f}", va="center",
                        fontsize=6.5, fontweight="bold" if round(r["score"], 3) == best else "normal",
                        color=LABEL_BLUE if method == ours else LABEL_GRAY,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
            ax.set_xlim(0, maximum * 1.3)
            ax.set_ylim(len(methods) - 0.45, -0.6)
            ax.set_xticks([0, 0.5, 1.0])
            ax.tick_params(length=2, pad=1.5, labelsize=7.5)
            ax.grid(axis="x", color="#e3e3e3", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(family, fontsize=7.5)
            if col == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([labels[m] for m in methods])
                ax.set_ylabel(paper.AXIS_LABELS[metric], fontsize=8)
                tick = ax.get_yticklabels()[methods.index(ours)]
                tick.set_color(BLUE)
                tick.set_fontweight("bold")
    fig.supxlabel("Normalized score (0 = best, 1 = worst of the seven main methods in each draw)",
                  fontsize=7.5, y=0.01)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    for ext, options in (("pdf", {"metadata": {"CreationDate": None}}), ("png", {"dpi": 240}), ("svg", {})):
        fig.savefig(output.with_suffix(f".{ext}"), **options)
    plt.close(fig)


def main():
    plus = pd.concat([pd.read_csv(OUTPUT / "per_draw_metrics.csv"),
                      pd.read_csv(OUTPUT / "per_draw_metrics_transpose.csv")], ignore_index=True)
    labels = {m: paper.LABELS[m] for m in BASELINES}
    paper.CORE = {METHOD: "MatrixFormer", **labels}
    paper.CANDIDATES = VARIANTS[1:]
    rows = []
    for family, designs in paper.FAMILIES.items():
        frames = []
        for _, path in designs:
            saved = pd.read_csv(paper.BENCHMARK / path / "results/per_draw_metrics.csv")
            frames.append(pd.concat([saved[saved["method"].isin(BASELINES)], plus[plus["path"] == path]],
                                    ignore_index=True))
        for metric in paper.METRICS:
            mean, se, n = paper.normalized_scores(frames, metric)
            for m in [*VARIANTS, *BASELINES]:
                rows.append({"family": family, "metric": metric, "method": m, "score": mean[m], "se": se[m],
                             "ci95": 1.96 * se[m], "n_scenarios": len(frames), "n_draws": n})
    scores = pd.DataFrame(rows)
    ours = scores[scores["method"].isin(VARIANTS)].groupby("method")["score"].mean().idxmin()
    scores["shown"] = scores["method"].isin([ours, *BASELINES])
    scores.to_csv(FIGURE.with_suffix(".csv"), index=False, float_format="%.6f")

    methods = scores[scores["shown"]].groupby("method")["score"].mean().sort_values().index.tolist()
    plot(scores, methods, ours, {**labels, ours: "MatrixFormer"}, FIGURE)
    print(f"shown orientation: {ours}; order: {methods}")
    print(scores[scores["shown"]].pivot_table(index="method", columns=["metric", "family"], values="score",
                                              sort=False).loc[methods].round(3).to_string())


if __name__ == "__main__":
    main()
