"""Shared plot styling and helpers for reading benchmark results."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from ..methods import CHECKPOINT_METHODS, display_name  # noqa: E402
from ..patterns import PATTERNS  # noqa: E402

HIGHLIGHT_COLOR = "#2A6FBB"  # default TabImpute
SPECIALIST_COLOR = "#7A2CA0"  # pattern-specific TabImpute-P
TRACKED_COLOR = "#FF8F2B"  # not-MIWAE
NEUTRAL_COLOR = "#D4D4D4"  # other baselines
NEUTRAL_COLOR_DARK = "#262626"  # tick label text

FIGURE_SIZE = (5.5, 6.5)
BARPLOT_STYLE = {"capsize": 0.2, "err_kws": {"color": "#999999"}, "edgecolor": "#6E6E6E"}

PATTERN_ORDER = list(PATTERNS)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 22,
    "axes.titlesize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 28,
})


def method_color(key: str) -> str:
    if key in CHECKPOINT_METHODS:
        return SPECIALIST_COLOR
    if key.startswith("tabimpute"):
        return HIGHLIGHT_COLOR
    if key.startswith("notmiwae"):
        return TRACKED_COLOR
    return NEUTRAL_COLOR


def ci95_halfwidth(values: pd.Series) -> float:
    """Half-width of a t-based 95% confidence interval for the mean."""
    values = values.dropna()
    n = len(values)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * values.sem())


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def bar_panel(ax, long: pd.DataFrame, metric: str, title: str, specialist: str | None = None,
              ascending: bool = True, tick_fontsize: int = 12, title_fontsize: int = 22) -> None:
    """Bar chart of `metric` per method (mean with 95% CI), sorted best first.

    `long` has columns "method" and `metric`. The `specialist` method (if any) is
    labelled "TabImpute-P".
    """
    import seaborn as sns

    if long.empty:
        ax.set_title(title, fontsize=title_fontsize, family="monospace")
        ax.text(0.5, 0.5, "no results", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        style_axis(ax)
        return
    order = long.groupby("method")[metric].mean().sort_values(ascending=ascending).index.tolist()
    sns.barplot(data=long, x="method", y=metric, hue="method", order=order,
                palette={m: method_color(m) for m in order}, ax=ax, legend=False, **BARPLOT_STYLE)
    labels = ["TabImpute-P" if m == specialist else display_name(m) for m in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)
    for label, m in zip(ax.get_xticklabels(), order):
        color = method_color(m)
        if color in (HIGHLIGHT_COLOR, SPECIALIST_COLOR):
            label.set_color(color)
            label.set_weight("bold")
            label.set_fontsize(tick_fontsize * 1.1)
        else:
            label.set_color(NEUTRAL_COLOR_DARK)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=title_fontsize, family="monospace")
    ax.set_ylim(bottom=0)
    style_axis(ax)


def legend_handles(include_specialist: bool, include_tracked: bool):
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=HIGHLIGHT_COLOR, label="TabImpute (Default; P = MCAR)")]
    if include_specialist:
        handles.append(Patch(facecolor=SPECIALIST_COLOR, label="TabImpute-P (Pattern-Specific)"))
    if include_tracked:
        handles.append(Patch(facecolor=TRACKED_COLOR, label="not-MIWAE (Tracked Baseline)"))
    handles.append(Patch(facecolor=NEUTRAL_COLOR, label="Other Baselines"))
    return handles
