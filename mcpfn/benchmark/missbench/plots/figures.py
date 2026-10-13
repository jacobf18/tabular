"""Paper figures (and a few extra diagnostic figures) built from the results CSVs."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..methods import METHODS, PAPER_METHODS, display_name
from ..patterns import PATTERNS
from .context import Context, MissingInput, require
from .style import (
    FIGURE_SIZE,
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    NEUTRAL_COLOR_DARK,
    PATTERN_ORDER,
    bar_panel,
    legend_handles,
    method_color,
    style_axis,
)


def _long(df: pd.DataFrame, methods: list[str], metric: str) -> pd.DataFrame:
    return df[df["method"].isin(methods)][["method", metric]]


def _panels(ctx: Context, panel_patterns: list[str], extra: list[str] = (), metric: str = "nrmse"):
    """(title, long data, specialist key) for "Overall" plus each pattern in `panel_patterns`."""
    df = ctx.scores("openml", metric)
    base = ctx.baselines(df) + [m for m in extra if m in set(df["method"])]
    specialists = ctx.specialists(df)
    panels = [("Overall", _long(df, base, metric), None)]
    for key in panel_patterns:
        spec = specialists.get(key)
        methods = base + ([spec] if spec else [])
        title = f"P = {PATTERNS[key].display}"
        panels.append((title, _long(df[df["pattern"] == key], methods, metric), spec))
    return panels, bool(specialists), any(m.startswith("notmiwae") for m in base)


def _bar_figure(ctx: Context, panels, nrows: int, ncols: int, filename: str, ylabel: str,
                has_specialists: bool, has_tracked: bool, ascending: bool = True, legend_y: float | None = 1.0,
                top: float | None = None):
    """Grid of bar panels. The legend sits at `legend_y` (figure coordinates; None for no legend),
    above the subplots, which end at `top`."""
    w, h = FIGURE_SIZE
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * w, nrows * h), squeeze=False)
    axes = axes.flatten()
    for i, (title, long, spec) in enumerate(panels):
        bar_panel(axes[i], long, long.columns[-1], title, specialist=spec, ascending=ascending)
        if i % ncols == 0:
            axes[i].set_ylabel(ylabel, fontsize=20)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout(pad=0.5)
    if top is not None:
        fig.subplots_adjust(top=top)
    if legend_y is not None:
        handles = legend_handles(has_specialists, has_tracked)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=len(handles),
                   fontsize=16, frameon=False)
    path = ctx.out(filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_rmse_highlights(ctx: Context):
    """Fig. 1: RMSE overall and for MCAR, Censor-MNAR and Self-MNAR."""
    panels, spec, tracked = _panels(ctx, ["mcar", "censor_mnar", "self_mnar"])
    return _bar_figure(ctx, panels, 1, 4, "fig1_rmse_overall_mcar_censor_self.pdf", "RMSE", spec, tracked,
                       legend_y=1.1)


def fig_rmse_all_patterns(ctx: Context):
    """Fig. 5: RMSE overall and for every missingness pattern."""
    panels, spec, tracked = _panels(ctx, PATTERN_ORDER)
    return _bar_figure(ctx, panels, 3, 4, "fig5_rmse_all_patterns.pdf", "RMSE", spec, tracked, top=0.93)


def fig_self_mnar_notmiwae(ctx: Context):
    """Fig. 6: Self-MNAR RMSE including not-MIWAE with a Self-MNAR propensity model."""
    df = ctx.scores("openml")
    require(df, ["notmiwae_self_mnar"], "Fig. 6")
    panels, spec, _ = _panels(ctx, ["self_mnar"], extra=["notmiwae", "notmiwae_self_mnar"])
    return _bar_figure(ctx, panels[1:], 1, 1, "fig6_self_mnar_with_notmiwae.pdf", "RMSE", spec, True,
                       legend_y=None)


def metric_bars(metric: str, ylabel: str, ascending: bool):
    """Overall + per-pattern bar charts of any metric in the results (extra figures)."""
    def make(ctx: Context):
        panels, spec, tracked = _panels(ctx, PATTERN_ORDER, metric=metric)
        return _bar_figure(ctx, panels, 3, 4, f"extra_{metric}_all_patterns.pdf", ylabel, spec, tracked,
                           ascending=ascending, top=0.93)

    make.__doc__ = f"{ylabel} overall and for every missingness pattern."
    return make


def fig_scatter_vs_missforest(ctx: Context):
    """Fig. 4a: per dataset-pattern RMSE of TabImpute vs. MissForest (or --compare A B)."""
    a, b = ctx.compare
    df = ctx.scores("openml")
    require(df, [a, b], "The scatter plot")
    table = df.pivot_table(index=["dataset", "pattern"], columns="method", values="nrmse")[[a, b]].dropna()
    x, y = table[a].to_numpy(), table[b].to_numpy()

    wins = int((x < y).sum())
    total = len(x)
    try:
        _, p_value = stats.wilcoxon(x, y, alternative="less", zero_method="pratt")
        p_str = "p<1e-4" if p_value < 1e-4 else f"p={p_value:.2e}"
    except ValueError:
        p_str = "p=n/a"

    hi = float(max(x.max(), y.max())) * 1.05
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.fill_between([0, hi], [0, hi], [hi, hi], color=HIGHLIGHT_COLOR, alpha=0.10, linewidth=0, zorder=0)
    ax.scatter(x, y, s=40, color=HIGHLIGHT_COLOR, marker="x", linewidth=1.2, alpha=0.8, zorder=3)
    ax.plot([0, hi], [0, hi], color=NEUTRAL_COLOR, linestyle="--", linewidth=1.2, zorder=1)

    # Label the points furthest from the diagonal with their pattern
    gap = np.abs(y - x)
    for i in np.argsort(gap)[-6:][::-1]:
        if gap[i] > 0:
            ax.annotate(PATTERNS[table.index[i][1]].display, xy=(x[i], y[i]), xytext=(6, 4),
                        textcoords="offset points", fontsize=10, color=NEUTRAL_COLOR_DARK, family="monospace")

    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{display_name(a)} (RMSE)", fontsize=18)
    ax.set_ylabel(f"{display_name(b)} (RMSE)", fontsize=18)
    ax.set_title(f"{display_name(a)} vs. {display_name(b)}\n{display_name(a)} better in {wins}/{total} = "
                 f"{100 * wins / total:.1f}%; Wilcoxon {p_str}", fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    path = ctx.out(f"fig4a_scatter_{a}_vs_{b}.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {display_name(a)} better than {display_name(b)} on {wins}/{total} dataset-pattern pairs ({p_str})")
    return [path]


def fig_runtime_per_entry(ctx: Context):
    """Fig. 4b: runtime per table entry for each method (MCAR, p=0.4)."""
    import seaborn as sns

    df = ctx.load("openml")
    df = df[(df["pattern"] == "mcar") & df["seconds"].notna()].copy()
    if df.empty:
        raise MissingInput("no runtimes recorded for MCAR")
    wanted = (ctx.methods or PAPER_METHODS) + ["tabimpute_cpu"]
    df = df[df["method"].isin(wanted)]
    df["per_entry"] = df["seconds"] / (df["n_rows"] * df["n_cols"])

    def label(row):
        on_gpu = row["device"] == "cuda" if isinstance(row["device"], str) else METHODS[row["method"]].uses_gpu
        name = "TabImpute" if row["method"] == "tabimpute_cpu" else display_name(row["method"])
        return f"{name} ({'GPU' if on_gpu else 'CPU'})" if row["method"].startswith("tabimpute") or on_gpu else name

    df["label"] = df.apply(label, axis=1)
    order = df.groupby("label")["per_entry"].mean().sort_values().index.tolist()
    colors = {lab: method_color(df.loc[df["label"] == lab, "method"].iloc[0]) for lab in order}

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.barplot(data=df, x="label", y="per_entry", hue="label", order=order, palette=colors, ax=ax,
                legend=False, capsize=0.2, err_kws={"color": "#999999"}, edgecolor="#6E6E6E")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=14)
    for tick, lab in zip(ax.get_xticklabels(), order):
        tick.set_color(colors[lab] if colors[lab] == HIGHLIGHT_COLOR else "#333333")
    ax.set_xlabel("")
    ax.set_ylabel("Time per entry", fontsize=16)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "" if v <= 0 else (f"{v * 1000:g} ms" if v < 1 else f"{v:g} s")))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    path = ctx.out("fig4b_runtime_per_entry.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    means = df.groupby("method")["seconds"].mean()
    if "tabimpute" in means and "tabpfn" in means:
        print(f"  TabImpute is {means['tabpfn'] / means['tabimpute']:.0f}x faster than TabPFN on average")
    return [path]


def fig_mcar_sweep(ctx: Context):
    """Fig. 7: RMSE vs. fraction of missing values under MCAR."""
    import seaborn as sns

    df = ctx.scores("openml_mcar_sweep")
    methods = ctx.methods or ["tabimpute", "hyperimpute", "missforest", "gain"]
    methods = [m for m in methods if m in set(df["method"])]
    if not methods:
        raise MissingInput("no results for the methods in Fig. 7")
    df = df[df["method"].isin(methods)].sort_values("p")
    palette = {m: HIGHLIGHT_COLOR if m == "tabimpute" else "#A8A8A8" for m in methods}

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="p", y="nrmse", hue="method", hue_order=methods, palette=palette, ax=ax)
    ax.legend_.remove()
    ax.set_xlabel("Fraction of Missing Values", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.tick_params(labelsize=10)
    lines = ax.get_lines()[:len(methods)]
    # Label each line at its right end, nudging labels apart so they do not overlap
    ends = sorted(((m, line.get_xdata()[-1], line.get_ydata()[-1], line) for m, line in zip(methods, lines)
                   if len(line.get_xdata())), key=lambda t: t[2])
    last_y = -np.inf
    for m, x_end, y_end, line in ends:
        y_end = max(y_end, last_y + 0.03)
        last_y = y_end
        if m == "tabimpute":
            line.set_linewidth(2.5)
        ax.text(x_end, y_end, f"  {display_name(m)}", color=line.get_color(), va="center", fontsize=10,
                fontweight="bold" if m == "tabimpute" else "normal")
    path = ctx.out("fig7_mcar_rmse_vs_missingness.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path]


# Settings used to draw the example masks in Fig. 3
_MASK_CONFIG = {
    "num_rows_low": 50, "num_rows_high": 50, "num_cols_low": 50, "num_cols_high": 50,
    "p_missing": 0.3, "threshold_quantile": 0.4, "n_core_items": 5, "n_genres": 3, "n_policies": 4,
    "latent_rank_low": 1, "latent_rank_high": 15, "latent_spike_p": 0.3, "latent_slab_sigma": 2.0,
    "apply_feature_warping_prob": 0.0, "apply_quantization_prob": 0.0, "spline_knot_k": [3, 5, 7],
    "gp_length_scale_low": 0.3, "gp_length_scale_high": 2.0, "fourier_dim_low": 100, "fourier_dim_high": 501,
    "rpca_beta_a": 2, "rpca_beta_b": 30, "soft_polarization_alpha": 1.3, "soft_polarization_epsilon": 0.45,
    "cascade_n_genres": 5, "cascade_delta": 1.5, "cluster_level_n_row_clusters": 8,
    "cluster_level_n_col_clusters": 8, "cluster_level_tau_r_std": 1.0, "spatial_block_n_blocks": 5,
    "spatial_block_p_geom": 0.2, "censor_quantile": 0.25, "two_phase_cheap_fraction": 0.4,
    "two_phase_beta": 2.5, "skip_logic_p_noise": 0.9, "cold_start_fraction": 0.3, "cold_start_gamma": 0.15,
    "mar_config": {"num_layers_upper": 3, "hidden_lower": 1, "hidden_upper": 100, "activation": "relu",
                   "N": 100, "T": 50, "row_neighbor_upper": 5, "col_neighbor_upper": 5, "seed": 42,
                   "neighbor_type": "random"},
    "mar_block_config": {"N": 100, "T": 50, "row_blocks": 10, "col_blocks": 10, "convolution_type": "mean"},
    "mar_sequential_bandit_config": {"algorithm": "epsilon_greedy", "pooling": False, "epsilon": 0.4,
                                     "epsilon_decay": 0.99, "random_seed": 42},
}
def fig_mask_examples(ctx: Context):
    """Fig. 3 (right): example missingness masks for each pattern."""
    import torch
    from matplotlib.colors import ListedColormap
    from tabimpute.model.encoders import normalize_data
    from tabimpute.prior import training_set_generation as tsg

    cmap = ListedColormap(["#ECECEC", "#55AFA9"])  # observed, missing
    n_grid_cols = 4
    n_grid_rows = int(np.ceil(len(PATTERN_ORDER) / n_grid_cols))
    fig = plt.figure(figsize=(3.0 * n_grid_cols, 3.4 * n_grid_rows))
    gs = fig.add_gridspec(n_grid_rows, n_grid_cols * 2)
    n_last = len(PATTERN_ORDER) % n_grid_cols
    for i, key in enumerate(PATTERN_ORDER):
        row, col = divmod(i, n_grid_cols)
        # Center the last, partially filled row
        offset = (n_grid_cols - n_last) if (n_last and row == n_grid_rows - 1) else 0
        ax = fig.add_subplot(gs[row, offset + 2 * col: offset + 2 * col + 2])

        # A 50 x 50 latent factor matrix from the training prior, then the pattern's mask
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        X = tsg.LatentFactorPrior(_MASK_CONFIG).generate_complete_matrix()
        X = normalize_data(torch.tensor(X.values, dtype=torch.float32))
        mask = torch.isnan(getattr(tsg, PATTERNS[key].cls_name)(_MASK_CONFIG)._induce_missingness(X))
        ax.imshow(mask.numpy(), cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(PATTERNS[key].display, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(0.6)
    fig.tight_layout()
    path = ctx.out("fig3_missingness_mask_examples.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_runtime_scaling(ctx: Context):
    """Runtime vs. number of rows on random matrices (from `run_benchmark.py runtime`)."""
    df = ctx.load("runtime")
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, group in df.groupby("method"):
        stats_ = group.groupby("num_rows")["seconds"].agg(["mean", "std"]).reset_index()
        ax.errorbar(stats_["num_rows"], stats_["mean"], yerr=stats_["std"], marker="o", capsize=4,
                    label=display_name(method), color=method_color(method) if method == "tabimpute" else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Number of rows ({int(df['num_cols'].iloc[0])} columns)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=11)
    style_axis(ax)
    fig.tight_layout()
    path = ctx.out("extra_runtime_scaling.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_uci_rmse(ctx: Context):
    """RMSE on the UCI datasets, one panel per missingness level."""
    df = ctx.scores("uci")
    methods = ctx.baselines(df)
    levels = sorted(df["p"].unique())
    # Average over repeats first so each dataset counts once per level
    df = df.groupby(["dataset", "p", "method"], as_index=False)["nrmse"].mean()
    panels = [(f"MCAR, p = {p:g}", _long(df[df["p"] == p], methods, "nrmse"), None) for p in levels]
    ncols = min(len(panels), 5)
    nrows = int(np.ceil(len(panels) / ncols))
    return _bar_figure(ctx, panels, nrows, ncols, "extra_uci_rmse.pdf", "RMSE", False,
                       any(m.startswith("notmiwae") for m in methods), legend_y=1.1)


def fig_win_rate(ctx: Context):
    """Share of datasets on which each method has the lowest RMSE, per pattern."""
    import seaborn as sns

    df = ctx.scores("openml", "rmse")
    table = df.pivot_table(index=["dataset", "pattern"], columns="method", values="rmse")
    table = table[ctx.baselines(df)].dropna()
    winners = table.idxmin(axis=1).rename("winner").reset_index()
    rates = (winners.groupby("pattern")["winner"].value_counts(normalize=True) * 100).rename("win_rate").reset_index()
    overall = (winners["winner"].value_counts(normalize=True) * 100).rename("win_rate").reset_index()
    overall["pattern"] = "overall"

    paths = []
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    for ax, key in zip(axes.flatten(), ["overall"] + PATTERN_ORDER):
        data = (overall if key == "overall" else rates[rates["pattern"] == key]).sort_values("win_rate")
        data = data.assign(label=data["winner"].map(display_name))
        sns.barplot(data=data, x="win_rate", y="label", hue="label", legend=False, ax=ax,
                    palette={display_name(m): method_color(m) for m in data["winner"]}, edgecolor="#6E6E6E")
        ax.set_title("Overall" if key == "overall" else PATTERNS[key].display, fontsize=18, family="monospace")
        ax.set_xlabel("Win rate (%)", fontsize=12)
        ax.set_ylabel("")
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    paths.append(ctx.out("extra_win_rate.pdf"))
    fig.savefig(paths[-1], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def fig_critical_difference(ctx: Context):
    """Critical difference diagrams (Friedman + Nemenyi) over all dataset-pattern pairs and per pattern."""
    try:
        import scikit_posthocs as sp
    except ImportError as e:
        raise MissingInput("needs scikit-posthocs (pip install scikit-posthocs)") from e

    df = ctx.scores("openml")
    table = df.pivot_table(index=["dataset", "pattern"], columns="method", values="nrmse")
    table = table[ctx.baselines(df)].dropna()
    table.columns = [display_name(m) for m in table.columns]

    paths = []
    for key in ["overall"] + PATTERN_ORDER:
        data = table if key == "overall" else table[table.index.get_level_values("pattern") == key]
        if len(data) < 2:
            continue
        ranks = data.rank(axis=1, method="average").mean().sort_values()
        p_values = sp.posthoc_nemenyi_friedman(data.to_numpy())
        p_values.index = p_values.columns = data.columns
        fig = plt.figure(figsize=(12, 6))
        sp.critical_difference_diagram(ranks, p_values)
        title = "all patterns" if key == "overall" else PATTERNS[key].display
        plt.title(f"Critical difference diagram, {title} (alpha = 0.05)", fontsize=14, pad=20)
        plt.tight_layout()
        paths.append(ctx.out(f"extra_critical_difference_{key}.pdf"))
        fig.savefig(paths[-1], dpi=300, bbox_inches="tight")
        plt.close(fig)
    stat, p = stats.friedmanchisquare(*[table[c] for c in table.columns])
    print(f"  Friedman test over all dataset-pattern pairs: chi2={stat:.2f}, p={p:.2e}")
    return paths
