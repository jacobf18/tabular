"""
Paper tables as LaTeX.

The tables use the pattern macros defined in `macros.tex` (written next to the
tables), plus the booktabs, multirow and arydshln packages.
"""
from __future__ import annotations

import pandas as pd

from .. import datasets as ds
from ..methods import display_name, specialist_key
from ..patterns import PATTERNS
from .context import Context, MissingInput, require
from .style import PATTERN_ORDER, ci95_halfwidth

# LaTeX macro for each pattern (as used in the paper source) and its rendering
PATTERN_MACROS = {
    "mcar": ("\\mcar", "\\texttt{MCAR}"),
    "col_mar": ("\\colmar", "\\texttt{Col-MAR}"),
    "self_mnar": ("\\mnarself", "\\texttt{Self-MNAR}"),
    "nn_mnar": ("\\nnmar", "\\texttt{NN-MNAR}"),
    "block_mnar": ("\\marblockneural", "\\texttt{Block-MNAR}"),
    "pol_mnar": ("\\polarmnar", "\\texttt{Pol-MNAR}"),
    "soft_pol_mnar": ("\\softpolarmnar", "\\texttt{Soft-Pol-MNAR}"),
    "latent_mnar": ("\\latentmnar", "\\texttt{Latent-MNAR}"),
    "clust_mnar": ("\\clustermnar", "\\texttt{Clust-MNAR}"),
    "phase_mnar": ("\\twophasemnar", "\\texttt{Phase-MNAR}"),
    "censor_mnar": ("\\censormnar", "\\texttt{Censor-MNAR}"),
}
GROUP_MACROS = {"MCAR": "\\mcar", "MAR": "\\mar", "MNAR": "\\mnar"}
# Short column headers for the specialized-model tables
SHORT_HEADERS = {
    "mcar": "\\mcar", "col_mar": "\\colmar", "self_mnar": "\\texttt{Self}", "nn_mnar": "\\texttt{NN}",
    "block_mnar": "\\texttt{Block}", "pol_mnar": "\\texttt{Pol}", "soft_pol_mnar": "\\texttt{Soft-Pol}",
    "latent_mnar": "\\texttt{Latent}", "clust_mnar": "\\texttt{Clust}", "phase_mnar": "\\texttt{Phase}",
    "censor_mnar": "\\texttt{Censor}",
}

PREAMBLE = """% Requires \\usepackage{booktabs, multirow, arydshln} and the macros in macros.tex
"""


def write_macros(ctx: Context):
    lines = ["% Pattern macros used by the generated tables"]
    lines += [f"\\providecommand{{{macro}}}{{{text}}}" for macro, text in PATTERN_MACROS.values()]
    lines += ["\\providecommand{\\mar}{\\texttt{MAR}}", "\\providecommand{\\mnar}{\\texttt{MNAR}}"]
    path = ctx.out("macros.tex")
    path.write_text("\n".join(lines) + "\n")
    return path


def _summary(df: pd.DataFrame, metric: str, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and 95% CI half-width per (pattern or "overall") x method.

    Pattern rows aggregate over datasets; the overall row over dataset-pattern pairs.
    """
    df = df[df["method"].isin(methods)]
    groups = list(df.groupby("pattern")) + [("overall", df)]
    means = pd.DataFrame({k: g.groupby("method")[metric].mean() for k, g in groups}).T
    cis = pd.DataFrame({k: g.groupby("method")[metric].apply(ci95_halfwidth) for k, g in groups}).T
    rows = [p for p in PATTERN_ORDER if p in means.index] + ["overall"]
    return means.loc[rows, methods], cis.loc[rows, methods]


def _row_labels(rows: list[str]):
    """Yield (group cell, pattern cell, separator before row) for the pattern rows of a table."""
    group_sizes: dict[str, int] = {}
    for r in rows:
        if r != "overall":
            group_sizes[PATTERNS[r].group] = group_sizes.get(PATTERNS[r].group, 0) + 1
    prev = None
    for r in rows:
        if r == "overall":
            yield "", "Overall", "\\midrule\n"
            continue
        group = PATTERNS[r].group
        sep = "\\noalign{\\vskip 1.5pt}\n\\hdashline\n\\noalign{\\vskip 1.5pt}\n" if prev not in (None, group) else ""
        if group != prev:
            n = group_sizes[group]
            cell = f"\\textbf{{{GROUP_MACROS[group]}}}" if n == 1 else f"\\multirow{{{n}}}{{*}}{{\\textbf{{{GROUP_MACROS[group]}}}}}"
        else:
            cell = ""
        prev = group
        yield cell, PATTERN_MACROS[r][0], sep


def _fmt(mean: float, ci: float | None, bold: bool) -> str:
    if pd.isna(mean):
        return "--"
    text = f"{mean:.3f}" if ci is None or pd.isna(ci) else f"{mean:.3f} $\\pm$ {ci:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def _method_table(means, cis, columns: list[str], headers: list[str], caption: str, label: str,
                  bold, max_cols: int = 5, header_row=None, corner: str = "Pattern") -> str:
    """Patterns as rows, `columns` as columns, split into tabulars of at most `max_cols` columns.

    `bold(row, column)` says whether a cell is bolded. `header_row(start, n)` optionally
    returns an extra header line for the tabular holding columns start..start+n.
    """
    out = [PREAMBLE, "\\begin{table*}[!ht]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    for start in range(0, len(columns), max_cols):
        cols = columns[start:start + max_cols]
        heads = headers[start:start + max_cols]
        out.append(f"\\begin{{tabular}}{{ll{'c' * len(cols)}}}")
        out.append("\\toprule")
        if header_row is not None:
            out.append(header_row(start, len(cols)))
        out.append(f" & {corner} & " + " & ".join(heads) + " \\\\")
        out.append("\\midrule")
        body = ""
        for (group_cell, pattern_cell, sep), row in zip(_row_labels(list(means.index)), means.index):
            cells = [_fmt(means.loc[row, c], None if cis is None else cis.loc[row, c], bold(row, c)) for c in cols]
            body += sep + f"{group_cell} & {pattern_cell} & " + " & ".join(cells) + " \\\\\n"
        out.append(body.rstrip("\n"))
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        if start + max_cols < len(columns):
            out.append("\\quad")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


def _write(ctx: Context, name: str, text: str):
    path = ctx.out(name)
    path.write_text(text)
    write_macros(ctx)
    return [path]


def _baseline_table(ctx: Context, metric: str, lower_is_better: bool, caption: str, label: str,
                    filename: str, max_methods: int | None = None):
    df = ctx.scores("openml", metric)
    methods = ctx.baselines(df)
    means, cis = _summary(df, metric, methods)
    # Order columns by overall score, best first
    order = means.loc["overall"].sort_values(ascending=lower_is_better).index.tolist()[:max_methods]
    # The best method across all methods is bolded, even if it is not shown
    best = means.min(axis=1) if lower_is_better else means.max(axis=1)
    text = _method_table(means, cis, order, [display_name(m) for m in order], caption, label,
                         bold=lambda r, c: abs(means.loc[r, c] - best[r]) < 1e-9)
    return _write(ctx, filename, text)


def table_rmse_main(ctx: Context):
    """Table 1: RMSE +- 95% CI by pattern for the five best methods overall."""
    return _baseline_table(ctx, "nrmse", True, "RMSE $\\pm$ 95\\% confidence interval by missingness pattern",
                           "tab:rmse_main", "table1_rmse_top5.tex", max_methods=5)


def table_rmse_full(ctx: Context):
    """Table 3: RMSE +- 95% CI by pattern for all methods."""
    return _baseline_table(ctx, "nrmse", True, "RMSE $\\pm$ 95\\% confidence interval by missingness pattern",
                           "tab:rmse_full", "table3_rmse_all_methods.tex")


def table_r2(ctx: Context):
    """Table 4: mean column-wise R^2 +- 95% CI by pattern."""
    return _baseline_table(ctx, "r2", False,
                           "Mean column-wise $R^2$ $\\pm$ 95\\% confidence interval by missingness pattern",
                           "tab:r2", "table4_r2.tex")


def _specialist_summary(ctx: Context):
    df = ctx.scores("openml")
    trained_on = ["mcar"] + list(ctx.specialists(df))
    if len(trained_on) < 2:
        raise MissingInput("no pattern-specific TabImpute results (run impute with --checkpoint PATTERN=PATH)")
    methods = [specialist_key(p) for p in trained_on]
    means, cis = _summary(df, "nrmse", methods)
    # Top-2 per row are bolded
    second_best = means.apply(lambda row: row.nsmallest(2).max(), axis=1)
    return trained_on, methods, means, cis, (lambda r, c: means.loc[r, c] <= second_best[r] + 1e-9)


def _group_header(trained_on: list[str]):
    def header(start: int, n: int) -> str:
        cells, i = [], start
        while i < start + n:
            group = PATTERNS[trained_on[i]].group
            j = i
            while j < start + n and PATTERNS[trained_on[j]].group == group:
                j += 1
            macro = f"\\textbf{{{GROUP_MACROS[group]}}}"
            cells.append(macro if j - i == 1 else f"\\multicolumn{{{j - i}}}{{{'c' if i == start else ':c'}}}{{{macro}}}")
            i = j
        return " & & " + " & ".join(cells) + " \\\\\n\\midrule"
    return header


def table_specialists(ctx: Context):
    """Table 5: RMSE +- 95% CI of TabImpute models trained on each pattern (top-2 bolded)."""
    trained_on, methods, means, cis, bold = _specialist_summary(ctx)
    text = _method_table(
        means, cis, methods, [SHORT_HEADERS[p] for p in trained_on],
        "RMSE $\\pm$ 95\\% Confidence Interval for Specialized TabImpute Models \\\\ "
        "The top-2 are bolded for each missingness pattern.",
        "tab:specialized_full", bold, header_row=_group_header(trained_on), corner="Trained on $\\Rightarrow$",
    )
    return _write(ctx, "table5_specialized_models_ci.tex", text)


def table_specialists_compact(ctx: Context):
    """Table 2: RMSE of TabImpute models trained on each pattern, one tabular (top-2 bolded)."""
    trained_on, methods, means, _, bold = _specialist_summary(ctx)
    text = _method_table(
        means, None, methods, [SHORT_HEADERS[p] for p in trained_on],
        "RMSE for specialized TabImpute models. The top-2 models are bolded for each missingness pattern.",
        "tab:specialized", bold, max_cols=len(methods), header_row=_group_header(trained_on),
        corner="Trained on $\\Rightarrow$",
    )
    return _write(ctx, "table2_specialized_models.tex", text)


def table_nonlinear(ctx: Context):
    """Table 7: default TabImpute vs. TabImpute trained on a nonlinear factor model."""
    df = ctx.scores("openml")
    methods = ["tabimpute", "tabimpute_nonlinear"]
    require(df, methods, "Table 7")
    means, cis = _summary(df, "nrmse", methods)
    best = means.min(axis=1)
    text = _method_table(means, cis, methods, ["TabImpute", "TabImpute (Nonlinear)"],
                         "RMSE $\\pm$ 95\\% confidence interval by missingness pattern for the default "
                         "(linear factor model) TabImpute and a TabImpute model trained under a nonlinear "
                         "factor model with \\mcar missingness.",
                         "tab:nonlinear", lambda r, c: abs(means.loc[r, c] - best[r]) < 1e-9)
    return _write(ctx, "table7_linear_vs_nonlinear.tex", text)


def table_pattern_parameters(ctx: Context):
    """Table 8: parameters of each synthetic missingness pattern."""
    out = ["\\begin{table}[!ht]", "\\centering", "\\caption{Synthetic Data Generation Parameters}",
           "\\label{tab:pattern_parameters}", "\\begin{tabular}{|l|l|c|c|}", "\\hline",
           "\\textbf{Missingness Pattern} & \\textbf{Parameter Name} & \\textbf{Symbol} & \\textbf{Value} \\\\",
           "\\hline"]
    for key in PATTERN_ORDER:
        params = PATTERNS[key].table_params
        for i, (name, symbol, value) in enumerate(params):
            first = PATTERN_MACROS[key][0] if len(params) == 1 else f"\\multirow{{{len(params)}}}{{*}}{{{PATTERN_MACROS[key][0]}}}"
            out.append(f"{first if i == 0 else ''} & {name} & {symbol} & {value} \\\\")
        out.append("\\hline")
    out += ["\\end{tabular}", "\\end{table}"]
    return _write(ctx, "table8_pattern_parameters.tex", "\n".join(out) + "\n")


def table_datasets(ctx: Context):
    """Table 9: the OpenML datasets."""
    out = ["\\begin{table}[!ht]", "\\centering",
           "\\caption{OpenML datasets. All data is under the CC-BY 4.0 license via OpenML. "
           "OpenML's code is under the 3-Clause BSD license.}",
           "\\label{tab:datasets}", "\\small", "\\begin{tabular}{p{3.5cm}lp{3cm}p{5cm}}", "\\toprule",
           "\\textbf{Dataset} & \\textbf{Size} & \\textbf{Domain} & \\textbf{Description} \\\\", "\\midrule"]
    for name, d in ds.OPENML_DATASETS.items():
        escaped = name.replace("_", "\\_")
        out.append(f"{escaped} & ${d.rows} \\times {d.cols}$ & {d.domain} & {d.description} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return _write(ctx, "table9_datasets.tex", "\n".join(out) + "\n")


def table_uci_rmse(ctx: Context):
    """RMSE +- 95% CI on the UCI datasets per missingness level (extra)."""
    df = ctx.scores("uci")
    methods = ctx.baselines(df)
    # Average over repeats so the CI is over datasets
    df = df.groupby(["dataset", "p", "method"], as_index=False)["nrmse"].mean()
    means = df.pivot_table(index="p", columns="method", values="nrmse")[methods]
    cis = df.groupby(["p", "method"])["nrmse"].apply(ci95_halfwidth).unstack()[methods]
    order = means.mean().sort_values().index.tolist()
    best = means.min(axis=1)
    lines = [PREAMBLE, "\\begin{table*}[!ht]", "\\centering",
             "\\caption{RMSE $\\pm$ 95\\% confidence interval on the UCI datasets under \\mcar}",
             "\\label{tab:uci_rmse}", f"\\begin{{tabular}}{{l{'c' * len(order)}}}", "\\toprule",
             "$p$ & " + " & ".join(display_name(m) for m in order) + " \\\\", "\\midrule"]
    for p in means.index:
        cells = [_fmt(means.loc[p, m], cis.loc[p, m], abs(means.loc[p, m] - best[p]) < 1e-9) for m in order]
        lines.append(f"{p:g} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return _write(ctx, "extra_uci_rmse_table.tex", "\n".join(lines) + "\n")


def table_win_rate(ctx: Context):
    """Win rate (% of datasets with the lowest RMSE) per pattern (extra)."""
    df = ctx.scores("openml", "rmse")
    table = df.pivot_table(index=["dataset", "pattern"], columns="method", values="rmse")
    methods = ctx.baselines(df)
    winners = table[methods].dropna().idxmin(axis=1).rename("winner").reset_index()
    rates = (pd.crosstab(winners["winner"], winners["pattern"], normalize="columns") * 100)
    rates = rates.reindex(index=methods, columns=[p for p in PATTERN_ORDER if p in rates.columns]).fillna(0)
    best = rates.max(axis=0)
    lines = ["\\begin{table}[!ht]", "\\centering", "\\caption{Win rates (\\%) by missingness pattern}",
             "\\label{tab:win_rates}", f"\\begin{{tabular}}{{l{'c' * len(rates.columns)}}}", "\\toprule",
             "Method & " + " & ".join(PATTERN_MACROS[p][0] for p in rates.columns) + " \\\\", "\\midrule"]
    for m in methods:
        cells = [f"\\textbf{{{v:.1f}}}" if v == best[p] and v > 0 else f"{v:.1f}" for p, v in rates.loc[m].items()]
        lines.append(f"{display_name(m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return _write(ctx, "extra_win_rate_table.tex", "\n".join(lines) + "\n")
