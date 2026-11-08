import ast
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from seaborn.algorithms import bootstrap


def overall_plot_data(frame, methods, variable="df_overall", script="plot_negative_rmse.py"):
    source = Path(__file__).resolve().parents[1] / "benchmark/openml" / script
    tree = ast.parse(source.read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)]
    assert len(assignments) == 1
    namespace = {"df_norm_all": frame, "df_all": frame, "_cols_overall": methods}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[variable]


def test_overall_uses_42_dataset_means_not_462_pattern_scores():
    index = pd.MultiIndex.from_product([range(42), range(11)], names=["dataset", "pattern"])
    values = np.random.default_rng(0).normal(size=(42 * 11, 3))
    frame = pd.DataFrame(values, index=index, columns=["TabImpute", "Baseline", "Excluded"])
    result = overall_plot_data(frame, ["TabImpute", "Baseline"])
    assert result.shape == (42, 2)
    np.testing.assert_allclose(result.to_numpy(), values[:, :2].reshape(42, 11, 2).mean(axis=1))
    np.testing.assert_allclose(result.mean(), frame[["TabImpute", "Baseline"]].mean())
    assert frame.shape == (462, 3)
    assert all(len(frame.xs(pattern, level="pattern")) == 42 for pattern in range(11))


def test_patterns_are_not_independent_bootstrap_units():
    index = pd.MultiIndex.from_tuples([("first", "MCAR"), ("first", "MAR"), ("second", "MCAR")])
    frame = pd.DataFrame({"TabImpute": [1.0, 1.0, 3.0]}, index=index)
    result = overall_plot_data(frame, ["TabImpute"])
    np.testing.assert_array_equal(result.TabImpute.to_numpy(), [1.0, 3.0])
    assert result.TabImpute.mean() == 2.0


def test_missing_score_keeps_existing_omission_without_dropping_the_dataset():
    index = pd.MultiIndex.from_product([['first', 'second'], ['MCAR', 'MAR']])
    frame = pd.DataFrame({'Baseline': [1.0, np.nan, 3.0, 5.0]}, index=index)
    result = overall_plot_data(frame, ['Baseline'])
    np.testing.assert_array_equal(result.Baseline.to_numpy(), [1.0, 4.0])
    assert result.Baseline.mean() == 2.5


def test_tables_use_the_same_dataset_means_as_plots():
    index = pd.MultiIndex.from_product([range(42), range(11)])
    frame = pd.DataFrame(np.random.default_rng(1).normal(size=(462, 2)), index=index,
                         columns=["TabImpute", "MissForest"])
    plotted = overall_plot_data(frame, list(frame.columns))
    tabulated = overall_plot_data(frame, list(frame.columns), variable="df_all_norm")
    pd.testing.assert_frame_equal(plotted, tabulated)


@pytest.mark.parametrize("mean,interval,threshold,expected", [
    (2., (1., 4.), None, " & 2.000 $\\pm$ 2.000"),
    (2., (0., 3.), None, " & 2.000 $\\pm$ 2.000"),
    (2., (1., 3.), None, " & 2.000 $\\pm$ 1.000"),
    (2., (2., 2.), None, " & 2.000 $\\pm$ 0.000"),
    (2., (1., 4.), 2., " & \\textbf{2.000 $\\pm$ 2.000}"),
    (2., (np.nan, np.nan), None, " & 2.000 $\\pm$ --"),
    (np.nan, (np.nan, np.nan), None, " & --"),
])
def test_table_formatter_uses_plus_minus_enclosing_both_bootstrap_limits(mean, interval, threshold, expected):
    source = Path(__file__).resolve().parents[1] / "benchmark/openml/plot_negative_rmse.py"
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"_fmt_estimate", "_fmt_cell"}]
    namespace = {"pd": pd, "np": np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["_fmt_cell"](mean, interval, threshold) == expected


def test_specialized_ci_table_stacks_two_blocks_without_changing_means_or_bolding():
    source = Path(__file__).resolve().parents[1] / "benchmark/openml/plot_negative_rmse.py"
    tree = ast.parse(source.read_text())
    variables = {"_square_patterns", "_square_pattern_groups", "_square_group_macro", "_square_col_header_macro"}
    namespace = {"pd": pd, "np": np}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in variables:
                    namespace[target.id] = ast.literal_eval(node.value)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"_fmt_estimate", "_fmt_cell", "_render_specialized_square_table"}]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    patterns = namespace["_square_patterns"]
    namespace["_square_columns"] = [(pattern, pattern) for pattern in patterns]
    namespace["_square_group_size"] = {"MCAR": 3, "MAR": 2, "MNAR": 6}
    namespace["patern_latex_names"] = {pattern: pattern for pattern in patterns}
    namespace["_square_means"] = {
        (pattern, training_pattern): 1. + column_index / 10
        for pattern in patterns + ["Overall"]
        for column_index, training_pattern in enumerate(patterns)
    }
    namespace["_square_intervals"] = {
        key: (mean - .1, mean + .2) for key, mean in namespace["_square_means"].items()
    }
    namespace["_row_thresholds"] = {pattern: 1.1 for pattern in patterns}
    namespace["_overall_thresh"] = 1.1
    with_ci = namespace["_render_specialized_square_table"](True)
    without_ci = namespace["_render_specialized_square_table"](False)
    assert with_ci.count("$\\pm$") == 133
    assert "$\\pm$" not in without_ci
    assert "\\label{tab:specialized_square}" in with_ci
    assert "\\label{tab:specialized_square_no_ci}" in without_ci
    assert with_ci.count("\\par\\medskip") == 1
    assert "\\begin{tabular}{llccc:cc}" in with_ci
    assert "\\begin{tabular}{llcccccc}" in with_ci
    assert "\\begin{tabular}{llccc:cc:cccccc}" in without_ci
    for table, block_count in [(with_ci, 2), (without_ci, 1)]:
        assert table.count("\\begin{tabular}") == block_count
        assert table.count("\\hdashline") == 2 * block_count
        for group, size in [("mcar", 3), ("mar", 2), ("mnar", 6)]:
            assert table.count(f"\\multirow{{{size}}}{{*}}{{\\textbf{{\\{group}}}}}") == block_count
    ci_blocks = with_ci.split("\\begin{tabular}")[1:]
    assert "\\multicolumn{3}{c:}{\\textbf{\\mcar}}" in ci_blocks[0]
    assert "\\multicolumn{2}{c}{\\textbf{\\mar}}" in ci_blocks[0]
    assert "\\multicolumn{6}{c}{\\textbf{\\mnar}}" in ci_blocks[1]
    block_rows = []
    for block, column_count in zip(ci_blocks, [5, 6]):
        rows = [line.split(" & ") for line in block.splitlines() if line.startswith(("\\multirow", " & "))]
        assert len(rows) == 12
        assert all(len(row) == column_count + 2 for row in rows)
        block_rows.append(rows)
    mean_rows = [line.split(" & ") for line in without_ci.splitlines() if line.startswith(("\\multirow", " & "))]
    for first_row, second_row, mean_row in zip(*block_rows, mean_rows):
        assert first_row[:2] == second_row[:2] == mean_row[:2]
        combined = [cell.removesuffix(r" \\") for cell in first_row[2:] + second_row[2:]]
        expected = [cell.removesuffix(r" \\") for cell in mean_row[2:]]
        assert [re.sub(r" \$\\pm\$ \d+\.\d{3}", "", cell) for cell in combined] == expected
        assert sum("\\textbf" in cell for cell in combined) == 2
        assert not any("\\textbf" in cell for cell in second_row[2:])
    assert namespace["_fmt_cell"](np.nan, (np.nan, np.nan), None, include_intervals=False) == " & --"


def test_tables_share_recategorized_pattern_order_and_groups():
    source = Path(__file__).resolve().parents[1] / "benchmark/openml/plot_negative_rmse.py"
    tree = ast.parse(source.read_text())
    variables = {"all_patterns", "_pattern_groups", "_square_patterns", "_square_pattern_groups"}
    settings = {
        target.id: ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in variables
    }
    expected_groups = {
        "MCAR": ["MCAR", "MNARLatentFactorPattern", "MNARClusterLevelPattern"],
        "MAR": ["MAR", "MNARTwoPhaseSubsetPattern"],
        "MNAR": ["MNAR", "MAR_Neural", "MAR_BlockNeural", "MNARPolarizationPattern",
                 "MNARSoftPolarizationPattern", "MNARCensoringPattern"],
    }
    expected_order = [pattern for patterns in expected_groups.values() for pattern in patterns]
    expected_mapping = {
        pattern: group for group, patterns in expected_groups.items() for pattern in patterns
    }
    assert settings["all_patterns"] == expected_order + ["Overall"]
    assert settings["_square_patterns"] == expected_order
    assert settings["_pattern_groups"] == {**expected_mapping, "Overall": None}
    assert settings["_square_pattern_groups"] == expected_mapping


def test_r_squared_table_matches_nrmse_pattern_grouping():
    base = Path(__file__).resolve().parents[1] / "benchmark/openml"
    settings = []
    for script in ["plot_negative_rmse.py", "plot_r_squared.py"]:
        tree = ast.parse((base / script).read_text())
        settings.append({
            target.id: ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in {"all_patterns", "_pattern_groups"}
        })
    assert settings[0] == settings[1]


def test_r_squared_overall_averages_patterns_within_datasets():
    index = pd.MultiIndex.from_product([range(42), range(11)])
    frame = pd.DataFrame({"TabImpute": np.arange(462) / 462}, index=index)
    actual = overall_plot_data(frame, list(frame), script="plot_r_squared.py")
    pd.testing.assert_frame_equal(actual, frame.groupby(level=0).mean())
    assert len(actual) == 42
    uneven = frame.drop(index=(0, 0))
    actual = overall_plot_data(uneven, list(uneven), script="plot_r_squared.py")
    pd.testing.assert_frame_equal(actual, uneven.groupby(level=0).mean())


def test_r_squared_ci_uses_same_symmetric_bootstrap_envelope():
    base = Path(__file__).resolve().parents[1] / "benchmark"
    statistics_tree = ast.parse((base / "missbench_statistics.py").read_text())
    table_tree = ast.parse((base / "openml/plot_r_squared.py").read_text())
    definitions = [
        node for tree in [statistics_tree, table_tree] for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"bootstrap_mean_interval", "_ci95_halfwidth"}
    ]
    namespace = {"np": np, "pd": pd, "bootstrap": bootstrap}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "r_squared_ci", "exec"), namespace)
    scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.95, np.nan])
    lower, upper = np.percentile(bootstrap(scores.dropna(), func=np.mean, n_boot=1000, seed=0), [2.5, 97.5])
    assert namespace["_ci95_halfwidth"](scores) == pytest.approx(max(scores.mean() - lower, upper - scores.mean()))
    assert np.isnan(namespace["_ci95_halfwidth"](pd.Series([0.5])))


def nonlinear_table_namespace():
    base = Path(__file__).resolve().parents[1] / "benchmark"
    statistics_tree = ast.parse((base / "missbench_statistics.py").read_text())
    table_tree = ast.parse((base / "openml/plot_negative_rmse.py").read_text())
    namespace = {"np": np, "pd": pd, "bootstrap": bootstrap}
    for node in ast.walk(table_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"all_patterns", "_pattern_groups", "_group_latex_macro"}:
                    namespace[target.id] = ast.literal_eval(node.value)
    definitions = [node for tree in [statistics_tree, table_tree] for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in {
                       "bootstrap_mean_interval", "_fmt_estimate", "_fmt_cell",
                       "_summarize_nonlinear_comparison", "_render_nonlinear_comparison_table",
                   }]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "nonlinear_table", "exec"), namespace)
    namespace["_group_size"] = {"MCAR": 3, "MAR": 2, "MNAR": 6}
    namespace["patern_latex_names"] = {pattern: pattern for pattern in namespace["all_patterns"]}
    return namespace


def test_nonlinear_comparison_bootstraps_datasets_and_weights_overall_equally():
    namespace = nonlinear_table_namespace()
    patterns = namespace["all_patterns"][:-1]
    index = pd.MultiIndex.from_product([range(42), patterns])
    scores = pd.DataFrame(np.random.default_rng(0).normal(size=(462, 2)), index=index,
                          columns=["TabImpute", "TabImpute (Nonlinear)"]).drop(index=(0, "MCAR"))
    summary = namespace["_summarize_nonlinear_comparison"](scores)
    assert len(summary) == 24
    for result in summary.to_dict("records"):
        data = scores.groupby(level=0).mean() if result["Pattern"] == "Overall" else scores.xs(result["Pattern"], level=1)
        values = data[result["Method"]].to_numpy()
        lower, upper = np.percentile(bootstrap(values, func=np.mean, n_boot=1000, seed=0), [2.5, 97.5])
        assert result["mean"] == pytest.approx(values.mean())
        assert result["ci_low"] == pytest.approx(lower)
        assert result["ci_high"] == pytest.approx(upper)
        assert result["ci_half_width"] == pytest.approx(max(values.mean() - lower, upper - values.mean()))
        assert result["datasets"] == (41 if result["Pattern"] == "MCAR" else 42)


def test_nonlinear_table_uses_recategorized_groups_plus_minus_and_row_best():
    namespace = nonlinear_table_namespace()
    patterns = namespace["all_patterns"][:-1]
    index = pd.MultiIndex.from_product([range(3), patterns])
    scores = pd.DataFrame({"TabImpute": 1., "TabImpute (Nonlinear)": 2.}, index=index)
    scores.loc[(slice(None), "MAR"), "TabImpute (Nonlinear)"] = .5
    scores.loc[(slice(None), "MNAR"), "TabImpute (Nonlinear)"] = 1.
    summary = namespace["_summarize_nonlinear_comparison"](scores)
    table = namespace["_render_nonlinear_comparison_table"](summary)
    assert table.count("$\\pm$") == 25
    assert table.count("\\hdashline") == 2
    assert "TabImpute & TabImpute (Nonlinear)" in table
    for group, size in [("mcar", 3), ("mar", 2), ("mnar", 6)]:
        assert table.count(f"\\multirow{{{size}}}{{*}}{{\\textbf{{\\{group}}}}}") == 1
    rows = [line.split(" & ") for line in table.splitlines() if line.startswith(("\\multirow", " & "))]
    assert [row[1] for row in rows] == namespace["all_patterns"]
    for pattern, row in zip(namespace["all_patterns"], rows):
        results = summary[summary.Pattern == pattern]
        assert len(row) == 4
        for result, cell in zip(results.to_dict("records"), row[2:]):
            assert ("\\textbf" in cell) == (result["mean"] == results["mean"].min())
            assert f"{result['mean']:.3f} $\\pm$ {result['ci_half_width']:.3f}" in cell
