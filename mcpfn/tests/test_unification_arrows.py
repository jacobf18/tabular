import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def plotting(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / '69d7e55fd826e0325c3115d0/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    specification = importlib.util.spec_from_file_location(
        'unification_vertical_figure', scripts / 'plot_unification_vertical.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(('start', 'end'), [
    ((40, 64), (80, 80)),
    ((100, 64), (100, 80)),
    ((160, 64), (120, 80)),
    ((80, 129), (40, 145)),
    ((100, 129), (100, 145)),
    ((120, 129), (160, 145)),
])
def test_arrowhead_points_toward_destination_with_inverted_axis(plotting, start, end):
    figure, axis = plt.subplots(figsize=(7.09, 2.72))
    try:
        axis.set_xlim(0, 510.236)
        axis.set_ylim(plotting.HEIGHT, 0)
        plotting.flow_arrow(axis, start, end)
        figure.canvas.draw()
        arrow = axis.patches[-1]
        path = arrow.get_path()
        vertices = arrow.get_transform().transform(path.vertices)
        head_start = np.flatnonzero(path.codes == PlotPath.MOVETO)[1]
        first_base, tip, second_base = vertices[head_start:head_start + 3]
        direction = tip - (first_base + second_base) / 2
        expected = axis.transData.transform(end) - axis.transData.transform(start)
        np.testing.assert_allclose(direction / np.linalg.norm(direction),
                                   expected / np.linalg.norm(expected), atol=1e-10)
        assert direction[1] < 0
    finally:
        plt.close(figure)


def test_recommendation_ranks_include_svdpp_and_preserve_other_domains(plotting):
    _, _, _, original = plotting.read_source(plotting.ROOT / 'figures/unification.pdf')
    updated = plotting.update_recommendation_ranks(
        original, plotting.ROOT / 'results/final_model/recsys_summary.csv')
    for benchmark_index in range(4):
        for method_index, method in enumerate(plotting.SOURCE_METHODS):
            assert updated[benchmark_index][plotting.METHODS.index(method)] == original[benchmark_index][method_index]
        assert updated[benchmark_index][plotting.METHODS.index('SVD++')] is None
    for benchmark_index, expected in enumerate((
        {'MatrixFormer': 1, 'Biased MF': 2, 'SVD++': 3, 'PMF': 4, 'k-NN': 5, 'ItemKNN': 6},
        {'MatrixFormer': 3, 'Biased MF': 1, 'SVD++': 2, 'PMF': 4, 'k-NN': 5, 'ItemKNN': 6},
    ), start=4):
        assert {method: updated[benchmark_index][plotting.METHODS.index(method)]
                for method in expected} == expected


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'nonfinite'])
def test_recommendation_ranks_reject_incomplete_or_invalid_scores(plotting, tmp_path, failure):
    scores = pd.read_csv(plotting.ROOT / 'results/final_model/recsys_summary.csv')
    if failure == 'missing':
        scores = scores[scores.method != 'SVD++']
    elif failure == 'duplicate':
        scores = pd.concat([scores, scores.iloc[:1]])
    else:
        scores.loc[0, 'value'] = np.inf
    path = tmp_path / 'scores.csv'
    scores.to_csv(path, index=False)
    with pytest.raises(ValueError):
        plotting.update_recommendation_ranks([[None] * len(plotting.SOURCE_METHODS)] * 6, path)


def test_overview_renders_svdpp_row_inside_figure(plotting):
    width, _, colors, original = plotting.read_source(plotting.ROOT / 'figures/unification.pdf')
    ranks = plotting.update_recommendation_ranks(
        original, plotting.ROOT / 'results/final_model/recsys_summary.csv')
    figure = plotting.draw(width, colors, ranks)
    try:
        figure.canvas.draw()
        labels = [label for label in figure.axes[0].texts if label.get_text() == 'SVD++']
        assert len(labels) == 1
        for label in figure.axes[0].texts:
            bounds = label.get_window_extent(figure.canvas.get_renderer())
            assert bounds.y0 >= figure.bbox.y0
            assert bounds.y1 <= figure.bbox.y1
    finally:
        plt.close(figure)
