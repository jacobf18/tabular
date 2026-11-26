import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
import numpy as np
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
