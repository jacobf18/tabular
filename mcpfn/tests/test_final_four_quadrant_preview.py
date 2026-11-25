import numpy as np
import pandas as pd
import pytest

from plot_final_four_quadrant_preview import aggregate_causal, draw_panel, select_causal


def test_causal_aggregation_weights_designs_not_draws():
    rows = []
    for path, values in [('first', [1., 3.]), ('second', [8., 10., 10., 12.])]:
        for index, value in enumerate(values):
            rows.append(dict(family='example', method='matrixformer', path=path, draw_name=str(index),
                             treated_block_rmse=value, tau_abs_error=value))
    frame = pd.DataFrame(rows)
    result = aggregate_causal(frame)
    assert (result.value == 6.).all()
    assert (result.units == 6).all()
    expected_half = 1.96*np.sqrt(1. + np.var([8., 10., 10., 12.], ddof=1)/4)/2
    np.testing.assert_allclose(result.ci_upper-result.value, expected_half)
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate_causal(pd.concat([frame, frame.iloc[:1]]))


def test_causal_selection_preserves_untreated_completion_for_rmse():
    scores = pd.DataFrame([dict(method=method, metric=metric, value=value)
                           for method, value in [('matrixformer', 1.), ('CausalPFN', 2.), ('CausalPFN [mu0]', 3.),
                                                 ('T-learner', 4.), ('T-learner [mu0]', 5.), ('matrixformer-plus', 0.)]
                           for metric in ['treated_block_rmse', 'tau_abs_error']])
    selected = select_causal(scores).set_index(['method', 'metric'])
    assert len(selected) == 6
    assert selected.loc[('CausalPFN', 'treated_block_rmse'), 'value'] == 3
    assert selected.loc[('CausalPFN', 'tau_abs_error'), 'value'] == 2
    assert selected.loc[('T-learner', 'treated_block_rmse'), 'value'] == 5
    assert selected.loc[('T-learner', 'tau_abs_error'), 'value'] == 4


def test_panel_labels_use_three_decimals_without_rounding_bar_lengths():
    import matplotlib.pyplot as plt
    from types import SimpleNamespace

    figure, axis = plt.subplots()
    data = pd.DataFrame([dict(dataset='example', method='matrixformer', label='MatrixFormer', value=.897538,
                              ci_lower=.87, ci_upper=.92, available=True)])
    style = SimpleNamespace(BLUE='blue', NEUTRAL='gray', ERROR='black', LABEL_BLUE='blue', LABEL_GRAY='gray',
                            style_axes=lambda selected: None)
    draw_panel(axis, data, style, (.82, 1.08, [.85, .95, 1.05], 4))
    assert axis.texts[0].get_text() == '0.898'
    assert axis.texts[0].get_fontsize() == 9.5
    bar = axis.patches[0]
    np.testing.assert_allclose(bar.get_x()+bar.get_width(), .897538)
    plt.close(figure)
