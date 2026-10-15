import pytest

from diagnose_dwip_recsys_rating_tails import describe_tails


def test_tail_description_respects_supplied_macro_weights():
    result = describe_tails([-3., 0., 3.], [1., 3., 5.], [.25, .5, .25])
    assert result["residual_mean"] == 0
    assert result["residual_std"] == pytest.approx(4.5 ** .5)
    assert result["below_minus_two"] == result["above_plus_two"] == .25
    assert result["rating_probabilities"] == {"1": .25, "2": 0., "3": .5, "4": 0., "5": .25}
    assert result["skewness"] == pytest.approx(0)
