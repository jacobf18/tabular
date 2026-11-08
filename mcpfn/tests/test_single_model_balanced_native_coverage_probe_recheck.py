from pathlib import Path

import pytest

import test_single_model_standardized_teacher_huber_probe_recheck as blueprint


SOURCE = Path(blueprint.__file__).read_text().replace(
    'import recheck_single_model_standardized_teacher_huber_probe as review',
    'import recheck_single_model_balanced_native_coverage_probe as review')
exec(compile(SOURCE, __file__, 'exec'), globals())


@pytest.mark.parametrize('bad', ['', '0' * 63, 'x' * 64, None, 1])
def test_cli_binds_exact_original_configuration(bad):
    with pytest.raises(ValueError, match='configuration SHA256'):
        review.review(bad)


def test_new_review_has_no_fixed_historical_registration_or_path():
    assert review.CONFIG_SHA256 is None
    assert review.ROOT.name == 'balanced_native_coverage_preflight_v1'
    assert 'native_coverage' in review.SOURCE and 'native_probe_shape' in review.SOURCE
    assert '7bf3acc2e92274eda8e5254546913f49353f1700091292d78b4c0ed93a0641d8' not in review.SOURCE
