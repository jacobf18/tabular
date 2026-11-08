from pathlib import Path
from textwrap import dedent

import pytest

import preflight_single_model_balanced_native_quarter as probe
import test_single_model_standardized_teacher_huber_probe_recheck as blueprint


SOURCE = Path(blueprint.__file__).read_text().replace(
    'import recheck_single_model_standardized_teacher_huber_probe as review',
    'import recheck_single_model_balanced_native_quarter_probe as review')
exec(compile(SOURCE, __file__, 'exec'), globals())


@pytest.mark.parametrize('bad', ['', '0' * 63, 'x' * 64, None, 1])
def test_cli_binds_exact_fresh_configuration(bad):
    with pytest.raises(ValueError, match='configuration SHA256'):
        review.review(bad)


def test_terminal_review_uses_registered_quarter_schema_and_keeps_native_guards():
    assert review.CONFIG_SHA256 is None
    assert review.ROOT.name == 'balanced_native_quarter_preflight_v1'
    assert probe.paired.FRACTION == .25
    assert 'import preflight_single_model_balanced_native_quarter as probe' in review.SOURCE
    assert 'native_coverage' in review.SOURCE and 'native_probe_shape' in review.SOURCE
    assert 'fixed_half_step' not in review.SOURCE
    assert 'fixed_quarter_step' in review.SOURCE
    assert 'config["displacement_fraction"] == probe.paired.FRACTION == .25' in review.SOURCE
    assert 'audit["raw_endpoint"]["fixed_quarter_step"]["fraction"] == .25' in review.SOURCE
    assert review.review.__globals__ is vars(review) and review.main.__globals__ is vars(review)
    assert review._review.__globals__ is vars(review)


@pytest.mark.parametrize('registered,audited', [(.25, .25), (.5, .25), (.25, .5), (.5, .5)])
def test_review_rejects_wrong_registered_or_audited_fraction(registered, audited):
    start = review.SOURCE.index('    require(config["displacement_fraction"]')
    end = review.SOURCE.index('    record = result[\'record\']', start)
    clause = compile(dedent(review.SOURCE[start:end]), '<quarter-fraction-guard>', 'exec')
    namespace = {'require': review.require, 'probe': probe, 'config': {'displacement_fraction': registered},
        'audit': {'raw_endpoint': {'fixed_quarter_step': {'fraction': audited}}}}
    if registered == audited == .25:
        exec(clause, namespace)
    else:
        with pytest.raises(ValueError, match='registered quarter displacement'):
            exec(clause, namespace)
