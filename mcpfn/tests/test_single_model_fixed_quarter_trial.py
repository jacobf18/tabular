from pathlib import Path

import test_single_model_fixed_half_trial as tests_blueprint


SOURCE = Path(tests_blueprint.__file__).read_text()
SOURCE = SOURCE.replace('single_model_fixed_half', 'single_model_fixed_quarter')
SOURCE = SOURCE.replace('fixed_half', 'fixed_quarter')
SOURCE = SOURCE.replace('= .25', '= WRONG_FRACTION')
SOURCE = SOURCE.replace(': .5', ': .25')
SOURCE = SOURCE.replace('= WRONG_FRACTION', '= .5')
exec(compile(SOURCE, __file__, 'exec'), globals())


def test_quarter_integration_is_explicit_and_original_half_module_is_unchanged():
    half = integration.half_blueprint
    assert integration.protection.FRACTION == .25 and half.protection.FRACTION == .5
    assert 'fixed_half' not in integration.SOURCE and 'half_step' not in integration.SOURCE
    assert 'Structured half-weight V8/V29 retention is unchanged' in integration.POLICY
    assert 'authorizes no scientific seed' in integration.POLICY
    source = 'config = {"policy": POLICY,\n    sources.update(Path(module.__file__).resolve()'
    adapted = integration.adapt_registration(source)
    assert 'test_single_model_fixed_quarter_trial.py' in adapted
    assert 'test_single_model_fixed_half_trial.py' not in adapted


@pytest.mark.parametrize('fraction,key', [(.25, 'fixed_quarter_step'), (.5, 'fixed_quarter_step'),
    (.25, 'fixed_half_step'), (.5, 'fixed_half_step')])
def test_preflight_requires_actual_quarter_optimizer_audit(fraction, key):
    result = {'record': {'displacement_fraction': fraction}}
    audit = {'raw_endpoint': {key: {'fraction': fraction, 'evaluated': True,
        'parameters': 77998496, 'actual_saved_displacements_checked': True}}}
    if fraction == .25 and key == 'fixed_quarter_step':
        integration.check_preflight_evidence(result, audit)
    else:
        with pytest.raises(ValueError, match='fixed-quarter optimizer audit'):
            integration.check_preflight_evidence(result, audit)
