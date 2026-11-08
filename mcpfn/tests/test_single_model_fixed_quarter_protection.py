from pathlib import Path

import test_single_model_fixed_half_protection as tests_blueprint
from single_model_registered_adaptation import replace_once


SOURCE = Path(tests_blueprint.__file__).read_text()
SOURCE = replace_once(SOURCE, 'import single_model_fixed_half_protection as protection',
    'import single_model_fixed_quarter_protection as protection')
SOURCE = SOURCE.replace('half_step', 'quarter_step')
SOURCE = replace_once(SOURCE, 'record["displacement_fraction"] == .5', 'record["displacement_fraction"] == .25')
SOURCE = replace_once(SOURCE, 'record["displacement_fraction"] = .25', 'record["displacement_fraction"] = WRONG_FRACTION')
SOURCE = replace_once(SOURCE, 'record["displacement_fraction"] = .5', 'record["displacement_fraction"] = .25')
SOURCE = replace_once(SOURCE, 'record["displacement_fraction"] = WRONG_FRACTION', 'record["displacement_fraction"] = .5')
SOURCE = replace_once(SOURCE, '"fraction": .5', '"fraction": .25')
SOURCE = SOURCE.replace('not_a_recomputed_half_rate', 'not_a_recomputed_quarter_rate')
exec(compile(SOURCE, __file__, 'exec'), globals())


def test_original_half_module_and_every_guard_remain_unchanged():
    half = protection.half_blueprint
    assert protection.FRACTION == .25 and half.FRACTION == .5
    for name in ('proposal_weights', 'gradient_layout', 'validate_layout', 'metric_decision',
            'paired_solve', 'paired_verify', 'paired_displacement', 'StepEvaluationError'):
        assert getattr(protection, name) is getattr(half, name)
    assert protection._update_namespace['quarter_step'] is protection.quarter_step
    assert 'quarter_step' not in half._update_namespace
    assert protection._audit_namespace['check_contraction'] is protection.check_contraction
    assert protection.PROTECTED_SOURCE_SHA256 != half.PROTECTED_SOURCE_SHA256
    assert protection.AUDIT_SOURCE_SHA256 != half.AUDIT_SOURCE_SHA256
    assert 'fixed_quarter_step' in protection.AUDIT_SOURCE
    assert 'fixed_half_step' not in protection.AUDIT_SOURCE
    assert 'Never search fractions' in protection.POLICY
    assert 'authorizes no seed' in protection.POLICY
