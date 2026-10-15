from copy import deepcopy

import pytest

import preflight_single_model_stronger_mean_distillation as probe
import tests.test_single_model_all_imputation_blend_preflight as previous_tests


def completed_reports():
    training, replay, gate, audit, closure, campaign, review = previous_tests.completed_reports()
    training.update(accepted_updates=56, training_episodes=1764,
        independently_regenerated_training_tasks=1764, independently_regenerated_rating_contexts=126)
    training['admission_attempts'] = {'data_rejected_steps': [2], 'optimizer_attempts': 63}
    gate['checkpoint_sha256'] = probe.SELECTED_SHA256
    closure.update(gate=deepcopy(gate), campaign=str(probe.ROOT / 'all_imputation_blend_v45_benchmarks'),
        review=str(probe.ROOT / 'all_imputation_blend_v45_review'))
    campaign['candidate']['checkpoint_sha256'] = probe.SELECTED_SHA256
    review['required_regressions'] = 219
    return training, replay, gate, audit, closure, campaign, review


@pytest.mark.parametrize('entry,key,value', [
    (0, 'accepted_updates', 57), (0, 'training_episodes', 1792),
    (0, 'admission_attempts', {'data_rejected_steps': [], 'optimizer_attempts': 64}),
    (1, 'endpoint_grids', 55), (2, 'passed', False), (2, 'checkpoint_sha256', 'other'),
    (3, 'cuda_initialized', True), (4, 'campaign', None), (5, 'all_evaluations_complete', False),
    (6, 'required_regressions', 218), (6, 'required_comparisons', 391), (6, 'comparison_tolerance', .01),
    (6, 'practical_equivalence_margin', .01), (6, 'goal_success_not_assessed', False)])
def test_complete_v45_failure_is_preserved(entry, key, value):
    reports = completed_reports()
    probe.validate_completed(*reports)
    reports[entry][key] = value
    with pytest.raises(ValueError):
        probe.validate_completed(*reports)


def test_only_new_seed_pair_and_output_are_used():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1171100901, 1172100901)
    assert probe.OUTPUT.name == 'stronger_mean_distillation_preflight_v1'
    assert probe.PARENT.name == 'selected_checkpoint.pth'
    assert probe.PARENT.parent.name == 'legacy_strata_v29'
    batch = probe.batch()
    assert len(batch['tasks']) == 28
    assert probe.blend_replay.storage_accounting(batch)['blend_tasks'] == 26
    assert probe.teacher_policy.POLICY == {'legacy': 'observed_blend', 'structured': 'observed_blend', 'recsys': 'specialist'}


def test_registration_and_runtime_bind_stronger_objective_without_changing_inference():
    assert '"mean_distillation": stronger.SPECIFICATION' in probe.REGISTER_SOURCE
    assert 'config["mean_distillation"] != stronger.SPECIFICATION' in probe.ADAPTED_SOURCE
    assert 'with stronger.training_objective(), teacher_policy.runtime(' in probe.ADAPTED_SOURCE
    assert probe.stronger.original.balanced_loss is probe.stronger.BASE_LOSS
    assert probe.ARITHMETIC == {'blas_threads': 24, 'torch_threads': 4}
    assert probe.retention.SPECIFICATION == probe.ALL_BLEND_BLUEPRINT.retention.SPECIFICATION
    assert probe.reference_identities() == probe.ALL_BLEND_BLUEPRINT.reference_identities()
    assert probe.reference_identities()['v8']['checkpoint'] == str(probe.ROOT / 'query_budget_v8/best_validation_checkpoint.pth')
    assert probe.reference_identities()['v29']['checkpoint'] == str(probe.PARENT)


def test_missing_closure_never_authorizes_new_scientific_work(monkeypatch):
    monkeypatch.setattr(probe, 'CLOSED_REPORTS', None)
    with pytest.raises(ValueError, match='eight complete immutable'):
        probe.require_completed_v45()
