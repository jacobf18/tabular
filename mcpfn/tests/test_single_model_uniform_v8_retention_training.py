import types

import test_single_model_family_balanced_retention_training as uniform_training_blueprint
import test_single_model_cohort_anchor_retention_training as uniform_reference_blueprint


SOURCE = uniform_training_blueprint.SOURCE.replace('family_balanced_retention', 'uniform_v8_retention')
exec(compile(SOURCE, __file__, 'exec'), globals())

for blueprint, names in (
    (uniform_training_blueprint, ('test_full_replay_rejects_changed_retention_weights',
        'test_qualification_rejects_rehashed_weight_changes')),
    (uniform_reference_blueprint, ('test_both_original_references_must_be_bound_in_full_trial_registration',
        'test_full_replay_never_swaps_or_mutates_references',
        'test_qualification_requires_cohort_bindings_as_well_as_full_grid_hashes')),
):
    for name in names:
        function = getattr(blueprint, name)
        assert function.__closure__ is None
        cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
        cloned.__kwdefaults__ = function.__kwdefaults__
        cloned.__dict__.update(function.__dict__)
        globals()[name] = cloned


def test_full_trial_replays_only_v8_for_all_original_imputation_tasks(tmp_path, monkeypatch):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    replay_all(state)
    hashes = state.finish()
    report = integration.require_report(training, {'artifact_sha256': hashes, 'admission_attempts': counts})
    assert report['reference_role_counts'] == {'v8': 26}
    assert [role for role, source in calls] == ['v8'] * 26
    assert report['reference'] == {role: {**identity, 'state_tensors': 254, 'bitwise_equal': True}
        for role, identity in probe.reference_identities().items()}
    assert integration.retention is probe.retention
    assert integration.retention.BACKWARD_SOURCE_SHA256 == integration.previous.retention.BACKWARD_SOURCE_SHA256
    for name in ('identity', 'cohort_identity', 'original_identity', 'verify_references'):
        assert getattr(integration, name).__globals__ is vars(integration)
