import inspect
import types

import test_single_model_general_native_retention_training as blueprint
import test_single_model_native_column_training as native_tests
import preflight_single_model_parent_structured_retention as probe
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


base_fixtures = types.ModuleType("parent_structured_test_fixtures")
base_fixtures.__dict__.update({**vars(blueprint.base_fixtures), "probe": probe})
exec(compile(blueprint.fixture_blueprint.FIXTURE_SOURCE, __file__, "exec"), vars(base_fixtures))
SOURCE = blueprint.SOURCE.replace("single_model_general_native_retention_training", "single_model_parent_structured_training")
SOURCE = SOURCE.replace("preflight_single_model_general_native_retention", "preflight_single_model_parent_structured_retention")
SOURCE = SOURCE.replace('("v8", "general")', '("v29", "general")')
SOURCE = SOURCE.replace('probe.reference_identities()["v8"]["checkpoint"]', 'str(probe.training.alignment.fixed.PARENT)')
SOURCE = SOURCE.replace('input_sha256={str(probe.PARENT): probe.PARENT_SHA256,',
    'input_sha256={str(probe.PARENT): probe.PARENT_SHA256, '
    'str(probe.training.alignment.fixed.PARENT): probe.probe_blueprint.reference_identities()["v8"]["sha256"],')
SOURCE = SOURCE.replace('"input_sha256": {identity["checkpoint"]: identity["sha256"] for identity in probe.reference_identities().values()}',
    '"input_sha256": {str(probe.training.alignment.fixed.PARENT): probe.probe_blueprint.reference_identities()["v8"]["sha256"], '
    '**{identity["checkpoint"]: identity["sha256"] for identity in probe.reference_identities().values()}}')
SOURCE = SOURCE.replace("config.update(batches=[batch],", "config.update(native_teacher_mean=integration.retention.MEAN_SPECIFICATION, batches=[batch],")
exec(compile(SOURCE, __file__, "exec"), globals())

for name in ("test_both_original_references_must_be_bound_in_full_trial_registration",
        "test_full_replay_never_swaps_or_mutates_references", "test_qualification_requires_cohort_bindings_as_well_as_full_grid_hashes",
        "test_full_replay_rejects_changed_retention_weights", "test_qualification_rejects_rehashed_weight_changes"):
    source = inspect.getsource(getattr(blueprint.test_blueprint, name)).replace('"v29"', '"general"')
    source = source.replace('decision["reference_role"] = "general"', 'decision["reference_role"] = "v29"')
    source = source.replace('cached["reference_role"] = "general"', 'cached["reference_role"] = "v29"')
    source = source.replace('cached["identity"] = probe.reference_identities()["general"]',
                            'cached["identity"] = probe.reference_identities()["v29"]')
    source = source.replace('rows[0]["reference_role"] = "general"', 'rows[0]["reference_role"] = "v29"')
    source = source.replace('"v8"', '"v29"')
    exec(compile(source, __file__, "exec"), globals())

function = native_tests.test_actual_full_training_identity_binds_native_mean_rule
cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
cloned.__dict__.update(function.__dict__)
globals()[function.__name__] = cloned


def test_full_trial_replays_all_general_native_and_actual_parent_structured_grids(tmp_path, monkeypatch):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    replay_all(state)
    hashes = state.finish()
    report = integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
    assert report["reference_role_counts"] == {"general": 2, "v29": 24}
    assert [role for role, source in calls].count("general") == 2
    assert [role for role, source in calls].count("v29") == 24
    assert report["reference"] == {role: {**identity, "state_tensors": 254, "bitwise_equal": True}
                                   for role, identity in probe.reference_identities().items()}
    assert integration.retention is probe.retention
    for value in vars(integration).values():
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(integration):
            assert unresolved_globals(value) == []
