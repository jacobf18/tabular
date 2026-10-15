import inspect
import types

import test_single_model_uniform_v8_retention_training as test_blueprint
import test_single_model_uniform_v8_retention_preflight as fixture_blueprint
import preflight_single_model_general_native_retention as probe


base_fixtures = types.ModuleType("native_general_test_fixtures")
base_fixtures.__dict__.update({**vars(fixture_blueprint), "probe": probe})
exec(compile(fixture_blueprint.FIXTURE_SOURCE, __file__, "exec"), vars(base_fixtures))
SOURCE = test_blueprint.SOURCE.replace("uniform_v8_retention", "general_native_retention")
SOURCE = SOURCE.replace("import test_single_model_general_native_retention_preflight as base_fixtures", "")
SOURCE = SOURCE.replace('("v8", "v29")', '("v8", "general")')
SOURCE = SOURCE.replace('input_sha256={identity["checkpoint"]: identity["sha256"] for identity in probe.reference_identities().values()},',
    'input_sha256={str(probe.PARENT): probe.PARENT_SHA256, '
    '**{identity["checkpoint"]: identity["sha256"] for identity in probe.reference_identities().values()}},')
exec(compile(SOURCE, __file__, "exec"), globals())

for name in ("test_both_original_references_must_be_bound_in_full_trial_registration",
        "test_full_replay_never_swaps_or_mutates_references",
        "test_qualification_requires_cohort_bindings_as_well_as_full_grid_hashes",
        "test_full_replay_rejects_changed_retention_weights", "test_qualification_rejects_rehashed_weight_changes"):
    function = getattr(test_blueprint, name)
    adapted = inspect.getsource(function).replace('"v29"', '"general"')
    adapted = adapted.replace('decision["reference_role"] = "general"', 'decision["reference_role"] = "v8"')
    adapted = adapted.replace('cached["reference_role"] = "general"', 'cached["reference_role"] = "v8"')
    adapted = adapted.replace('cached["identity"] = probe.reference_identities()["general"]',
                              'cached["identity"] = probe.reference_identities()["v8"]')
    adapted = adapted.replace('rows[0]["reference_role"] = "general"', 'rows[0]["reference_role"] = "v8"')
    exec(compile(adapted, __file__, "exec"), globals())


def test_full_trial_replays_exact_general_native_and_v8_structured_grids(tmp_path, monkeypatch):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    replay_all(state)
    hashes = state.finish()
    report = integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
    assert report["reference_role_counts"] == {"general": 2, "v8": 24}
    assert [role for role, source in calls].count("general") == 2
    assert [role for role, source in calls].count("v8") == 24
    assert report["reference"] == {role: {**identity, "state_tensors": 254, "bitwise_equal": True}
                                   for role, identity in probe.reference_identities().items()}
    assert integration.retention is probe.retention
    for name in ("identity", "cohort_identity", "original_identity", "verify_references"):
        assert getattr(integration, name).__globals__ is vars(integration)
