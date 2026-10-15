import inspect
import types

import pytest

import test_single_model_general_native_retention_training as blueprint
import preflight_single_model_native_column_mean as probe


base_fixtures = types.ModuleType("native_column_test_fixtures")
base_fixtures.__dict__.update({**vars(blueprint.base_fixtures), "probe": probe})
SOURCE = blueprint.SOURCE.replace("single_model_general_native_retention_training", "single_model_native_column_mean_training")
SOURCE = SOURCE.replace("preflight_single_model_general_native_retention", "preflight_single_model_native_column_mean")
SOURCE = SOURCE.replace("config.update(batches=[batch],", "config.update(native_teacher_mean=integration.retention.MEAN_SPECIFICATION, batches=[batch],")
exec(compile(SOURCE, __file__, "exec"), globals())
for name in ("test_both_original_references_must_be_bound_in_full_trial_registration",
        "test_full_replay_never_swaps_or_mutates_references", "test_qualification_requires_cohort_bindings_as_well_as_full_grid_hashes",
        "test_full_replay_rejects_changed_retention_weights", "test_qualification_rejects_rehashed_weight_changes",
        "test_full_trial_replays_exact_general_native_and_v8_structured_grids"):
    function = getattr(blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


@pytest.mark.parametrize("mutation", ["missing", "weight", "components", "coverage"])
def test_actual_full_training_identity_binds_native_mean_rule(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    integration.identity(config)
    config["native_teacher_mean"] = deepcopy(config["native_teacher_mean"])
    if mutation == "missing":
        config.pop("native_teacher_mean")
    elif mutation == "weight":
        config["native_teacher_mean"]["teacher_mean_huber_weight"] = 2.
    elif mutation == "components":
        config["native_teacher_mean"]["native_components"] = [6]
    else:
        config["native_teacher_mean"]["coverage"] = "omit singleton columns"
    with pytest.raises((ValueError, KeyError)):
        integration.identity(config)
