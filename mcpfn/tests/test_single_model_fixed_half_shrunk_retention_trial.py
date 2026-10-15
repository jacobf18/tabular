import inspect

import test_single_model_anchor_retention_trial as previous_tests
from single_model_registered_adaptation import replace_once


def fixed_preflight():
    records = original_tests.valid_preflight()
    records[0]["record"].update(proposal_policy=training.paired.POLICY, displacement_fraction=.5)
    records[1]["raw_endpoint"]["fixed_half_step"] = {"fraction": .5, "evaluated": True,
        "parameters": 77998496, "actual_saved_displacements_checked": True}
    return records


SOURCE = inspect.getsource(previous_tests)
for old, new in {
    "train_single_matrixformer_anchor_retention": "train_single_matrixformer_fixed_half_shrunk_retention",
    "881100901": "971100901", "882100901": "972100901",
    "891100901": "981100901", "892100901": "982100901",
    "training.paired is training.predecessor.paired": "training.paired is training.preflight.paired",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
assert SOURCE.count("records = original_tests.valid_preflight()") == 2
SOURCE = SOURCE.replace("records = original_tests.valid_preflight()", "records = fixed_preflight()")
exec(compile(SOURCE, __file__, "exec"), globals())


def test_v39_changes_only_registered_retention_and_displacement_not_qualification():
    assert training.OUTPUT.name == "fixed_half_shrunk_retention_v39"
    assert training.paired is training.fixed_trial.protection
    assert training.paired is not training.previous_trial.paired
    assert training.retention.SPECIFICATION["prior_count"] == 1.
    assert training.retention_training.retention is training.retention is training.preflight.retention
    assert training.capacity.probe is training.preflight
    assert training.TRIAL_QUALIFICATION == training.previous_trial.TRIAL_QUALIFICATION
    assert training.STORAGE_ACCOUNTING["scratch"]["snapshot_vector_allowance"] == 10 * 4 * training.TRAINABLE_PARAMETERS
    assert training.previous_trial.STORAGE_ACCOUNTING["scratch"]["snapshot_vector_allowance"] == 8 * 4 * training.TRAINABLE_PARAMETERS
    assert 'config["displacement_fraction"] != paired.FRACTION' in training.ADAPTED_SOURCE
    assert '"displacement_fraction": paired.FRACTION' in training.REGISTER_SOURCE
    assert 'receipt["canonical_artifacts_checked"] != 6' in training.ADAPTATION_SOURCE
    previous = training.original.alignment.fixed.seeds_in([training.previous_trial.batches(), training.previous_trial.confirmation_cases()])
    current = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    assert len(current) == 3868 and not current.intersection(previous)


@pytest.mark.parametrize("field,value", [("fraction", .25), ("evaluated", False), ("parameters", 2), ("actual_saved_displacements_checked", False)])
def test_full_registration_does_not_accept_legacy_or_incomplete_contraction_evidence(field, value):
    records = fixed_preflight()
    records[-1].pop("accepted")
    records[-1]["retention_audited"] = True
    training.validate_preflight(*records, "digest")
    records[1]["raw_endpoint"]["fixed_half_step"][field] = value
    with pytest.raises(ValueError):
        training.validate_preflight(*records, "digest")
