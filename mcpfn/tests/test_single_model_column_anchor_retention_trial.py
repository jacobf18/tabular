import inspect

import test_single_model_anchor_retention_trial as previous_tests


SOURCE = inspect.getsource(previous_tests)
for old, new in {
    "train_single_matrixformer_anchor_retention": "train_single_matrixformer_column_anchor_retention",
    "881100901": "911100901", "882100901": "912100901",
    "891100901": "921100901", "892100901": "922100901",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_new_trial_uses_column_normalization_and_disjoint_previous_trial_seeds():
    assert training.retention.SPECIFICATION["variance_floor"] == 1e-6
    assert training.retention_training.retention is training.retention
    assert training.preflight.retention is training.retention
    assert training.capacity.probe is training.preflight
    assert training.OUTPUT.name == "column_anchor_retention_v38"
    previous = training.original.alignment.fixed.seeds_in(
        [training.previous_trial.batches(), training.previous_trial.confirmation_cases()])
    current = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    assert len(current) == 3868 and not previous.intersection(current)
