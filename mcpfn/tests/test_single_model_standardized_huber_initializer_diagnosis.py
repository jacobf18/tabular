from pathlib import Path


TEST_SOURCE = Path(__file__).with_name("test_single_model_native_column_initializer_diagnosis.py").read_text()
for old, new in {
    "diagnose_single_model_native_column_initializer": "diagnose_single_model_standardized_huber_initializer",
    "native_column_mean_v48": "standardized_teacher_huber_v54",
    "bound_to_v48": "bound_to_v54",
    "V48": "V54",
    '"selected_step": 10': '"selected_step": 50',
    '"accepted_updates": 56': '"accepted_updates": 57',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit V54 initializer test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_adaptation_binds_both_source_blueprints_and_preserves_science():
    assert "Path(diagnostic_blueprint.__file__).resolve()" in diagnosis.DIAGNOSTIC_SOURCE
    assert 'REPO / "tests/test_single_model_native_column_initializer_diagnosis.py"' in diagnosis.DIAGNOSTIC_SOURCE
    assert diagnosis.SELECTED_SHA256 == "3693644e3cba0407cc69cf52581f671f5b9534a6a85583bedf5fd9789868fcd1"
    assert diagnosis.CLOSURE_SHA256 == "d327dee4de4a35b7d6f609509754724bf685b99ff9693a4e33f51c6e5cbcfd9a"
    assert "V54" in diagnosis.POLICY and "seven failed comparisons" in diagnosis.POLICY
    assert "V48" not in diagnosis.POLICY
    assert diagnosis.SCRATCH.parent == diagnosis.training.SCRATCH
