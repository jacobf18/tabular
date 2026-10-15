from pathlib import Path

from single_model_registered_adaptation import replace_once


TEST_SOURCE = Path(__file__).with_name("test_single_model_native_column_initializer_diagnosis.py").read_text()
for old, new in {
    "diagnose_single_model_native_column_initializer": "diagnose_single_model_parent_structured_initializer",
    "native_column_mean_v48": "parent_structured_v49",
    "bound_to_v48": "bound_to_v49",
    "V48": "V49",
    '"selected_step": 10': '"selected_step": 25',
    '"accepted_updates": 56': '"accepted_updates": 62',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit V49 initializer test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
TEST_SOURCE = replace_once(TEST_SOURCE,
    '(4, "selected_step", 25)', '(4, "selected_step", 10)')
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_adaptation_binds_both_source_blueprints_and_preserves_science():
    assert "Path(diagnostic_blueprint.__file__).resolve()" in diagnosis.DIAGNOSTIC_SOURCE
    assert 'REPO / "tests/test_single_model_native_column_initializer_diagnosis.py"' in diagnosis.DIAGNOSTIC_SOURCE
    assert diagnosis.SELECTED_SHA256 == "6b1942ab8e84a47bea23c52fac515c4b827e5678ebe1a11566e980b7af9dc990"
    assert diagnosis.CLOSURE_SHA256 == "6d3ab8f6b9562d8cdf7e93d41c71d804f277ffa9e4c7adc1ce1eeb601ae595e6"
    assert "V49" in diagnosis.POLICY and "two failed comparisons" in diagnosis.POLICY
    assert "V48" not in diagnosis.POLICY
    assert diagnosis.SCRATCH.parent == diagnosis.training.SCRATCH
