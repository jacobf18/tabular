from pathlib import Path


TEST_SOURCE = Path(__file__).with_name("test_single_model_native_column_initializer_diagnosis.py").read_text()
for old, new in {
    "diagnose_single_model_native_column_initializer": "diagnose_single_model_balanced_native_quarter_initializer",
    "native_column_mean_v48": "balanced_native_quarter_v55",
    "bound_to_v48": "bound_to_v55",
    "V48": "V55",
    '"selected_step": 10': '"selected_step": 64',
    '"accepted_updates": 56': '"accepted_updates": 40',
    '(0, "selected_step", 64)': '(0, "selected_step", 50)',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit V55 initializer test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())
