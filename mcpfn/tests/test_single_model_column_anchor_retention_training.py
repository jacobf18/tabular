import inspect

import test_single_model_anchor_retention_training as previous_tests


SOURCE = inspect.getsource(previous_tests)
for old, new in {
    "single_model_anchor_retention_training": "single_model_column_anchor_retention_training",
    "preflight_single_model_anchor_retention": "preflight_single_model_column_anchor_retention",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
SOURCE = SOURCE.replace('from test_single_model_anchor_retention_preflight import retention_fixture',
    'import test_single_model_anchor_retention_preflight as base_fixtures\n'
    'from types import FunctionType\n'
    'assert base_fixtures.retention_fixture.__closure__ is None\n'
    'retention_fixture = FunctionType(base_fixtures.retention_fixture.__code__, '
    '{**vars(base_fixtures), "probe": probe}, "retention_fixture", base_fixtures.retention_fixture.__defaults__)')
exec(compile(SOURCE, __file__, "exec"), globals())
