import inspect

import test_single_model_anchor_retention_audit as previous_tests


SOURCE = inspect.getsource(previous_tests)
assert "import audit_single_model_anchor_retention as audit" in SOURCE
SOURCE = SOURCE.replace("import audit_single_model_anchor_retention as audit", "import audit_single_model_column_anchor_retention as audit")
exec(compile(SOURCE, __file__, "exec"), globals())
