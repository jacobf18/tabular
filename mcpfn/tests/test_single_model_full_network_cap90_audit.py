import inspect

import test_single_model_full_network_audit as template


SOURCE = inspect.getsource(template).replace("import audit_single_model_full_network as audit",
    "import audit_single_model_full_network_cap90 as audit")
exec(compile(SOURCE, __file__, "exec"), globals())
