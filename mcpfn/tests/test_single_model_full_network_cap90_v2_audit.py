import test_single_model_full_network_cap90_audit as predecessor


SOURCE = predecessor.SOURCE.replace("import audit_single_model_full_network_cap90 as audit",
    "import audit_single_model_full_network_cap90_v2 as audit")
exec(compile(SOURCE, __file__, "exec"), globals())
