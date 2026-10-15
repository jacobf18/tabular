import inspect

import test_single_model_anchor_retention_audit as previous_tests


SOURCE = inspect.getsource(previous_tests)
assert "import audit_single_model_anchor_retention as audit" in SOURCE
SOURCE = SOURCE.replace("import audit_single_model_anchor_retention as audit", "import audit_single_model_fixed_half_shrunk_retention as audit")
exec(compile(SOURCE, __file__, "exec"), globals())


def test_all_admitted_records_and_terminal_optimizer_use_fixed_half_auditor():
    assert audit.checks is audit.training.paired is audit.fixed_trial.protection
    assert 'checks.check_record(row,' in audit.ADAPTED_SOURCE
    assert audit.audit_final_endpoint.__globals__["checks"] is audit.fixed_trial.protection
    assert '"fixed_half_training": True' in audit.ADAPTED_SOURCE
    assert '"displacement_fraction": training.paired.FRACTION' in audit.ADAPTED_SOURCE
    assert '"complete_training_audit": complete_trial' in audit.ADAPTED_SOURCE
