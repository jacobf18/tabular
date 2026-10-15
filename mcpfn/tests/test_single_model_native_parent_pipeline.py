import test_single_model_dual_parent_pipeline as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    "run_single_model_dual_parent_pipeline": "run_single_model_native_parent_pipeline",
    "audit_single_model_dual_parent": "audit_single_model_native_parent",
    "train_single_matrixformer_dual_parent": "train_single_matrixformer_native_parent",
    "dual_parent_v50": "native_parent_v51",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit native-parent pipeline test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_pipeline_requires_new_capacity_receipt_and_all_reference_components():
    assert pipeline.retention_training is pipeline.training_audit.retention_training
    assert "require_report_components" in pipeline.retention_training.require_report.__code__.co_names
    assert pipeline.training.CAPACITY_RECEIPT_SHA256 == "9d9379e6a38f1ac661843b5c076059b4e2a7e50a5c63d5c4cf5aa6f02cd9983e"
