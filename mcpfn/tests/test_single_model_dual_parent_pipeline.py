from pathlib import Path

import test_single_model_parent_structured_pipeline as parent_tests


TEST_SOURCE = Path(parent_tests.__file__).read_text()
for old, new in {
    "run_single_model_parent_structured_pipeline": "run_single_model_dual_parent_pipeline",
    "audit_single_model_parent_structured": "audit_single_model_dual_parent",
    "train_single_matrixformer_parent_structured": "train_single_matrixformer_dual_parent",
    "parent_structured_v49": "dual_parent_v50",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit dual-parent pipeline test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_pipeline_requires_complete_component_evidence_not_only_combined_grids():
    assert pipeline.retention_training is pipeline.training_audit.retention_training
    assert "require_report_components" in pipeline.retention_training.require_report.__code__.co_names
    assert pipeline.training.CAPACITY_RECEIPT_SHA256 == "055bc901667df789527eef698c22085dcb81fbab795577f0c42d055f66f0213c"
