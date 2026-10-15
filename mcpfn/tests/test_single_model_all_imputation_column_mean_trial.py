import test_single_model_native_parent_trial as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    "train_single_matrixformer_native_parent": "train_single_matrixformer_all_imputation_column_mean",
    "train_single_matrixformer_dual_parent as previous_training": "train_single_matrixformer_native_parent as previous_training",
    "native_parent_v51": "all_imputation_column_mean_v52",
    "test_only_structured_reference_changes_with_fresh_full_trial_and_confirmation_seeds":
        "test_only_structured_teacher_mean_changes_with_fresh_full_trial_and_confirmation_seeds",
    "1331100901": "1361100901", "1332100901": "1362100901",
    "1341100901": "1371100901", "1342100901": "1372100901",
    "require_completed_v50": "require_completed_v51", "closed_v50": "closed_v51",
    '"9d9379e6a38f1ac661843b5c076059b4e2a7e50a5c63d5c4cf5aa6f02cd9983e"':
        'training.capacity.sha256(training.capacity.OUTPUT / "external_recheck.json")',
    "1517751498": "1248288852",
    "'canonical_artifacts_checked'] != 8": "'canonical_artifacts_checked'] != 2",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit all-imputation trial test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_all_stage_budgets_and_original_reference_policy_remain_unchanged():
    assert training.diagnostics is training.all_trial.component_storage
    assert training.STORAGE_ACCOUNTING == previous_training.STORAGE_ACCOUNTING
    assert training.diagnostics.WRITE_BUDGETS == previous_training.diagnostics.WRITE_BUDGETS
    assert training.diagnostics.RESERVES == previous_training.diagnostics.RESERVES
    assert training.retention.SPECIFICATION == previous_training.retention.SPECIFICATION
    assert training.retention.REFERENCE_SHA256S == previous_training.retention.REFERENCE_SHA256S
    assert training.retention.MEAN_SPECIFICATION != previous_training.retention.MEAN_SPECIFICATION
    assert training.retention.MEAN_SPECIFICATION["scope"] == "every original native and structured imputation query; no ratings"
    assert sum(training.STORAGE_ACCOUNTING["scratch"].values()) <= 48 * 1024**3


@pytest.mark.parametrize("mutation", ["drop", "wrong_v8", "wrong_v29", "general", "native", "structured_mean", "all_mean"])
def test_probe_requirement_rejects_missing_reference_or_mean_components(monkeypatch, mutation):
    probe = training.preflight
    documents = {name: deepcopy(probe.read(probe.OUTPUT / name)) for name in
        ("result.json", "audit.json", "gate.json", "retention_audit.json", "config.json", "status.json")}
    mean_directory = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")
    mean_documents = {name: deepcopy(probe.read(mean_directory / name)) for name in ("audit.json", "config.json")}
    monkeypatch.setattr(probe, "verify", lambda output: None)
    monkeypatch.setattr(probe, "require_completed_v51", lambda: {})
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "read", lambda path: (mean_documents if path.parent == mean_directory else documents)[path.name])
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: None)
    report = documents["retention_audit.json"]
    if mutation == "drop":
        report["component_grids"] = 49
    elif mutation == "general":
        report["component_role_counts"] = {"general": 2, "v8": 24, "v29": 24}
    elif mutation == "native":
        report["reference_role_counts"] = {"general": 2, "structured_mean": 24}
    elif mutation in ("structured_mean", "all_mean"):
        mean_documents["audit.json"]["structured_reweighted_tasks" if mutation == "structured_mean" else "imputation_reweighted_tasks"] -= 1
    else:
        report["component_role_counts"][mutation.removeprefix("wrong_")] -= 1
    with pytest.raises(ValueError):
        training.all_trial.require_probe(probe, lambda *arguments: None)
