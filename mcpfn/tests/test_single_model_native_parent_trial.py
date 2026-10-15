import test_single_model_dual_parent_trial as parent_tests


TEST_SOURCE = parent_tests.TEST_SOURCE
for old, new in {
    "train_single_matrixformer_dual_parent": "train_single_matrixformer_native_parent",
    "train_single_matrixformer_parent_structured as previous_training": "train_single_matrixformer_dual_parent as previous_training",
    "dual_parent_v50": "native_parent_v51",
    "1301100901": "1331100901", "1302100901": "1332100901",
    "1311100901": "1341100901", "1312100901": "1342100901",
    "require_completed_v49": "require_completed_v50", "closed_v49": "closed_v50",
    "055bc901667df789527eef698c22085dcb81fbab795577f0c42d055f66f0213c":
        "9d9379e6a38f1ac661843b5c076059b4e2a7e50a5c63d5c4cf5aa6f02cd9983e",
    "1475237654": "1517751498",
    "'canonical_artifacts_checked'] != 4": "'canonical_artifacts_checked'] != 8",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit native-parent trial test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_complete_component_budget_matches_previous_trial_without_another_increase():
    assert training.diagnostics is training.all_trial.component_storage
    assert training.STORAGE_ACCOUNTING == previous_training.STORAGE_ACCOUNTING
    assert training.diagnostics.WRITE_BUDGETS == previous_training.diagnostics.WRITE_BUDGETS
    assert training.diagnostics.RESERVES == previous_training.diagnostics.RESERVES
    allowance = training.STORAGE_ACCOUNTING["scratch"]["dual_parent_component_allowance"]
    assert allowance == training.diagnostics.component_bytes(training.batches())
    assert sum(training.STORAGE_ACCOUNTING["scratch"].values()) <= 48 * 1024**3


@pytest.mark.parametrize("mutation", ["drop", "wrong_v8", "wrong_v29", "general", "native"])
def test_probe_requirement_rejects_incomplete_or_old_reference_components(monkeypatch, mutation):
    probe = training.preflight
    documents = {name: deepcopy(probe.read(probe.OUTPUT / name)) for name in
        ("result.json", "audit.json", "gate.json", "retention_audit.json", "config.json", "status.json")}
    monkeypatch.setattr(probe, "verify", lambda output: None)
    monkeypatch.setattr(probe, "require_completed_v50", lambda: {})
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "read", lambda path: documents[path.name])
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: None)
    report = documents["retention_audit.json"]
    if mutation == "drop":
        report["component_grids"] = 49
    elif mutation == "general":
        report["component_role_counts"] = {"general": 2, "v8": 24, "v29": 24}
    elif mutation == "native":
        report["reference_role_counts"] = {"general": 2, "structured_mean": 24}
    else:
        report["component_role_counts"][mutation.removeprefix("wrong_")] -= 1
    with pytest.raises(ValueError):
        training.all_trial.require_probe(probe, lambda *arguments: None)
