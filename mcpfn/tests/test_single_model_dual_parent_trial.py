from pathlib import Path

import test_single_model_parent_structured_trial as parent_tests


TEST_SOURCE = Path(parent_tests.__file__).read_text()
for old, new in {
    "train_single_matrixformer_parent_structured": "train_single_matrixformer_dual_parent",
    "parent_structured_v49": "dual_parent_v50",
    "1271100901": "1301100901", "1272100901": "1302100901",
    "1281100901": "1311100901", "1282100901": "1312100901",
    '("v29", "general")': '("v29", "general", "v8")',
    '{"v29", "general"}': '{"v29", "general", "v8"}',
    "require_completed_v48": "require_completed_v49", "closed_v48": "closed_v49",
    "de3f132f725b09c4d8a4225b14e41187f63a668921213b37cb181409ba2b5e49":
        "055bc901667df789527eef698c22085dcb81fbab795577f0c42d055f66f0213c",
    "1922740672": "1475237654",
    '"STORAGE_ACCOUNTING", ': '',
    '"diagnostics", ': '',
    "'canonical_artifacts_checked'] != 5": "'canonical_artifacts_checked'] != 4",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit dual-parent trial test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
TEST_SOURCE = TEST_SOURCE.replace("import train_single_matrixformer_native_column_mean as previous_training",
    "import train_single_matrixformer_parent_structured as previous_training")
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_expanded_component_budget_is_explicit_and_changes_no_other_allowance():
    assert training.diagnostics is training.all_trial.component_storage
    current = deepcopy(training.STORAGE_ACCOUNTING)
    allowance = current["scratch"].pop("dual_parent_component_allowance")
    assert allowance == training.diagnostics.component_bytes(training.batches())
    assert current == previous_training.STORAGE_ACCOUNTING
    assert sum(training.STORAGE_ACCOUNTING["scratch"].values()) <= training.diagnostics.WRITE_BUDGETS["training"]["scratch"]
    assert sum(training.STORAGE_ACCOUNTING["scratch"].values()) > previous_training.diagnostics.WRITE_BUDGETS["training"]["scratch"]
    assert training.diagnostics.RESERVES == previous_training.diagnostics.RESERVES


@pytest.mark.parametrize("mutation", ["drop", "wrong_role_count"])
def test_original_probe_requirement_cannot_omit_dual_reference_components(monkeypatch, mutation):
    probe = training.preflight
    documents = {name: deepcopy(probe.read(probe.OUTPUT / name)) for name in
        ("result.json", "audit.json", "gate.json", "retention_audit.json", "config.json", "status.json")}
    monkeypatch.setattr(probe, "verify", lambda output: None)
    monkeypatch.setattr(probe, "require_completed_v49", lambda: {})
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "read", lambda path: documents[path.name])
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: None)
    if mutation == "drop":
        documents["retention_audit.json"]["component_grids"] = 49
    else:
        documents["retention_audit.json"]["component_role_counts"]["v8"] = 23
    with pytest.raises(ValueError):
        training.all_trial.require_probe(probe, lambda *arguments: None)
