import pytest

import audit_single_model_top_three_blocks as audit
import test_single_model_legacy_strata_audit as inherited
import test_single_model_legacy_strata_training as inherited_training


@pytest.mark.parametrize("eligible", [False, True])
def test_every_scheduled_admission_is_independently_replayed(tmp_path, monkeypatch, eligible):
    monkeypatch.setattr(inherited, "audit", audit)
    inherited.test_each_scheduled_admission_is_replayed_including_data_rejection(tmp_path, monkeypatch, eligible)


@pytest.mark.parametrize("mutation", [None, "terminal_as_actual", "model", "record"])
def test_terminal_audit_uses_last_actual_attempt(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "audit", audit)
    inherited.test_terminal_optimizer_audit_uses_last_real_attempt_and_exact_final_state(tmp_path, monkeypatch, mutation)


@pytest.mark.parametrize("mutation", ["mask", "hash", "path", "fabricated_task", "fabricated_metric"])
def test_rejected_draw_evidence_cannot_be_repaired_or_fabricated(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "audit", audit)
    inherited.test_data_noop_audit_rejects_altered_or_fabricated_evidence(tmp_path, monkeypatch, mutation)


@pytest.mark.parametrize("mutation", ["state", "optimizer", "accepted", "signed_zero"])
def test_empty_endpoint_requires_bitwise_initializer(monkeypatch, mutation):
    monkeypatch.setattr(inherited_training, "audit", audit)
    inherited_training.test_all_noop_endpoint_audit_checks_full_initializer_and_optimizer(mutation)


def test_all_expanded_coordinates_and_frozen_tensors_are_checked():
    assert audit.checks is audit.training.paired
    assert audit.checks.SCOPE == "point_only_legacy_strata_other_paired_means"
    assert 'parent_report_path = training.INITIAL_RUN / "validation_25.json"' in audit.ADAPTED_SOURCE
    assert 'len(selected) != 90 or sum(row[2] for row in selected) != 36036497' in audit.ADAPTED_SOURCE
    assert 'frozen["frozen_tensors"] != 164' in audit.ADAPTED_SOURCE
    assert 'top_two_block_training' not in audit.ADAPTED_SOURCE
    assert 'len(decision["comparisons"]) != 22' in audit.ADAPTED_SOURCE
    for name in ("admission_inventory", "replay_admission", "audit_final_endpoint", "load_regenerated_task",
                 "regeneration_evidence", "run_audit"):
        assert getattr(audit, name).__code__ is getattr(audit.template, name).__code__
        assert getattr(audit, name).__globals__ is vars(audit)
