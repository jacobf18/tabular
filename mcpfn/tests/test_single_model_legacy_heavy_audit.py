"""Exercise the V30 independent admission and actual-endpoint auditing paths."""

import pytest

import audit_single_model_legacy_heavy as audit
import test_single_model_legacy_strata_audit as inherited
import test_single_model_legacy_strata_training as inherited_training


@pytest.mark.parametrize("eligible", [False, True])
def test_v30_independently_replays_every_scheduled_admission(tmp_path, monkeypatch, eligible):
    monkeypatch.setattr(inherited, "audit", audit)
    inherited.test_each_scheduled_admission_is_replayed_including_data_rejection(tmp_path, monkeypatch, eligible)


@pytest.mark.parametrize("mutation", [None, "terminal_as_actual", "model", "record"])
def test_v30_terminal_audit_uses_exact_last_optimizer_attempt(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "audit", audit)
    inherited.test_terminal_optimizer_audit_uses_last_real_attempt_and_exact_final_state(tmp_path, monkeypatch, mutation)


@pytest.mark.parametrize("mutation", ["mask", "hash", "path", "fabricated_task", "fabricated_metric"])
def test_v30_rejected_draws_cannot_be_repaired_or_have_fabricated_metrics(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "audit", audit)
    inherited.test_data_noop_audit_rejects_altered_or_fabricated_evidence(tmp_path, monkeypatch, mutation)


@pytest.mark.parametrize("mutation", ["state", "optimizer", "accepted", "signed_zero"])
def test_v30_empty_endpoint_requires_bitwise_original_initializer(monkeypatch, mutation):
    monkeypatch.setattr(inherited_training, "audit", audit)
    inherited_training.test_all_noop_endpoint_audit_checks_full_initializer_and_optimizer(mutation)


def test_parent_development_reference_uses_only_audited_v29_step25():
    assert 'parent_report_path = training.INITIAL_RUN / "validation_25.json"' in audit.ADAPTED_SOURCE
    assert 'validation_10.json' not in audit.ADAPTED_SOURCE
    for name in ("admission_inventory", "replay_admission", "audit_final_endpoint", "load_regenerated_task",
                 "regeneration_evidence", "run_audit"):
        function = getattr(audit, name)
        assert function.__code__ is getattr(audit.template, name).__code__
        assert function.__globals__ is vars(audit)
