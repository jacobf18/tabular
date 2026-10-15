from copy import deepcopy

import pytest

import single_model_full_network_storage as diagnostics
import test_single_model_scratch_diagnostics as inherited


@pytest.fixture
def planned(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "diagnostics", diagnostics)
    return inherited.planned.__wrapped__(tmp_path, monkeypatch)


def test_only_new_diagnostics_move_and_ordinary_checkpoints_stay(planned):
    inherited.test_only_new_diagnostics_move_and_ordinary_checkpoints_stay(planned)


@pytest.mark.parametrize("kind", ["file", "directory", "broken_link", "target"])
def test_existing_evidence_is_never_replaced(planned, kind):
    inherited.test_existing_evidence_is_never_replaced(planned, kind)


def test_scope_targets_budgets_and_one_shot_boundaries_remain_strict(planned):
    inherited.test_changed_scope_or_target_or_budgets_are_rejected(planned)


def test_prepared_validation_remains_one_shot(planned):
    inherited.test_prepared_validation_directory_is_still_one_shot(planned)


@pytest.mark.parametrize("stage", list(diagnostics.WRITE_BUDGETS))
@pytest.mark.parametrize("volume", list(diagnostics.RESERVES))
def test_actual_volume_admission_keeps_each_original_reserve(monkeypatch, stage, volume):
    monkeypatch.setattr(inherited, "diagnostics", diagnostics)
    inherited.test_each_stage_accounts_for_its_actual_volume_without_spending_reserves(monkeypatch, stage, volume)


def test_only_new_training_allowances_expand_and_previous_module_is_unchanged(planned):
    assert diagnostics.RESERVES == diagnostics.template.RESERVES
    assert diagnostics.RESERVES is not diagnostics.template.RESERVES
    assert diagnostics.template.WRITE_BUDGETS["training"] == {
        "checkpoint": 1536 * 1024**2, "root": 16 * 1024**2, "scratch": 16384 * 1024**2}
    assert diagnostics.WRITE_BUDGETS["training"] == {
        "checkpoint": 2304 * 1024**2, "root": 16 * 1024**2, "scratch": 32768 * 1024**2}
    for stage in diagnostics.WRITE_BUDGETS:
        if stage != "training":
            assert diagnostics.WRITE_BUDGETS[stage] == diagnostics.template.WRITE_BUDGETS[stage]
    changed = deepcopy(planned)
    changed["stage_write_budget_bytes"] = diagnostics.template.WRITE_BUDGETS
    with pytest.raises(ValueError):
        diagnostics.require_plan(changed, "training")


def test_cloned_storage_functions_resolve_only_the_new_namespace():
    for name in ("layout", "require_plan", "require_filesystems", "prepare", "verify", "require_empty_validation", "require_capacity"):
        assert getattr(diagnostics, name).__globals__ is vars(diagnostics)
        assert getattr(diagnostics, name).__code__ is getattr(diagnostics.template, name).__code__
