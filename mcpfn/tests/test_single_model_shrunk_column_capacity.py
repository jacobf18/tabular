import builtins
import inspect

import pytest

import relocate_single_model_shrunk_column_capacity as relocation
import test_single_model_column_anchor_capacity as column_tests
import test_single_model_full_network_capacity as inherited
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize("mutation", [None, "live", "updates", "attempts", "config", "qualified", "preflight", "scope", "optimizer", "selection"])
def test_v38_tail_requires_its_exact_complete_original_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "relocation", relocation)
    inherited.test_each_tail_requires_its_exact_original_terminal_identity(tmp_path, monkeypatch, "unselected_tail", mutation)


@pytest.mark.parametrize("count", [4, 6])
def test_vector_copy_and_publication_preserve_every_original_byte(tmp_path, monkeypatch, count):
    monkeypatch.setattr(inherited.inherited, "relocation", relocation)
    inherited.inherited.test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count)


def test_exact_scope_keeps_new_probe_all_selections_and_teachers_untouched():
    assert set(relocation.OWNERS) == {"column_anchor_retention_v38", "column_anchor_retention_preflight_v1",
        "replicated_decoder_preflight_v1", "structured_teacher_preflight_v1", "point_inclusive_preflight_v1"}
    assert relocation.OWNERS["column_anchor_retention_v38"] == {"kind": "unselected_tail", "run": "column_anchor_retention_v38",
        "step": 64, "updates": 59, "selected": 0, "grids": 1368, "view_flag": "full_network_training"}
    assert relocation.OWNERS["column_anchor_retention_preflight_v1"] == {"kind": "vector", "parameters": 77998496, "frozen": 16}
    for name in ("replicated_decoder_preflight_v1", "structured_teacher_preflight_v1", "point_inclusive_preflight_v1"):
        assert relocation.OWNERS[name] == {"kind": "vector", "parameters": 11227016, "frozen": 248}
    assert relocation.probe.OUTPUT.name == "shrunk_column_anchor_retention_preflight_v1"
    assert relocation.probe.OUTPUT.name not in relocation.OWNERS
    assert relocation.TRAINING_ADMISSION_BYTES == 21743271936
    assert 'len(records) != 5' in relocation.MAIN_SOURCE
    assert '"canonical_artifacts_checked": 5' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index("copying.copy_and_verify(row)") < relocation.MAIN_SOURCE.index("copying.publish(row)")
    assert relocation.copying is not relocation.previous.copying
    assert relocation.previous.copying.copy_and_verify is not relocation.copy_and_verify


def test_generated_functions_keep_only_new_module_globals_without_closures():
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert function.__closure__ is None, name
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
    for name in ("inventory_original", "checkpoint_identity", "require_idle", "copy_and_verify", "require_probe", "main"):
        assert getattr(relocation, name).__globals__ is vars(relocation)


@pytest.mark.parametrize("mutation", [None, "rejected", "coverage", "specification", "nonfinite", "incomplete", "gpu_audit"])
def test_requires_the_complete_accepted_new_probe_and_every_actual_reference(monkeypatch, mutation):
    monkeypatch.setattr(column_tests, "relocation", relocation)
    column_tests.test_new_probe_is_required_with_dynamic_complete_reference_coverage(monkeypatch, mutation)
