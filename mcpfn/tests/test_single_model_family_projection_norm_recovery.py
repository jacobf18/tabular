import pytest
import torch

import audit_single_model_family_projection_recovery as recovery


def test_only_implicit_cpu_fp32_norms_are_promoted_and_restored():
    original = torch.linalg.vector_norm
    values = torch.tensor([1., 1e-4, 1e-8])
    with recovery.precise_cpu_norms() as calls:
        actual = torch.linalg.vector_norm(values)
        explicit = torch.linalg.vector_norm(values, dtype=torch.float32)
        already_double = torch.linalg.vector_norm(values.double())
    assert torch.linalg.vector_norm is original
    assert actual.dtype == already_double.dtype == torch.float64
    assert explicit.dtype == torch.float32
    assert torch.equal(actual, original(values.double()))
    assert calls == [{"elements": 3, "input_dtype": "torch.float32", "accumulation_dtype": "torch.float64"}]


def test_recovery_restores_function_on_failure_and_rejects_changed_output_dtype():
    original = torch.linalg.vector_norm
    with pytest.raises(ValueError, match="explicit output tensor"):
        with recovery.precise_cpu_norms():
            torch.linalg.vector_norm(torch.ones(3), out=torch.empty(()))
    assert torch.linalg.vector_norm is original
    with pytest.raises(RuntimeError, match="fixture"):
        with recovery.precise_cpu_norms():
            raise RuntimeError("fixture")
    assert torch.linalg.vector_norm is original


def prepare(tmp_path, *, clipped=False, recorded=None):
    gradient = torch.tensor([.1, .2, .05]) if not clipped else torch.ones(3)
    path = tmp_path / "optimizer_step.pt"
    torch.save({"combined_gradient": gradient, "applied_gradient": gradient if not clipped else gradient * .1}, path)
    recovery.write_json(tmp_path / "parameter_inventory.json", [["first", [2], 2], ["second", [1], 1]])
    value = float(gradient.double().norm()) if recorded is None else recorded
    recovery.write_json(tmp_path / "result.json", {"optimizer_step_artifact": str(path), "global_gradient_norm_before_clip": value})


def test_precision_diagnosis_keeps_original_tolerances_and_inactive_branch(tmp_path):
    prepare(tmp_path)
    record = recovery.diagnose(tmp_path)
    assert record["unchanged_relative_tolerance"] == 1e-6 and record["unchanged_absolute_tolerance"] == 1e-10
    assert record["applied_gradient_exactly_equals_combined"] and not record["global_clip_active"]
    assert record["cpu_fp64_relative_difference"] < 1e-12


@pytest.mark.parametrize("kind", ["different_norm", "clipped", "changed_applied"])
def test_recovery_cannot_hide_real_gradient_changes(tmp_path, kind):
    prepare(tmp_path, clipped=kind == "clipped", recorded=.4 if kind == "different_norm" else None)
    if kind == "changed_applied":
        path = tmp_path / "optimizer_step.pt"
        state = torch.load(path, weights_only=True)
        state["applied_gradient"] *= .9
        torch.save(state, path)
    with pytest.raises((AssertionError, ValueError)):
        recovery.diagnose(tmp_path)
