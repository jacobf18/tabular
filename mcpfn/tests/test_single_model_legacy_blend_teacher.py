import ast
from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import pytest
import torch

import single_model_legacy_blend_teacher as teacher
import single_model_distillation as distillation
import train_single_matrixformer_parent_teacher as original
from test_single_model_general_imputation_teacher import fixture


class ObservedOnlyEpisode:
    def __init__(self, source, metadata):
        self.source = source
        self.metadata = metadata

    @property
    def target(self):
        raise AssertionError("Teacher blending must not access hidden query truth")


@pytest.mark.parametrize("domain,structured,role", [
    ("imputation", False, "observed_blend"), ("imputation", True, "general"),
    ("recsys", False, "specialist")])
def test_actual_callbacks_frozen_identities_and_observed_only_targets(tmp_path, monkeypatch, domain, structured, role):
    models, identities = fixture(tmp_path)
    source = torch.tensor([[[1., float("nan")], [2., -1.], [float("nan"), 4.]]])
    center = torch.tensor([[[1., 30.], [2., -1.], [50., 4.]]])
    outputs = {id(models[1]): center - .5, id(models[2]): center + .5}
    calls = []
    options = {"axis_batch_size": 7}

    def predict(model, observed, criterion, **actual_options):
        assert observed is source and actual_options == options and not torch.is_grad_enabled()
        calls.append(model)
        return outputs[id(model)].clone()

    monkeypatch.setattr(original, "streamed_point_predictions", predict)
    monkeypatch.setattr(distillation, "streamed_point_predictions", predict)
    monkeypatch.setattr(original, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(distillation, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())
    episode = ObservedOnlyEpisode(source, {"domain": domain, "structured": structured, "gradient_data": "synthetic"})
    previous, reference = teacher.mixture.parent_teacher_target, teacher.mixture.reference_teacher_target
    with teacher.runtime({}, tmp_path, identities):
        prediction, routing = teacher.mixture.parent_teacher_target(models[0], models[1:], None, episode, options)
        assert routing["target"] == role and routing["policy"] == teacher.POLICY
        assert not prediction.requires_grad and teacher.mixture.reference_teacher_target is reference
        if role == "observed_blend":
            assert calls == [models[2], models[1]]
            assert torch.equal(prediction, center)
            assert routing["specialist_weight"] == pytest.approx(.5)
            assert routing["evidence_cells"] == 4
            assert routing["teacher_identity"] == {name: identities[name] for name in ("general", "specialist")}
        else:
            chosen = models[1] if role == "general" else models[2]
            assert calls == [chosen] and torch.equal(prediction, outputs[id(chosen)])
            assert routing["teacher_identity"] == identities[role]
    assert teacher.mixture.parent_teacher_target is previous
    assert teacher.mixture.reference_teacher_target is reference


@pytest.mark.parametrize("domain,structured", [("imputation", False), ("imputation", True), ("recsys", False)])
@pytest.mark.parametrize("role", ["general", "specialist"])
def test_both_original_teachers_must_match_their_checkpoint_before_any_prediction(tmp_path, monkeypatch, domain, structured, role):
    models, identities = fixture(tmp_path)
    models[1 if role == "general" else 2].weight.add_(1)
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())
    monkeypatch.setattr(teacher, "observed_blend_target", lambda *args: pytest.fail("Unverified teacher reached inference"))
    monkeypatch.setattr(teacher, "parent_teacher_target", lambda *args: pytest.fail("Unverified teacher reached inference"))
    episode = SimpleNamespace(metadata={"domain": domain, "structured": structured, "gradient_data": "synthetic"})
    previous = teacher.mixture.parent_teacher_target
    with pytest.raises(ValueError, match="Actual frozen teacher"):
        with teacher.runtime({}, tmp_path, identities):
            teacher.mixture.parent_teacher_target(models[0], models[1:], None, episode)
    assert teacher.mixture.parent_teacher_target is previous


@pytest.mark.parametrize("metadata", [
    {"gradient_data": "real", "domain": "imputation"},
    {"gradient_data": "synthetic", "domain": "unknown"},
    {"gradient_data": "synthetic", "domain": "recsys", "structured": True},
    {"gradient_data": "synthetic", "domain": "imputation", "structured": "yes"}])
def test_unsupported_metadata_is_rejected(metadata):
    with pytest.raises(ValueError):
        teacher.role_for(metadata)


def test_legacy_role_never_selects_components_or_uses_errors():
    for component in range(8):
        assert teacher.role_for({"domain": "imputation", "gradient_data": "synthetic", "component": component,
            "queries": 1, "target_std": 0., "validation_error": float("inf")}) == "observed_blend"


def test_verified_model_cannot_be_replaced_mid_context(tmp_path, monkeypatch):
    models, identities = fixture(tmp_path)
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())
    monkeypatch.setattr(teacher, "observed_blend_target", lambda *args: (torch.ones(1), {}))
    episode = SimpleNamespace(metadata={"domain": "imputation", "gradient_data": "synthetic"})
    with teacher.runtime({}, tmp_path, identities):
        teacher.mixture.parent_teacher_target(models[0], models[1:], None, episode)
        replacement = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        replacement.load_state_dict(models[1].state_dict())
        with pytest.raises(ValueError, match="replace"):
            teacher.mixture.parent_teacher_target(models[0], [replacement, models[2]], None, episode)


def test_blend_failure_restores_training_hook_without_changing_evaluation(tmp_path, monkeypatch):
    models, identities = fixture(tmp_path)
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())

    def fail(*arguments):
        raise RuntimeError("blend failure")

    monkeypatch.setattr(teacher, "observed_blend_target", fail)
    previous, reference = teacher.mixture.parent_teacher_target, teacher.mixture.reference_teacher_target
    with pytest.raises(RuntimeError, match="blend failure"):
        with teacher.runtime({}, tmp_path, identities):
            episode = SimpleNamespace(metadata={"domain": "imputation", "gradient_data": "synthetic"})
            teacher.mixture.parent_teacher_target(models[0], models[1:], None, episode)
    assert teacher.mixture.parent_teacher_target is previous
    assert teacher.mixture.reference_teacher_target is reference


def test_scope_validation_and_checkpoint_checks_remain_identical_and_old_policy_is_unchanged():
    assert teacher.POLICY == {"legacy": "observed_blend", "structured": "general", "recsys": "specialist"}
    assert teacher.predecessor.POLICY == {"legacy": "general", "structured": "general", "recsys": "specialist"}
    before, after = ast.parse(teacher.predecessor.ADAPTED_SOURCE), ast.parse(teacher.ADAPTED_SOURCE)
    for name in ("role_for", "identities", "verify_model"):
        before_node = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == name)
        after_node = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(before_node) == ast.dump(after_node)
        assert inspect.unwrap(getattr(teacher, name)).__globals__ is vars(teacher)
    assert teacher.observed_blend_target is distillation.teacher_target
