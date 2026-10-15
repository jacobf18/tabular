import ast
from contextlib import nullcontext
import hashlib
import inspect
from types import FunctionType

import pytest
import torch

import single_model_all_imputation_blend_teacher as teacher
import single_model_distillation as distillation
import single_model_legacy_blend_teacher as legacy
import test_single_model_legacy_blend_teacher as legacy_test_blueprint
import train_single_matrixformer_parent_teacher as original
from single_model_structured_mixture import COMPONENTS, FAMILIES
from test_single_model_general_imputation_teacher import fixture
from test_single_model_legacy_blend_teacher import ObservedOnlyEpisode


def inherit_test(name):
    blueprint = getattr(legacy_test_blueprint, name)
    namespace = {**blueprint.__globals__, "teacher": teacher}
    inherited = FunctionType(blueprint.__code__, namespace, name, blueprint.__defaults__, blueprint.__closure__)
    inherited.__kwdefaults__ = blueprint.__kwdefaults__
    inherited.__dict__.update(blueprint.__dict__)
    return inherited


for inherited_name in (
    "test_both_original_teachers_must_match_their_checkpoint_before_any_prediction",
    "test_unsupported_metadata_is_rejected",
    "test_legacy_role_never_selects_components_or_uses_errors",
    "test_verified_model_cannot_be_replaced_mid_context",
    "test_blend_failure_restores_training_hook_without_changing_evaluation",
):
    globals()[inherited_name] = inherit_test(inherited_name)


CASES = [("imputation", False, None, component) for component in range(8)]
CASES += [("imputation", True, family, component) for family in FAMILIES for component in COMPONENTS]
CASES += [("recsys", False, None, None)]


@pytest.mark.parametrize("domain,structured,family,component", CASES)
def test_all_original_callbacks_are_observed_only_and_preserve_evaluation(
        tmp_path, monkeypatch, domain, structured, family, component):
    models, identities = fixture(tmp_path)
    source = torch.tensor([[[1., float("nan")], [2., -1.], [float("nan"), 4.]]])
    center = torch.tensor([[[1., 30.], [2., -1.], [50., 4.]]])
    outputs = {id(models[1]): center - .5, id(models[2]): center + .5}
    options = {"axis_batch_size": 7}
    calls = []

    def predict(model, observed, criterion, **actual_options):
        assert observed is source and actual_options == options and not torch.is_grad_enabled()
        calls.append(model)
        return outputs[id(model)].clone()

    monkeypatch.setattr(original, "streamed_point_predictions", predict)
    monkeypatch.setattr(distillation, "streamed_point_predictions", predict)
    monkeypatch.setattr(original, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(distillation, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())
    metadata = {"domain": domain, "structured": structured, "gradient_data": "synthetic",
                "family": family, "component": component, "seed": 0, "validation_error": float("inf")}
    episode = ObservedOnlyEpisode(source, metadata)
    previous, reference = teacher.mixture.parent_teacher_target, teacher.mixture.reference_teacher_target
    with teacher.runtime({}, tmp_path, identities):
        prediction, routing = teacher.mixture.parent_teacher_target(models[0], models[1:], None, episode, options)
        assert not prediction.requires_grad
        assert routing["policy"] == teacher.POLICY
        assert teacher.mixture.reference_teacher_target is reference
        if domain == "imputation":
            assert routing["target"] == "observed_blend"
            assert calls == [models[2], models[1]]
            assert torch.equal(prediction, center)
            assert routing["specialist_weight"] == pytest.approx(.5)
            assert routing["evidence_cells"] == 4
            assert routing["teacher_identity"] == {name: identities[name] for name in ("general", "specialist")}
        else:
            assert routing["target"] == "specialist"
            assert calls == [models[2]]
            assert torch.equal(prediction, outputs[id(models[2])])
            assert routing["teacher_identity"] == identities["specialist"]
    assert teacher.mixture.parent_teacher_target is previous
    assert teacher.mixture.reference_teacher_target is reference


@pytest.mark.parametrize("structured", [False, True])
def test_metadata_does_not_select_teacher_using_seed_or_error(structured):
    for seed in (0, 1, 2147483647):
        for error in (0., float("nan"), float("inf")):
            assert teacher.role_for({"domain": "imputation", "gradient_data": "synthetic",
                "structured": structured, "seed": seed, "validation_error": error,
                "queries": 0, "target_std": 0.}) == "observed_blend"


@pytest.mark.parametrize("role", ["initialization", "general", "specialist"])
def test_changed_checkpoint_bytes_are_rejected(tmp_path, monkeypatch, role):
    models, identities = fixture(tmp_path)
    with open(identities[role]["checkpoint"], "ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())
    monkeypatch.setattr(teacher, "observed_blend_target", lambda *args: pytest.fail("Unchecked checkpoint"))
    episode = ObservedOnlyEpisode(torch.ones(1, 2, 2), {
        "domain": "imputation", "structured": True, "gradient_data": "synthetic"})
    previous = teacher.mixture.parent_teacher_target
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        with teacher.runtime({}, tmp_path, identities):
            teacher.mixture.parent_teacher_target(models[0], models[1:], None, episode)
    assert teacher.mixture.parent_teacher_target is previous


def test_only_policy_and_docstring_change_from_immutable_legacy_blueprint():
    assert teacher.POLICY == {"legacy": "observed_blend", "structured": "observed_blend", "recsys": "specialist"}
    assert legacy.POLICY == {"legacy": "observed_blend", "structured": "general", "recsys": "specialist"}
    before, after = ast.parse(legacy.ADAPTED_SOURCE), ast.parse(teacher.ADAPTED_SOURCE)
    for tree in (before, after):
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "POLICY"
                                                   for target in node.targets))]
    assert ast.dump(before) == ast.dump(after)
    for name in ("runtime", "role_for", "identities", "verify_model"):
        assert inspect.unwrap(getattr(teacher, name)).__globals__ is vars(teacher)
    assert teacher.observed_blend_target is distillation.teacher_target
    assert teacher.ADAPTED_SOURCE_SHA256 == hashlib.sha256(teacher.ADAPTED_SOURCE.encode()).hexdigest()
    assert len(FAMILIES) == 12 and len(COMPONENTS) == 4 and len(CASES) == 57
