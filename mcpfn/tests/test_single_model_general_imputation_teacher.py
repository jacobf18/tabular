from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from recsys_evaluation_common import sha256
import single_model_general_imputation_teacher as teacher
import train_single_matrixformer_parent_teacher as original


def fixture(tmp_path):
    models = [torch.nn.Linear(2, 2).eval().requires_grad_(False) for index in range(3)]
    identities = {}
    for role, model in zip(("initialization", "general", "specialist"), models):
        path = tmp_path / f"{role}.pt"
        torch.save({"model": model.state_dict()}, path)
        identities[role] = {"checkpoint": str(path), "sha256": sha256(path)}
    return models, identities


@pytest.mark.parametrize("domain,structured,role,index", [
    ("imputation", False, "general", 1), ("imputation", True, "general", 1),
    ("recsys", False, "specialist", 2)])
def test_actual_teacher_and_provenance(tmp_path, monkeypatch, domain, structured, role, index):
    models, identities = fixture(tmp_path)
    episode = SimpleNamespace(source=torch.ones(1, 2, 2), metadata={
        "domain": domain, "structured": structured, "gradient_data": "synthetic"})
    called = []

    def predict(model, source, criterion, **options):
        called.append(model)
        return model(source)

    monkeypatch.setattr(original, "streamed_point_predictions", predict)
    monkeypatch.setattr(original, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())
    previous = teacher.mixture.parent_teacher_target
    reference = teacher.mixture.reference_teacher_target
    with teacher.runtime({}, tmp_path, identities):
        target, routing = teacher.mixture.parent_teacher_target(models[0], models[1:], None, episode)
        assert called == [models[index]]
        assert torch.equal(target, models[index](episode.source)) and not target.requires_grad
        assert routing["target"] == role and routing["teacher_identity"] == identities[role]
        assert routing["policy"] == teacher.POLICY
        assert teacher.mixture.reference_teacher_target is reference
    assert teacher.mixture.parent_teacher_target is previous


@pytest.mark.parametrize("corruption", ["weights", "hash", "training", "gradient"])
def test_actual_state_not_merely_role_label_is_checked(tmp_path, corruption):
    models, identities = fixture(tmp_path)
    model, identity = models[1], identities["general"]
    if corruption == "weights":
        model.weight.add_(1)
    elif corruption == "hash":
        identity["sha256"] = "wrong"
    elif corruption == "training":
        model.train()
    else:
        model.requires_grad_(True)
    with pytest.raises(ValueError):
        teacher.verify_model(model, identity)


@pytest.mark.parametrize("metadata", [
    {"gradient_data": "real", "domain": "imputation"},
    {"gradient_data": "synthetic", "domain": "unknown"},
    {"gradient_data": "synthetic", "domain": "recsys", "structured": True},
    {"gradient_data": "synthetic", "domain": "imputation", "structured": "yes"}])
def test_unsupported_episode_rejected(metadata):
    with pytest.raises(ValueError):
        teacher.role_for(metadata)


def test_scope_restored_on_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())
    previous = teacher.mixture.parent_teacher_target
    with pytest.raises(RuntimeError, match="interrupted"):
        with teacher.runtime({}, tmp_path, {}):
            raise RuntimeError("interrupted")
    assert teacher.mixture.parent_teacher_target is previous


def test_parent_identity_cannot_be_substituted():
    with pytest.raises(ValueError):
        teacher.identities({"parent_checkpoint": "candidate", "teachers": ["general", "specialist"]}, "v8")


def test_only_teacher_role_changes_without_mutating_original_policy_or_evaluation():
    import ast
    import inspect

    assert teacher.POLICY == {"legacy": "general", "structured": "general", "recsys": "specialist"}
    assert teacher.template.POLICY == {"legacy": "initialization", "structured": "general", "recsys": "specialist"}
    assert teacher.POLICY is not teacher.template.POLICY
    original_tree = ast.parse(inspect.getsource(teacher.template))
    actual_tree = ast.parse(teacher.ADAPTED_SOURCE)
    for name in ("role_for", "identities", "verify_model", "runtime"):
        original_node = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        actual_node = next(node for node in actual_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual_node) == ast.dump(original_node)
        assert inspect.unwrap(getattr(teacher, name)).__globals__ is teacher.__dict__


def test_teacher_model_cannot_be_replaced_mid_context(tmp_path, monkeypatch):
    models, identities = fixture(tmp_path)
    episode = SimpleNamespace(source=torch.ones(1, 2, 2), metadata={
        "domain": "imputation", "structured": False, "gradient_data": "synthetic"})
    monkeypatch.setattr(original, "streamed_point_predictions", lambda model, source, criterion, **options: model(source))
    monkeypatch.setattr(original, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(teacher.mixture, "runtime", lambda *args: nullcontext())
    previous = teacher.mixture.parent_teacher_target
    with teacher.runtime({}, tmp_path, identities):
        teacher.mixture.parent_teacher_target(models[0], models[1:], None, episode)
        replacement = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        with pytest.raises(ValueError, match="replace a verified"):
            teacher.mixture.parent_teacher_target(models[0], [replacement, models[2]], None, episode)
    assert teacher.mixture.parent_teacher_target is previous
