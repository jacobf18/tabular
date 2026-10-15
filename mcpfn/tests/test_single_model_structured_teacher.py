from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from recsys_evaluation_common import sha256
import single_model_structured_teacher as teacher
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
    ("imputation", False, "initialization", 0), ("imputation", True, "general", 1),
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
