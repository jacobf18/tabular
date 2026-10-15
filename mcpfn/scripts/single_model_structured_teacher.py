"""Training-only structured-general distillation with checkpoint-bound teacher identities."""

from contextlib import contextmanager
from pathlib import Path

import torch

from recsys_evaluation_common import sha256
import single_model_structured_mixture as mixture
from streamed_large_matrix_training import STREAM_OPTIONS
from train_single_matrixformer_parent_teacher import parent_teacher_target


POLICY = {"legacy": "initialization", "structured": "general", "recsys": "specialist"}


def role_for(metadata):
    if metadata.get("gradient_data") != "synthetic":
        raise ValueError("Teachers require synthetic training data")
    domain = metadata.get("domain")
    structured = metadata.get("structured", False)
    if not isinstance(structured, bool) or domain not in {"imputation", "recsys"}:
        raise ValueError("Require an explicit supported synthetic domain and boolean structured flag")
    if domain == "recsys":
        if structured:
            raise ValueError("Structured teachers apply only to imputation")
        return POLICY["recsys"]
    return POLICY["structured" if structured else "legacy"]


def identities(config, parent):
    paths = [str(parent), *config["teachers"]]
    if len(paths) != 3 or config["parent_checkpoint"] != paths[0] or len(set(paths)) != 3:
        raise ValueError("Require original V8, original general and original specialist identities")
    return {role: {"checkpoint": path, "sha256": config["input_sha256"][path]}
            for role, path in zip(("initialization", "general", "specialist"), paths)}


def verify_model(model, identity):
    path = Path(identity["checkpoint"])
    if sha256(path) != identity["sha256"]:
        raise ValueError("Teacher checkpoint bytes changed")
    expected = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["model"]
    actual = model.state_dict()
    if (model.training or any(parameter.requires_grad for parameter in model.parameters())
            or actual.keys() != expected.keys()
            or any(not torch.equal(value.detach().cpu(), expected[name]) for name, value in actual.items())):
        raise ValueError("Actual frozen teacher tensors must equal the declared checkpoint")
    return {**identity, "state_tensors": len(actual), "bitwise_equal": True}


@contextmanager
def runtime(config, output, teacher_identities):
    original = mixture.parent_teacher_target
    verified = {}

    def target(initialization, teachers, criterion, episode, options=STREAM_OPTIONS):
        if len(teachers) != 2:
            raise ValueError("Require the two original teacher models")
        models = {"initialization": initialization, "general": teachers[0], "specialist": teachers[1]}
        for role, model in models.items():
            if role not in verified:
                verify_model(model, teacher_identities[role])
                verified[role] = model
            elif verified[role] is not model:
                raise ValueError("Never replace a verified training teacher during a run")
        role = role_for(episode.metadata)
        chosen = teachers[0] if role == "general" else initialization
        prediction, routing = parent_teacher_target(chosen, teachers, criterion, episode, options)
        return prediction, {**routing, "target": role, "policy": POLICY,
                            "teacher_identity": teacher_identities[role]}

    mixture.parent_teacher_target = target
    try:
        with mixture.runtime(config, output):
            yield
    finally:
        mixture.parent_teacher_target = original
