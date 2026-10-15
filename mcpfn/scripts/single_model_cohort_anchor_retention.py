"""Use frozen V8 retention on native/two-phase tasks and V29 on other imputation."""

from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
import types

import numpy as np
import torch

import single_model_all_imputation_retention as predecessor
from single_model_all_imputation_retention import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json,
)
from single_model_structured_mixture import FAMILIES


REFERENCE_SHA256S = {
    "v8": "7b79b3b650c1e44b2a2b6f3b69839623871406b7a60ef5a17fd08d253125ba76",
    "v29": predecessor.REFERENCE_SHA256,
}
SPECIFICATION = {name: value for name, value in predecessor.SPECIFICATION.items() if name != "reference_sha256"}
SPECIFICATION.update(reference_sha256s=REFERENCE_SHA256S,
    reference_assignment={"legacy": "v8", "mnar_two_phase_subset": "v8", "other_structured": "v29", "recsys": None},
    assignment_inputs="validated original synthetic domain, component and structured-family metadata only")
POLICY = ("Prospective reference-assignment experiment after V41's audited failed confirmation and full284-case "
    "initializer diagnosis. Keep unchanged V29 initialization. Use the original frozen V8 retention reference "
    "for all eight legacy imputation components and every original two-phase-subset MNAR component; use frozen "
    "V29 for all other structured imputation. Ratings retain the exact original callback with no retention. "
    "Assignment uses only declared synthetic training cohort metadata, never seeds as selectors, query truth, "
    "errors, validation labels, columns or benchmark identity. Keep weight10, equal queried-column averaging, "
    "query-count-shrunk variance, one unit-variance pseudocount,1e-6 floor and all original queries unchanged. "
    "Keep V41's legacy observed-blend/general-structured/specialist-rating targets, priors,.5/.25/.25 masses, "
    "Huber/NLL/CDF/query/point losses, fixed-half optimizer protection and every original qualification guard. "
    "Both references are frozen, independently checkpoint-verified and training-only; inference remains one "
    "ordinary network. Save every role decision and full reference/source/mask for independent two-reference "
    "replay. This standalone policy does not authorize training seeds, a probe, a full trial or qualification.")


def reference_role(metadata):
    if not predecessor.eligible(metadata):
        return None
    if not metadata.get("structured", False):
        return "v8"
    if metadata.get("family") not in FAMILIES:
        raise ValueError("Require an original structured family before assigning its frozen reference")
    return "v8" if metadata["family"] == "mnar_two_phase_subset" else "v29"


def eligible(metadata):
    return reference_role(metadata) is not None


def validate_identities(identities):
    if set(identities) != set(REFERENCE_SHA256S):
        raise ValueError("Require both original frozen references")
    for role, digest in REFERENCE_SHA256S.items():
        identity = identities[role]
        if set(identity) != {"checkpoint", "sha256"} or identity["sha256"] != digest or not identity["checkpoint"]:
            raise ValueError("Never substitute a reference checkpoint or invert cohort identities")


def make_references(student, identities):
    validate_identities(identities)
    references = {}
    for role, identity in identities.items():
        if sha256(identity["checkpoint"]) != identity["sha256"]:
            raise ValueError("Reference bytes must match before allocating the frozen model")
        reference = deepcopy(student).eval().requires_grad_(False)
        state = torch.load(identity["checkpoint"], map_location="cpu", weights_only=True, mmap=True)["model"]
        reference.load_state_dict(state, strict=True)
        verify_model(reference, identity)
        references[role] = reference
    return references


for name in ("validate_target", "shrunk_variance", "column_coefficients", "independent_penalty"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Do not retain predecessor loss closures")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE_SHA256 = predecessor.BACKWARD_SOURCE_SHA256
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())


@contextmanager
def runtime(backend, directory, references, identities, *, query_seed_offset):
    directory = Path(directory)
    original = backend.backward_episode
    validate_identities(identities)
    if (set(references) != set(identities) or references["v8"] is references["v29"]
            or not isinstance(original, partial) or original.func is not query.backward_episode
            or original.args or original.keywords != {"query_seed_offset": query_seed_offset}
            or type(query_seed_offset) is not int or query_seed_offset < 0):
        raise ValueError("Bind both distinct references and the unchanged original query callback")
    verified = {role: verify_model(model, identities[role]) for role, model in references.items()}
    directory.mkdir()
    records, seeds = [], set()
    complete = False

    def backward(student, criterion, episode, teacher, weight):
        role = reference_role(episode.metadata)
        seed = episode.metadata["seed"]
        if seed in seeds:
            raise ValueError("Never replace a consumed synthetic reference")
        seeds.add(seed)
        record = {"seed": seed, "eligible": role is not None, "reference_role": role,
            "shape": list(episode.target.shape), "queries": int(episode.mask.sum()), "metadata": dict(episode.metadata)}
        if role is None:
            value = original(student, criterion, episode, teacher, weight)
        else:
            reference, identity = references[role], identities[role]
            pointers = {parameter.data_ptr() for parameter in student.parameters()}
            if any(student is model or pointers.intersection(parameter.data_ptr() for parameter in model.parameters())
                    for model in references.values()):
                raise ValueError("Neither frozen reference may alias the trainable student")
            with torch.no_grad(), compute_context(episode.source.device):
                prediction = streamed_point_predictions(reference, episode.source, criterion, **STREAM_OPTIONS).detach()
            validate_target(episode, prediction)
            path = directory / f"reference_{seed}.pt"
            atomic_torch_save({"prediction": prediction.float().cpu(), "source": episode.source.detach().cpu(),
                "mask": episode.mask.detach().cpu(), "identity": identity, "reference_role": role,
                "specification": SPECIFICATION, "metadata": dict(episode.metadata)}, path)
            record.update(artifact=str(path), sha256=sha256(path))
            records.append(record)
            value = _backward_episode(student, criterion, episode, teacher, weight,
                query_seed_offset=query_seed_offset, retention_target=prediction)
        if role is None:
            records.append(record)
        return value

    backend.backward_episode = backward
    try:
        yield
        if any(verify_model(model, identities[role]) != verified[role] for role, model in references.items()):
            raise ValueError("Both references must remain bitwise unchanged")
        complete = True
    finally:
        backend.backward_episode = original
        write_json(directory / "manifest.json", {"complete": complete, "policy": POLICY,
            "specification": SPECIFICATION, "reference": verified, "records": records,
            "backward_source_sha256": BACKWARD_SOURCE_SHA256, "inference_ensemble": False})
