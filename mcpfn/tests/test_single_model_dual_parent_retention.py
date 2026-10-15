from contextlib import nullcontext
from copy import deepcopy
from functools import partial
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

import single_model_dual_parent_retention as retention
import single_model_parent_structured_retention as predecessor
import test_single_model_anchor_retention as originals
import test_single_model_native_column_mean as native_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize("component", range(8))
def test_native_general_reference_and_mean_weighting_are_unchanged(monkeypatch, component):
    monkeypatch.setattr(native_tests, "retention", retention)
    native_tests.test_every_native_component_uses_same_column_formula_and_general_reference(component)


@pytest.mark.parametrize("family", retention.FAMILIES)
@pytest.mark.parametrize("component", retention.SPECIFICATION["structured_components"])
def test_uniform_fixed_mean_never_uses_errors_or_family_exceptions(family, component):
    metadata = originals.metadata(component, structured=True, family=family)
    assert retention.reference_role(metadata) == "structured_mean"
    assert retention.reference_weights("structured_mean") == {"v8": .5, "v29": .5}
    assert retention.retention_weighting(metadata) == predecessor.retention_weighting(metadata)
    metadata.update(seed=999999, validation_failure=True, query_truth="not inspected", error=-1e12)
    assert retention.reference_role(metadata) == "structured_mean"


def test_rating_and_loss_math_are_unchanged():
    assert retention.reference_role(originals.metadata(domain="recsys")) is None
    assert retention.BACKWARD_SOURCE == predecessor.BACKWARD_SOURCE
    assert retention.BACKWARD_SOURCE_SHA256 == predecessor.BACKWARD_SOURCE_SHA256
    changed = {"reference_sha256s", "reference_assignment", "initialization"}
    for name in predecessor.SPECIFICATION.keys() - changed:
        assert retention.SPECIFICATION[name] == predecessor.SPECIFICATION[name]
    for name in predecessor.MEAN_SPECIFICATION.keys() - {"retention"}:
        assert retention.MEAN_SPECIFICATION[name] == predecessor.MEAN_SPECIFICATION[name]


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("cohort", ["native", "mar", "mar_block_neural", "mnar_two_phase_subset"])
def test_dense_and_streamed_loss_and_every_parameter_gradient_match(monkeypatch, transposed, cohort):
    monkeypatch.setattr(native_tests, "retention", retention)
    native_tests.test_real_streamed_model_matches_dense_loss_and_every_parameter(monkeypatch, transposed, cohort)


def test_combination_is_exact_half_sum_and_native_is_identical():
    components = {"v8": torch.tensor([[1e30, 3., -9.]]), "v29": torch.tensor([[1e30, -1., 7.]])}
    saved = deepcopy(components)
    result = retention.combine_predictions("structured_mean", components)
    assert torch.equal(result, components["v8"] * .5 + components["v29"] * .5)
    assert all(torch.equal(components[name], saved[name]) for name in components)
    assert not result.requires_grad
    assert retention.combine_predictions("general", {"general": result}) is result
    permuted = {name: prediction[:, [2, 0, 1]] for name, prediction in components.items()}
    assert torch.equal(retention.combine_predictions("structured_mean", permuted), result[:, [2, 0, 1]])
    batched = {name: prediction.unsqueeze(0) for name, prediction in components.items()}
    assert torch.equal(retention.combine_predictions("structured_mean", batched), result.unsqueeze(0))


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "dtype", "gradient", "not_tensor", "dimensions"])
def test_malformed_components_cannot_be_combined(failure):
    components = {"v8": torch.zeros(2, 3), "v29": torch.ones(2, 3)}
    if failure == "missing":
        components.pop("v8")
    elif failure == "extra":
        components["general"] = torch.zeros(2, 3)
    else:
        components["v29"] = {"shape": torch.zeros(2, 4), "dtype": torch.zeros(2, 3).double(),
            "gradient": torch.zeros(2, 3, requires_grad=True), "not_tensor": None,
            "dimensions": torch.zeros(6)}[failure]
    with pytest.raises(ValueError):
        retention.combine_predictions("structured_mean", components)


def test_callbacks_resolve_only_current_globals():
    for function in vars(retention).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(retention):
            assert unresolved_globals(function) == []


def test_invalid_first_component_is_rejected_regardless_of_dictionary_order():
    with pytest.raises(ValueError):
        retention.combine_predictions("structured_mean", {"v29": torch.zeros(2, 3), "v8": None})


@pytest.fixture
def runtime_fixture(monkeypatch):
    references = {role: torch.nn.Linear(2, 1).eval().requires_grad_(False) for role in retention.REFERENCE_SHA256S}
    identities = {role: {"checkpoint": f"/{role}.pth", "sha256": digest}
        for role, digest in retention.REFERENCE_SHA256S.items()}
    calls = []

    def original(*arguments, **keywords):
        calls.append(("original",))
        return 7.

    def verify(model, identity):
        return {**identity, "values": [value.detach().flatten().tolist() for value in model.state_dict().values()]}

    def predict(model, source, criterion, **options):
        role = next(role for role, reference in references.items() if reference is model)
        calls.append(("predict", role))
        return torch.full_like(source, {"general": 1., "v8": 2., "v29": 4.}[role])

    def backward(student, criterion, episode, teacher, weight, *, query_seed_offset, retention_target):
        assert not retention_target.requires_grad and query_seed_offset == 5
        calls.append(("retention", retention_target.clone()))
        return 3.

    monkeypatch.setattr(retention.query, "backward_episode", original)
    monkeypatch.setattr(retention, "verify_model", verify)
    monkeypatch.setattr(retention, "streamed_point_predictions", predict)
    monkeypatch.setattr(retention, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(retention, "_backward_episode", backward)
    backend = SimpleNamespace(backward_episode=partial(original, query_seed_offset=5))
    return references, identities, backend, calls


def test_runtime_preserves_both_components_and_native_and_rating_paths(tmp_path, runtime_fixture):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    student = torch.nn.Linear(2, 1)
    metadata = [originals.metadata(), originals.metadata("low_rank_mcar", structured=True, family="mar"),
        originals.metadata(domain="recsys")]
    with retention.runtime(backend, tmp_path / "targets", references, identities, query_seed_offset=5):
        for index, fields in enumerate(metadata):
            episode = originals.episode()
            episode.metadata = {**fields, "seed": 1000 + index}
            assert backend.backward_episode(student, None, episode, episode.target, .25) == (7. if index == 2 else 3.)
    assert backend.backward_episode is callback
    assert [call[1] for call in calls if call[0] == "predict"] == ["general", "v8", "v29"]
    assert sum(call[0] == "original" for call in calls) == 1
    manifest = json.loads((tmp_path / "targets/manifest.json").read_text())
    assert manifest["complete"] and set(manifest["reference"]) == set(identities)
    for record in manifest["records"]:
        if not record["eligible"]:
            assert "artifact" not in record
            continue
        payload = torch.load(record["artifact"], weights_only=True)
        role = record["reference_role"]
        assert payload["identity"] == retention.target_identity(role, identities)
        assert payload["component_weights"] == retention.reference_weights(role)
        assert torch.equal(payload["prediction"], retention.combine_predictions(role, payload["components"]))
        assert (payload["prediction"] == (1. if role == "general" else 3.)).all()


@pytest.mark.parametrize("failure", ["duplicate", "changed_reference", "student_alias", "reference_alias", "wrong_identity"])
def test_invalid_runtime_restores_original_callback(tmp_path, runtime_fixture, failure):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    student = references["general"] if failure == "student_alias" else torch.nn.Linear(2, 1)
    if failure == "reference_alias":
        references["v8"] = references["v29"]
    if failure == "wrong_identity":
        identities["v8"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        with retention.runtime(backend, tmp_path / "targets", references, identities, query_seed_offset=5):
            episode = originals.episode()
            backend.backward_episode(student, None, episode, episode.target, .25)
            if failure == "duplicate":
                backend.backward_episode(student, None, episode, episode.target, .25)
            if failure == "changed_reference":
                with torch.no_grad():
                    references["v8"].weight.add_(1.)
    assert backend.backward_episode is callback
