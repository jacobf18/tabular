from copy import deepcopy
import inspect
import json

import pytest
import torch

import single_model_native_parent_retention as retention
import single_model_dual_parent_retention as predecessor
import test_single_model_anchor_retention as originals
import test_single_model_column_anchor_retention as columns
import test_single_model_dual_parent_retention as predecessor_tests
import test_single_model_native_column_mean as native_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize("component", range(8))
@pytest.mark.parametrize("transposed", [False, True])
def test_all_native_components_use_parent_without_changing_mean_weights(component, transposed):
    episode = originals.episode()
    episode.metadata = originals.metadata(component, transposed=transposed,
        prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    episode.target, episode.mask = columns.column_example()
    if transposed:
        episode.target, episode.mask = episode.target.transpose(-1, -2), episode.mask.transpose(-1, -2)
    assert retention.reference_role(episode.metadata) == "v29"
    assert retention.reference_weights("v29") == {"v29": 1.}
    with retention.stronger.training_objective():
        actual, expected = retention.teacher_mean_weights(episode), predecessor.teacher_mean_weights(episode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    assert retention.retention_weighting(episode.metadata) == predecessor.retention_weighting(episode.metadata)
    episode.metadata.update(seed=999999, error=-1e12, validation_failure=True, query_truth="not inspected")
    assert retention.reference_role(episode.metadata) == "v29"


@pytest.mark.parametrize("family", retention.FAMILIES)
@pytest.mark.parametrize("component", retention.SPECIFICATION["structured_components"])
def test_every_structured_family_keeps_exact_reference_and_penalty(family, component):
    metadata = originals.metadata(component, structured=True, family=family)
    assert retention.reference_role(metadata) == predecessor.reference_role(metadata) == "structured_mean"
    assert retention.reference_weights("structured_mean") == predecessor.reference_weights("structured_mean")
    assert retention.retention_weighting(metadata) == predecessor.retention_weighting(metadata)
    components = {"v8": torch.tensor([[1e-8, 4.], [5., -4.]]), "v29": torch.tensor([[1e-7, 2.], [1., 4.]])}
    torch.testing.assert_close(retention.combine_predictions("structured_mean", components),
        predecessor.combine_predictions("structured_mean", components), rtol=0, atol=0)


def test_single_target_is_exact_frozen_parent_and_never_general():
    parent = torch.tensor([[.1, -.7], [12., 3.]])
    assert retention.combine_predictions("v29", {"v29": parent}) is parent
    identities = {role: {"checkpoint": f"/{role}.pth", "sha256": digest}
        for role, digest in retention.REFERENCE_SHA256S.items()}
    assert retention.target_identity("v29", identities) == identities["v29"]
    assert retention.target_identity("structured_mean", identities) == predecessor.target_identity("structured_mean", identities)
    for role, components in (("general", {"general": parent}), ("v29", {"general": parent}),
            ("v29", {"v29": parent, "v8": parent}), ("v29", {})):
        with pytest.raises(ValueError):
            retention.combine_predictions(role, components)


@pytest.mark.parametrize("mutation", ["dtype", "gradient", "batch", "none"])
def test_native_target_keeps_original_full_grid_contract(mutation):
    value = torch.zeros(2, 3)
    if mutation == "dtype":
        value = value.double()
    elif mutation == "gradient":
        value.requires_grad_()
    elif mutation == "batch":
        value = value.repeat(2, 1, 1)
    else:
        value = None
    with pytest.raises(ValueError):
        retention.combine_predictions("v29", {"v29": value})


def test_nonfinite_complete_target_is_rejected_before_backward():
    episode = originals.episode()
    value = episode.target.clone()
    value[0, 0, 0] = float("nan")
    combined = retention.combine_predictions("v29", {"v29": value})
    with pytest.raises(ValueError):
        retention.validate_target(episode, combined)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("cohort", ["native", "mar", "mar_block_neural", "mnar_two_phase_subset"])
def test_dense_and_streamed_loss_and_every_parameter_gradient_match(monkeypatch, transposed, cohort):
    monkeypatch.setattr(native_tests, "retention", retention)
    native_tests.test_real_streamed_model_matches_dense_loss_and_every_parameter(monkeypatch, transposed, cohort)


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "dtype", "gradient", "not_tensor", "dimensions"])
def test_structured_component_contract_is_unchanged(monkeypatch, failure):
    monkeypatch.setattr(predecessor_tests, "retention", retention)
    predecessor_tests.test_malformed_components_cannot_be_combined(failure)


def test_loss_runtime_and_all_other_specifications_are_unchanged():
    assert retention.BACKWARD_SOURCE == predecessor.BACKWARD_SOURCE
    assert retention.BACKWARD_SOURCE_SHA256 == predecessor.BACKWARD_SOURCE_SHA256
    assert retention.RUNTIME_SOURCE == predecessor.RUNTIME_SOURCE
    assert retention.REFERENCE_SHA256S == predecessor.REFERENCE_SHA256S
    expected = deepcopy(predecessor.SPECIFICATION)
    for key in ("reference_assignment", "initialization"):
        expected[key] = retention.SPECIFICATION[key]
    expected["reference_combination"]["scope"] = retention.SPECIFICATION["reference_combination"]["scope"]
    assert retention.SPECIFICATION == expected
    for key in predecessor.MEAN_SPECIFICATION.keys() - {"retention"}:
        assert retention.MEAN_SPECIFICATION[key] == predecessor.MEAN_SPECIFICATION[key]
    assert retention.reference_role(originals.metadata(domain="recsys")) is None
    assert retention.teacher_mean_weights(type("Episode", (), {"metadata": originals.metadata(domain="recsys")})()) is None
    for function in vars(retention).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(retention):
            assert unresolved_globals(function) == []


@pytest.fixture
def runtime_fixture(monkeypatch):
    monkeypatch.setattr(predecessor_tests, "retention", retention)
    return predecessor_tests.runtime_fixture.__wrapped__(monkeypatch)


def test_runtime_preserves_parent_native_components_and_does_not_infer_general(tmp_path, runtime_fixture):
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
    assert [call[1] for call in calls if call[0] == "predict"] == ["v29", "v8", "v29"]
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
        assert (payload["prediction"] == (4. if role == "v29" else 3.)).all()


@pytest.mark.parametrize("failure", ["duplicate", "changed_reference", "student_alias", "reference_alias", "wrong_identity"])
def test_invalid_runtime_restores_original_callback(tmp_path, runtime_fixture, failure):
    predecessor_tests.test_invalid_runtime_restores_original_callback(tmp_path, runtime_fixture, failure)


@pytest.mark.parametrize("changes", [dict(gradient_data="real"), dict(seed=True), dict(seed=-1),
    dict(domain="unknown"), dict(structured=1), dict(component=True), dict(component=8),
    dict(prior="wrong"), dict(structured=True, component="new_prior"), dict(domain="recsys", structured=True)])
def test_malformed_metadata_fails_closed(changes):
    with pytest.raises(ValueError):
        retention.reference_role(originals.metadata(**changes))
