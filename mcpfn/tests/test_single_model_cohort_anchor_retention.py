from contextlib import nullcontext
from copy import deepcopy
from functools import partial
import json
from types import SimpleNamespace

import pytest
import torch

import single_model_cohort_anchor_retention as retention
import test_single_model_anchor_retention as original_tests
import test_single_model_column_anchor_retention as column_tests
import test_single_model_shrunk_column_anchor_retention as shrunk_tests


@pytest.mark.parametrize("component", range(8))
def test_every_original_legacy_component_uses_v8(component):
    metadata = original_tests.metadata(component, prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    assert retention.reference_role(metadata) == "v8"


@pytest.mark.parametrize("component", retention.SPECIFICATION["structured_components"])
@pytest.mark.parametrize("family", retention.FAMILIES)
def test_structured_assignment_is_fixed_for_all_original_components_and_families(component, family):
    metadata = original_tests.metadata(component, structured=True, family=family)
    assert retention.reference_role(metadata) == ("v8" if family == "mnar_two_phase_subset" else "v29")
    metadata.update(seed=987654, error=1e30, query_count=0, validation_failure=True, query_truth="not inspected")
    assert retention.reference_role(metadata) == ("v8" if family == "mnar_two_phase_subset" else "v29")


@pytest.mark.parametrize("changes", [dict(gradient_data="real"), dict(seed=True), dict(seed=-1), dict(domain="unknown"),
    dict(structured=1), dict(component=True), dict(component=8), dict(prior="wrong"),
    dict(structured=True, component="new_prior"), dict(domain="recsys", structured=True),
    dict(structured=True, component="low_rank_mcar", family="unknown")])
def test_malformed_metadata_cannot_choose_a_reference(changes):
    with pytest.raises(ValueError):
        retention.reference_role(original_tests.metadata(**changes))


def test_ratings_do_not_acquire_a_reference():
    assert retention.reference_role(original_tests.metadata(domain="recsys")) is None
    assert not retention.eligible(original_tests.metadata(domain="recsys"))


@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize("weight", [.25, 1.])
def test_unchanged_callback_matches_dense_gradients_and_detaches_reference(monkeypatch, matched, weight):
    monkeypatch.setattr(shrunk_tests, "retention", retention)
    shrunk_tests.test_callback_matches_independent_dense_gradient_with_detached_targets(monkeypatch, matched, weight)


@pytest.mark.parametrize("transposed", [False, True])
def test_every_streamed_model_gradient_still_matches_dense(monkeypatch, transposed):
    monkeypatch.setattr(column_tests, "retention", retention)
    column_tests.test_full_streamed_network_matches_dense_loss_and_all_parameter_gradients(monkeypatch, transposed)


def test_shrinkage_singletons_constants_and_count_dependence_are_unchanged(monkeypatch):
    monkeypatch.setattr(shrunk_tests, "retention", retention)
    shrunk_tests.test_exact_shrinkage_formula_keeps_empty_single_and_constant_columns()
    shrunk_tests.test_permutation_invariant_but_replication_intentionally_changes_count_shrinkage()


@pytest.fixture
def runtime_fixture(monkeypatch):
    references = {role: torch.nn.Linear(2, 1).eval().requires_grad_(False) for role in ("v8", "v29")}
    identities = {role: {"checkpoint": f"/{role}.pth", "sha256": digest}
        for role, digest in retention.REFERENCE_SHA256S.items()}
    calls = []

    def original(*arguments, **keywords):
        calls.append(("original", arguments, keywords))
        return 7.

    def verify(model, identity):
        return {**identity, "state_tensors": len(model.state_dict()), "bitwise_equal": True,
            "values": [value.detach().flatten().tolist() for value in model.state_dict().values()]}

    def predict(model, source, criterion, **options):
        role = next(role for role, reference in references.items() if reference is model)
        calls.append(("predict", role))
        return torch.full_like(source, 1. if role == "v8" else 2.)

    def backward(student, criterion, episode, teacher, weight, *, query_seed_offset, retention_target):
        assert not retention_target.requires_grad and query_seed_offset == 5
        assert teacher is episode.target and weight == .25
        calls.append(("retention", retention.reference_role(episode.metadata), retention_target.clone()))
        return 3.

    monkeypatch.setattr(retention.query, "backward_episode", original)
    monkeypatch.setattr(retention, "verify_model", verify)
    monkeypatch.setattr(retention, "streamed_point_predictions", predict)
    monkeypatch.setattr(retention, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(retention, "_backward_episode", backward)
    backend = SimpleNamespace(backward_episode=partial(original, query_seed_offset=5))
    return references, identities, backend, calls


def test_runtime_preserves_complete_roles_sources_and_exact_rating_callback(tmp_path, runtime_fixture):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    student = torch.nn.Linear(2, 1)
    metadata = [original_tests.metadata(),
        original_tests.metadata("low_rank_mcar", structured=True, family="mnar_two_phase_subset"),
        original_tests.metadata("slow_tail_mcar", structured=True, family="mar"),
        original_tests.metadata(domain="recsys")]
    with retention.runtime(backend, tmp_path / "references", references, identities, query_seed_offset=5):
        for index, fields in enumerate(metadata):
            episode = original_tests.episode()
            episode.metadata = {**fields, "seed": 1000 + index}
            assert backend.backward_episode(student, None, episode, episode.target, .25) == (7. if index == 3 else 3.)
    assert backend.backward_episode is callback
    assert [call[1] for call in calls if call[0] == "predict"] == ["v8", "v8", "v29"]
    rating = [call for call in calls if call[0] == "original"]
    assert len(rating) == 1 and rating[0][2] == {"query_seed_offset": 5}
    manifest = json.loads((tmp_path / "references/manifest.json").read_text())
    assert manifest["complete"] is True and set(manifest["reference"]) == {"v8", "v29"}
    assert [record["reference_role"] for record in manifest["records"]] == ["v8", "v8", "v29", None]
    assert "artifact" not in manifest["records"][-1]
    for record in manifest["records"][:-1]:
        payload = torch.load(record["artifact"], weights_only=True)
        assert payload["reference_role"] == record["reference_role"]
        assert payload["identity"] == identities[record["reference_role"]]
        assert payload["mask"].dtype == torch.bool and torch.isnan(payload["source"][payload["mask"]]).all()
        assert payload["prediction"].shape == payload["source"].shape


@pytest.mark.parametrize("failure", ["reused_seed", "student_alias", "unused_reference_changed", "callback"])
def test_runtime_fails_closed_and_restores_callback(tmp_path, runtime_fixture, failure):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    student = references["v8"] if failure == "student_alias" else torch.nn.Linear(2, 1)
    episode = original_tests.episode()
    with pytest.raises((ValueError, RuntimeError)):
        with retention.runtime(backend, tmp_path / "references", references, identities, query_seed_offset=5):
            backend.backward_episode(student, None, episode, episode.target, .25)
            if failure == "reused_seed":
                backend.backward_episode(student, None, episode, episode.target, .25)
            elif failure == "unused_reference_changed":
                with torch.no_grad():
                    references["v29"].weight.add_(1)
            elif failure == "callback":
                raise RuntimeError("sentinel")
    assert backend.backward_episode is callback
    assert json.loads((tmp_path / "references/manifest.json").read_text())["complete"] is False


@pytest.mark.parametrize("failure", ["missing", "swapped", "digest", "duplicate_model"])
def test_invalid_reference_pair_cannot_create_artifacts(tmp_path, runtime_fixture, failure):
    references, identities, backend, calls = runtime_fixture
    identities = deepcopy(identities)
    if failure == "missing":
        identities.pop("v8")
    elif failure == "swapped":
        identities["v8"], identities["v29"] = identities["v29"], identities["v8"]
    elif failure == "digest":
        identities["v8"]["sha256"] = "changed"
    else:
        references["v8"] = references["v29"]
    with pytest.raises(ValueError):
        with retention.runtime(backend, tmp_path / "references", references, identities, query_seed_offset=5):
            pytest.fail("Invalid identities must never enter training")
    assert not (tmp_path / "references").exists()


def test_only_reference_assignment_changes_the_loss_specification():
    for key, value in retention.predecessor.SPECIFICATION.items():
        if key != "reference_sha256":
            assert retention.SPECIFICATION[key] == value
    assert retention.BACKWARD_SOURCE == retention.predecessor.BACKWARD_SOURCE
    assert retention.BACKWARD_SOURCE_SHA256 == retention.predecessor.BACKWARD_SOURCE_SHA256
    assert retention.SPECIFICATION["weight"] == 10.
    assert retention.SPECIFICATION["prior_count"] == retention.SPECIFICATION["prior_variance"] == 1.
    assert retention.SPECIFICATION["variance_floor"] == 1e-6
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("corrupt", [None, "bytes", "shape"])
def test_make_references_loads_both_exact_checkpoints_without_changing_student(tmp_path, monkeypatch, corrupt):
    student = torch.nn.Linear(2, 1)
    initial = deepcopy(student.state_dict())
    states, identities = {}, {}
    for index, role in enumerate(("v8", "v29")):
        reference = torch.nn.Linear(3 if corrupt == "shape" and role == "v29" else 2, 1)
        with torch.no_grad():
            reference.weight.fill_(index + 2.)
            reference.bias.fill_(index + 3.)
        states[role] = reference.state_dict()
        path = tmp_path / f"{role}.pth"
        torch.save({"model": states[role]}, path)
        identities[role] = {"checkpoint": str(path), "sha256": retention.sha256(path)}
    monkeypatch.setattr(retention, "REFERENCE_SHA256S", {role: identity["sha256"] for role, identity in identities.items()})
    if corrupt == "bytes":
        with open(identities["v29"]["checkpoint"], "ab") as handle:
            handle.write(b"changed")
    if corrupt:
        with pytest.raises((ValueError, RuntimeError)):
            retention.make_references(student, identities)
    else:
        references = retention.make_references(student, identities)
        assert set(references) == {"v8", "v29"}
        pointers = {parameter.data_ptr() for parameter in student.parameters()}
        for role, reference in references.items():
            assert not reference.training and all(not parameter.requires_grad for parameter in reference.parameters())
            for name, value in reference.state_dict().items():
                assert torch.equal(value, states[role][name]) and value.data_ptr() not in pointers
                pointers.add(value.data_ptr())
    assert student.training and all(parameter.requires_grad for parameter in student.parameters())
    assert all(torch.equal(value, initial[name]) for name, value in student.state_dict().items())
