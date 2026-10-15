import inspect
import json

import pytest
import torch

import single_model_parent_structured_retention as retention
import single_model_native_column_mean as predecessor
import test_single_model_anchor_retention as originals
import test_single_model_cohort_anchor_retention as cohorts
import test_single_model_native_column_mean as native_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize("component", range(8))
def test_native_mean_and_general_reference_are_unchanged(monkeypatch, component):
    monkeypatch.setattr(native_tests, "retention", retention)
    native_tests.test_every_native_component_uses_same_column_formula_and_general_reference(component)
    metadata = originals.metadata(component, prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    assert retention.retention_weighting(metadata) == predecessor.retention_weighting(metadata)


@pytest.mark.parametrize("family", retention.FAMILIES)
@pytest.mark.parametrize("component", ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"))
def test_every_structured_family_uses_frozen_parent_uniformly(family, component):
    episode = originals.episode()
    episode.metadata = originals.metadata(component, structured=True, family=family)
    assert retention.reference_role(episode.metadata) == "v29"
    assert retention.retention_weighting(episode.metadata) == predecessor.retention_weighting(episode.metadata)
    assert retention.teacher_mean_weights(episode) is None
    episode.metadata.update(seed=999999, error=1e12, validation_failure=True)
    assert retention.reference_role(episode.metadata) == "v29"


@pytest.mark.parametrize("changes", [dict(gradient_data="real"), dict(seed=True), dict(seed=-1),
    dict(domain="unknown"), dict(structured=1), dict(component=True), dict(component=8),
    dict(prior="wrong"), dict(structured=True, component="new_prior"), dict(domain="recsys", structured=True)])
def test_invalid_metadata_cannot_select_a_reference(changes):
    with pytest.raises(ValueError):
        retention.reference_role(originals.metadata(**changes))


def test_only_training_reference_identity_changes():
    changed = {"reference_sha256s", "reference_assignment", "initialization"}
    assert retention.SPECIFICATION.keys() == predecessor.SPECIFICATION.keys()
    for name in predecessor.SPECIFICATION.keys() - changed:
        assert retention.SPECIFICATION[name] == predecessor.SPECIFICATION[name]
    assert retention.MEAN_SPECIFICATION.keys() == predecessor.MEAN_SPECIFICATION.keys()
    for name in predecessor.MEAN_SPECIFICATION.keys() - {"retention"}:
        assert retention.MEAN_SPECIFICATION[name] == predecessor.MEAN_SPECIFICATION[name]
    assert retention.BACKWARD_SOURCE == predecessor.BACKWARD_SOURCE
    assert retention.BACKWARD_SOURCE_SHA256 == predecessor.BACKWARD_SOURCE_SHA256
    assert retention.RUNTIME_SOURCE == predecessor.RUNTIME_SOURCE.replace(
        'references["v8"] is references["general"]', 'references["v29"] is references["general"]')
    assert set(retention.REFERENCE_SHA256S) == {"general", "v29"}
    assert retention.REFERENCE_SHA256S["general"] == predecessor.REFERENCE_SHA256S["general"]


def test_every_callback_binds_the_new_reference_policy():
    for value in vars(retention).values():
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(retention):
            assert unresolved_globals(value) == []
    for name in ("runtime", "_backward_episode", "teacher_mean_weights", "make_references", "validate_identities"):
        assert inspect.unwrap(getattr(retention, name)).__globals__ is vars(retention)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("cohort", ["native", "mar", "mar_block_neural", "mnar_two_phase_subset"])
def test_dense_and_streamed_loss_and_all_parameter_gradients_agree(monkeypatch, transposed, cohort):
    monkeypatch.setattr(native_tests, "retention", retention)
    native_tests.test_real_streamed_model_matches_dense_loss_and_every_parameter(monkeypatch, transposed, cohort)


@pytest.fixture
def runtime_fixture(monkeypatch):
    monkeypatch.setattr(cohorts, "retention", retention)
    references, identities, backend, calls = cohorts.runtime_fixture.__wrapped__(monkeypatch)
    references["general"] = references.pop("v8")

    def predict(model, source, criterion, **options):
        role = next(role for role, reference in references.items() if reference is model)
        calls.append(("predict", role))
        return torch.full_like(source, 1. if role == "general" else 2.)

    monkeypatch.setattr(retention, "streamed_point_predictions", predict)
    return references, identities, backend, calls


def test_runtime_saves_correct_targets_and_preserves_rating_callback(tmp_path, runtime_fixture):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    student = torch.nn.Linear(2, 1)
    metadata = [originals.metadata(), originals.metadata("low_rank_mcar", structured=True, family="mar"),
        originals.metadata("slow_tail_mcar", structured=True, family="mnar_skip_logic"), originals.metadata(domain="recsys")]
    with retention.runtime(backend, tmp_path / "references", references, identities, query_seed_offset=5):
        for index, fields in enumerate(metadata):
            episode = originals.episode()
            episode.metadata = {**fields, "seed": 1000 + index}
            assert backend.backward_episode(student, None, episode, episode.target, .25) == (7. if index == 3 else 3.)
    assert backend.backward_episode is callback
    assert [entry[1] for entry in calls if entry[0] == "predict"] == ["general", "v29", "v29"]
    assert sum(entry[0] == "original" for entry in calls) == 1
    manifest = json.loads((tmp_path / "references/manifest.json").read_text())
    assert manifest["complete"] and set(manifest["reference"]) == {"general", "v29"}
    assert [row["reference_role"] for row in manifest["records"]] == ["general", "v29", "v29", None]
    for row in manifest["records"]:
        if row["eligible"]:
            payload = torch.load(row["artifact"], weights_only=True)
            assert payload["identity"] == identities[row["reference_role"]]
            assert payload["weighting"] == retention.retention_weighting(row["metadata"])
            expected = 1. if row["reference_role"] == "general" else 2.
            assert (payload["prediction"] == expected).all()
        else:
            assert "artifact" not in row


@pytest.mark.parametrize("failure", ["duplicate", "changed_reference", "student_alias", "reference_alias", "wrong_identity"])
def test_invalid_runtime_restores_the_original_callback(tmp_path, runtime_fixture, failure):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    student = references["general"] if failure == "student_alias" else torch.nn.Linear(2, 1)
    if failure == "reference_alias":
        references["v29"] = references["general"]
    if failure == "wrong_identity":
        identities["v29"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        with retention.runtime(backend, tmp_path / "references", references, identities, query_seed_offset=5):
            episode = originals.episode()
            backend.backward_episode(student, None, episode, episode.target, .25)
            if failure == "duplicate":
                backend.backward_episode(student, None, episode, episode.target, .25)
            if failure == "changed_reference":
                with torch.no_grad():
                    references["v29"].weight.add_(1.)
    assert backend.backward_episode is callback
