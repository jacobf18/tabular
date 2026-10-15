import inspect
import json
import types

import pytest
import torch

import single_model_general_native_retention as retention
import single_model_uniform_v8_retention as predecessor
import test_single_model_anchor_retention as original_tests
import test_single_model_cohort_anchor_retention as cohort_tests
import test_single_model_family_balanced_retention as weighting_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize("component", range(8))
def test_every_native_component_uses_general_with_unchanged_weights(component):
    metadata = original_tests.metadata(component, prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    assert retention.reference_role(metadata) == "general"
    assert retention.retention_weighting(metadata) == predecessor.retention_weighting(metadata)
    assert retention.reference_role({**metadata, "seed": 98765, "error": 1e9, "validation_failure": True}) == "general"


@pytest.mark.parametrize("family", retention.FAMILIES)
@pytest.mark.parametrize("component", ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"))
def test_all_structured_families_keep_v8_without_performance_based_exceptions(family, component):
    metadata = original_tests.metadata(component, structured=True, family=family)
    assert retention.reference_role(metadata) == "v8"
    assert retention.retention_weighting(metadata) == predecessor.retention_weighting(metadata)


def test_only_reference_assignment_changes_and_all_backward_math_is_unchanged():
    assert retention.BACKWARD_SOURCE == predecessor.BACKWARD_SOURCE
    assert retention.BACKWARD_SOURCE_SHA256 == predecessor.BACKWARD_SOURCE_SHA256
    changed = {"reference_sha256s", "reference_assignment", "assignment_inputs", "unused_v29_reference"}
    for key in predecessor.SPECIFICATION.keys() - changed:
        assert retention.SPECIFICATION[key] == predecessor.SPECIFICATION[key]
    assert set(retention.REFERENCE_SHA256S) == {"v8", "general"}
    assert retention.REFERENCE_SHA256S["v8"] == predecessor.REFERENCE_SHA256S["v8"]
    assert retention.REFERENCE_SHA256S["general"] == "c0ddf75274c8ff92ca5fd22c1263369ffa7e55eb7cb85fe80961b0afa101f7a9"
    assert retention.RUNTIME_SOURCE == predecessor.RUNTIME_SOURCE.replace(
        'references["v8"] is references["v29"]', 'references["v8"] is references["general"]')


def test_actual_loss_and_runtime_bind_only_new_globals():
    for name in ("_backward_episode", "runtime", "retention_weighting", "validate_identities", "make_references"):
        function = inspect.unwrap(getattr(retention, name))
        assert function.__globals__ is vars(retention) and function.__closure__ is None
        assert unresolved_globals(function) == []


@pytest.fixture
def runtime_fixture(monkeypatch):
    monkeypatch.setattr(cohort_tests, "retention", retention)
    references, identities, backend, calls = cohort_tests.runtime_fixture.__wrapped__(monkeypatch)
    references["general"] = references.pop("v29")
    return references, identities, backend, calls


def test_runtime_saves_correct_full_targets_without_rating_reference_inference(tmp_path, runtime_fixture):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    student = torch.nn.Linear(2, 1)
    metadata = [original_tests.metadata(),
        original_tests.metadata("low_rank_mcar", structured=True, family="mar"),
        original_tests.metadata("slow_tail_mcar", structured=True, family="mnar_skip_logic"),
        original_tests.metadata(domain="recsys")]
    with retention.runtime(backend, tmp_path / "references", references, identities, query_seed_offset=5):
        for index, fields in enumerate(metadata):
            episode = original_tests.episode()
            episode.metadata = {**fields, "seed": 1000 + index}
            assert backend.backward_episode(student, None, episode, episode.target, .25) == (7. if index == 3 else 3.)
    assert backend.backward_episode is callback
    assert [entry[1] for entry in calls if entry[0] == "predict"] == ["general", "v8", "v8"]
    assert sum(entry[0] == "original" for entry in calls) == 1
    manifest = json.loads((tmp_path / "references/manifest.json").read_text())
    assert manifest["complete"] and set(manifest["reference"]) == {"v8", "general"}
    assert [row["reference_role"] for row in manifest["records"]] == ["general", "v8", "v8", None]
    for row in manifest["records"]:
        assert row["weighting"] == retention.retention_weighting(row["metadata"])
        if row["eligible"]:
            payload = torch.load(row["artifact"], weights_only=True)
            assert payload["identity"] == identities[row["reference_role"]]
            assert payload["reference_role"] == row["reference_role"]
            assert payload["prediction"].shape == payload["source"].shape
            assert payload["mask"].dtype == torch.bool
        else:
            assert "artifact" not in row


def test_reference_mutation_restores_callback_and_records_failure(tmp_path, runtime_fixture):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    with pytest.raises(ValueError, match="bitwise unchanged"):
        with retention.runtime(backend, tmp_path / "references", references, identities, query_seed_offset=5):
            references["general"].weight.add_(1)
    assert backend.backward_episode is callback
    assert not json.loads((tmp_path / "references/manifest.json").read_text())["complete"]


for name in ("test_legacy_task_weighting_is_exactly_unchanged", "test_ratings_acquire_no_retention_weight",
             "test_invalid_cohort_metadata_cannot_choose_a_weight",
             "test_structured_callback_changes_only_retention_and_matches_dense_gradient",
             "test_legacy_callback_dense_gradients_stay_unchanged",
             "test_real_streamed_network_legacy_gradients_stay_unchanged",
             "test_real_streamed_structured_model_matches_all_dense_parameter_gradients",
             "test_task_mixture_cannot_drop_or_reweight_families"):
    function = getattr(weighting_tests, name)
    copied = types.FunctionType(function.__code__, {**function.__globals__, "retention": retention}, name, function.__defaults__)
    copied.__dict__.update(function.__dict__)
    copied.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = copied
