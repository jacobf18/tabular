import inspect
import json
import types

import pytest
import torch

import single_model_uniform_v8_retention as retention
import single_model_family_balanced_retention as predecessor
import test_single_model_anchor_retention as original_tests
import test_single_model_family_balanced_retention as previous_tests


@pytest.mark.parametrize("family", retention.FAMILIES)
@pytest.mark.parametrize("component", ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"))
def test_every_original_family_and_component_uses_v8_without_changing_weights(family, component):
    metadata = original_tests.metadata(component, structured=True, family=family)
    assert retention.reference_role(metadata) == "v8"
    assert retention.retention_weighting(metadata) == predecessor.retention_weighting(metadata)
    changed = {**metadata, "seed": 4239, "validation_failure": True, "error": 1e30, "queries": 1}
    assert retention.reference_role(changed) == "v8"
    assert retention.retention_weighting(changed) == retention.retention_weighting(metadata)


def test_unmodified_loss_and_backward_sources_and_all_reference_identities():
    assert retention.BACKWARD_SOURCE == predecessor.BACKWARD_SOURCE
    assert retention.RUNTIME_SOURCE == predecessor.RUNTIME_SOURCE
    assert retention.REFERENCE_SHA256S == predecessor.REFERENCE_SHA256S
    for key in predecessor.SPECIFICATION.keys() - {"reference_assignment", "assignment_inputs"}:
        assert retention.SPECIFICATION[key] == predecessor.SPECIFICATION[key]
    assert retention.retention_weighting.__globals__["reference_role"] is retention.reference_role
    assert retention._backward_episode.__globals__["SPECIFICATION"] is retention.SPECIFICATION


def test_eligibility_and_retention_targets_do_not_add_rating_inference():
    metadata = original_tests.metadata(domain="recsys")
    assert retention.reference_role(metadata) is None
    assert not retention.eligible(metadata)
    assert retention.retention_weighting(metadata) == predecessor.retention_weighting(metadata)


fixture = previous_tests.runtime_fixture.__wrapped__
runtime_fixture = pytest.fixture(types.FunctionType(fixture.__code__,
    {**fixture.__globals__, "retention": retention}, fixture.__name__, fixture.__defaults__))


def test_runtime_uses_only_v8_targets_and_preserves_exact_rating_callback(tmp_path, runtime_fixture):
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
    assert [call[1] for call in calls if call[0] == "predict"] == ["v8"] * 3
    rating = [call for call in calls if call[0] == "original"]
    assert len(rating) == 1 and rating[0][2] == {"query_seed_offset": 5}
    manifest = json.loads((tmp_path / "references/manifest.json").read_text())
    assert manifest["complete"] is True and set(manifest["reference"]) == {"v8", "v29"}
    assert [record["reference_role"] for record in manifest["records"]] == ["v8", "v8", "v8", None]
    assert "artifact" not in manifest["records"][-1]
    for record in manifest["records"]:
        assert record["weighting"] == retention.retention_weighting(record["metadata"])
        if record["eligible"]:
            payload = torch.load(record["artifact"], weights_only=True)
            assert payload["identity"] == identities["v8"] and payload["reference_role"] == "v8"
            assert payload["weighting"] == record["weighting"]
            assert payload["mask"].dtype == torch.bool and torch.isnan(payload["source"][payload["mask"]]).all()
            assert payload["prediction"].shape == payload["source"].shape


def test_production_loss_and_runtime_use_only_the_new_namespace_without_closures():
    for name in ("_backward_episode", "runtime", "independent_penalty", "column_coefficients"):
        function = inspect.unwrap(getattr(retention, name))
        assert function.__globals__ is vars(retention) and function.__closure__ is None
    assert not torch.cuda.is_initialized()


for name, function in vars(previous_tests).items():
    if name.startswith("test_") and name not in {
            "test_every_family_receives_same_retention_mass_without_changing_reference",
            "test_runtime_keeps_exact_rating_callback_and_records_declared_weights",
            "test_no_closures_or_changes_to_original_reference_math"}:
        namespace = {**function.__globals__, "retention": retention}
        copied = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
        copied.__dict__.update(function.__dict__)
        copied.__kwdefaults__ = function.__kwdefaults__
        globals()[name] = copied
