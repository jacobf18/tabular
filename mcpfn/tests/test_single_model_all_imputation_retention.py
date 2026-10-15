import pytest
import torch

import single_model_all_imputation_retention as retention
import single_model_structured_mixture as mixture
import test_single_model_anchor_retention as original_tests
import test_single_model_column_anchor_retention as column_tests
import test_single_model_shrunk_column_anchor_retention as shrunk_tests


@pytest.mark.parametrize("component", range(8))
def test_every_original_legacy_component_is_eligible(component):
    prior = "anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar"
    metadata = original_tests.metadata(component, prior=prior)
    assert retention.eligible(metadata)
    assert retention.predecessor.eligible(metadata) is (component in (6, 7))


@pytest.mark.parametrize("component", mixture.COMPONENTS)
@pytest.mark.parametrize("family", mixture.FAMILIES)
def test_every_structured_prior_and_pattern_is_eligible_without_error_selection(component, family):
    metadata = original_tests.metadata(component, structured=True, family=family, error=1e9,
        query_count=0, target_variance=0.)
    assert retention.eligible(metadata)
    assert retention.predecessor.eligible(metadata) is (component == "anchor_latent_mcar")


@pytest.mark.parametrize("changes", [dict(gradient_data="real"), dict(seed=True), dict(seed=-1),
    dict(domain="unknown"), dict(structured=1), dict(component=True), dict(component=8),
    dict(prior="wrong"), dict(structured=True, component="new_prior"), dict(domain="recsys", structured=True)])
def test_original_metadata_validation_still_fails_closed(changes):
    with pytest.raises(ValueError):
        retention.eligible(original_tests.metadata(**changes))


def test_original_shrinkage_and_independent_penalty_are_bitwise_unchanged():
    target, mask = column_tests.column_example()
    prediction = torch.nan_to_num(target) + .3
    reference = torch.nan_to_num(target) - .4
    assert torch.equal(retention.column_coefficients(target, mask),
        retention.predecessor.column_coefficients(target, mask))
    assert retention.independent_penalty(target, mask, prediction, reference) == (
        retention.predecessor.independent_penalty(target, mask, prediction, reference))


def test_exact_shrinkage_formula_keeps_empty_single_and_constant_columns(monkeypatch):
    monkeypatch.setattr(shrunk_tests, "retention", retention)
    shrunk_tests.test_exact_shrinkage_formula_keeps_empty_single_and_constant_columns()


def test_column_permutation_and_intentional_count_dependence_remain_unchanged(monkeypatch):
    monkeypatch.setattr(shrunk_tests, "retention", retention)
    shrunk_tests.test_permutation_invariant_but_replication_intentionally_changes_count_shrinkage()


@pytest.mark.parametrize("mutation", ["shape", "batch", "mask_dtype", "mask_shape", "target_dtype", "empty", "nonfinite"])
def test_invalid_query_inputs_never_drop_columns(monkeypatch, mutation):
    monkeypatch.setattr(column_tests, "retention", retention)
    column_tests.test_invalid_column_inputs_fail_without_omission(mutation)


def episode_factory(component, structured):
    original = original_tests.episode

    def make_episode():
        episode = original()
        prior = "anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar"
        episode.metadata = original_tests.metadata(component, structured=structured, prior=prior, family="mar")
        return episode

    return make_episode


@pytest.mark.parametrize("component,structured", [(0, False), (6, False), ("low_rank_mcar", True)])
@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize("weight", [.25, 1.])
def test_native_and_spectral_callbacks_match_dense_gradient_and_detach_targets(
        monkeypatch, component, structured, matched, weight):
    monkeypatch.setattr(shrunk_tests, "retention", retention)
    monkeypatch.setattr(original_tests, "episode", episode_factory(component, structured))
    shrunk_tests.test_callback_matches_independent_dense_gradient_with_detached_targets(monkeypatch, matched, weight)


@pytest.mark.parametrize("transposed", [False, True])
def test_streamed_full_network_and_every_gradient_match_dense(monkeypatch, transposed):
    monkeypatch.setattr(column_tests, "retention", retention)
    column_tests.test_full_streamed_network_matches_dense_loss_and_all_parameter_gradients(monkeypatch, transposed)


@pytest.mark.parametrize("component,structured", [(0, False), (7, False),
    ("low_rank_mcar", True), ("slow_tail_mcar", True), ("steep_tail_mcar", True), ("anchor_latent_mcar", True)])
@pytest.mark.parametrize("failure", [False, True])
def test_full_native_and_spectral_reference_cache_and_runtime_restoration(
        tmp_path, monkeypatch, component, structured, failure):
    monkeypatch.setattr(original_tests, "retention", retention)
    monkeypatch.setattr(original_tests, "episode", episode_factory(component, structured))
    original_tests.test_runtime_keeps_original_teacher_saves_all_tasks_and_restores_callback(tmp_path, monkeypatch, failure)


def test_rating_callback_is_exact_original_without_reference_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(original_tests, "retention", retention)
    original_tests.test_nonanchor_runtime_uses_exact_original_callback_without_reference_inference(
        tmp_path, monkeypatch, "recsys", 6)


def test_reference_identity_failure_cannot_create_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(original_tests, "retention", retention)
    original_tests.test_reference_identity_failure_does_not_create_artifacts_or_change_callback(tmp_path, monkeypatch)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nonfinite", "empty"])
def test_invalid_references_are_rejected(monkeypatch, mutation):
    monkeypatch.setattr(original_tests, "retention", retention)
    original_tests.test_invalid_reference_never_drops_queries(mutation)


def test_ratings_never_enter_the_additional_backward_callback():
    episode = original_tests.episode()
    episode.metadata["domain"] = "recsys"
    with pytest.raises(ValueError):
        retention.validate_target(episode, episode.target)


def test_only_retention_scope_changes_and_all_runtime_globals_are_local():
    excluded = {"legacy_components", "structured_component"}
    for key, value in retention.predecessor.SPECIFICATION.items():
        if key not in excluded:
            assert retention.SPECIFICATION[key] == value
    assert retention.SPECIFICATION["legacy_components"] == list(range(8))
    assert retention.SPECIFICATION["structured_components"] == list(mixture.COMPONENTS)
    assert retention.predecessor.SPECIFICATION["legacy_components"] == [6, 7]
    assert retention.BACKWARD_SOURCE == retention.predecessor.BACKWARD_SOURCE
    assert retention.RUNTIME_SOURCE == retention.predecessor.RUNTIME_SOURCE
    assert retention.BACKWARD_SOURCE_SHA256 == retention.predecessor.BACKWARD_SOURCE_SHA256
    for function in (retention.validate_target, retention.column_coefficients, retention.independent_penalty,
                     retention._backward_episode, retention.runtime.__wrapped__):
        assert function.__globals__ is vars(retention)
    assert not torch.cuda.is_initialized()
