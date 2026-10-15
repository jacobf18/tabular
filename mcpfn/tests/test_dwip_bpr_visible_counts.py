import pytest
import torch

from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import synthetic_ranking_objective


def arguments():
    return dict(seed=318, rows=24, cols=32, parent_rows=64, parent_cols=96,
                density_range=(.15, .15), normalization="two_way_user_scale", query_limit=96,
                point_target="expected_rating", ordinal_targets=True, preference_family="mixed",
                exclude_unknown_queries=True, context_ordering="first_observation_catalog",
                ranking_query_mode="user_catalog", negatives_per_user=32, ranking_max_users=8,
                relevance_targets=True, ranking_user_scope="parent_positive")


@pytest.mark.parametrize("ordering", ["first_observation", "first_observation_catalog"])
@pytest.mark.parametrize("normalization", ["column", "two_way", "two_way_user_scale"])
def test_visible_frequency_teacher_is_exactly_the_encoded_synthetic_mask(ordering, normalization):
    options = {**arguments(), "context_ordering": ordering, "normalization": normalization}
    episode = sample_rating_episode(**options, relevance_target="visible_item_frequency")
    expected = torch.isfinite(episode.source).float().mean(1, keepdim=True).expand_as(episode.source)
    torch.testing.assert_close(episode.relevance_probabilities, expected, rtol=0, atol=0)
    assert not episode.relevance_probabilities.requires_grad
    assert episode.metadata["gradient_data"] == "synthetic"
    assert episode.metadata["relevance_target"] == "visible_item_frequency"
    assert episode.metadata["relevance_targets"] is True
    assert ((expected >= 0) & (expected <= 1)).all()


def test_visible_teacher_changes_only_ranking_targets_not_sampling_or_rating_supervision():
    conditional = sample_rating_episode(**arguments(), relevance_target="conditional")
    distilled = sample_rating_episode(**arguments(), relevance_target="visible_item_frequency")
    for name, value in vars(conditional).items():
        if isinstance(value, torch.Tensor) and name != "relevance_probabilities":
            torch.testing.assert_close(getattr(distilled, name), value, equal_nan=True, rtol=0, atol=0)
    assert not torch.equal(conditional.relevance_probabilities, distilled.relevance_probabilities)
    assert {key: value for key, value in conditional.metadata.items() if key != "relevance_target"} == {
        key: value for key, value in distilled.metadata.items() if key != "relevance_target"}
    global_frequency = distilled.global_col_support.float() / distilled.metadata["parent_rows"]
    assert not torch.allclose(distilled.relevance_probabilities[0, 0], global_frequency)


def test_visible_teacher_is_independent_of_rating_normalization_values():
    first = sample_rating_episode(**arguments(), relevance_target="visible_item_frequency")
    second = sample_rating_episode(**{**arguments(), "normalization": "global"}, relevance_target="visible_item_frequency")
    assert torch.equal(torch.isfinite(first.source), torch.isfinite(second.source))
    assert torch.equal(first.relevance_probabilities, second.relevance_probabilities)
    assert not torch.allclose(first.source, second.source, equal_nan=True)


def test_visible_count_distillation_produces_finite_synthetic_gradients():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, targets):
            return logits[:, 0].square() * 0

    episode = sample_rating_episode(**arguments(), relevance_target="visible_item_frequency")
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    loss, diagnostics = synthetic_ranking_objective(logits, episode, Criterion(),
        ranking_objective="topk_expected_pairwise", temperature=.1, ranking_boundary_weight=1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert diagnostics["ranking_paired_users"] > 0
    assert diagnostics["visible_mask_teacher_verified"] is True
    episode.relevance_probabilities = episode.relevance_probabilities.clone() * .5
    with pytest.raises(ValueError, match="exactly match the encoded synthetic mask"):
        synthetic_ranking_objective(logits, episode, Criterion(), ranking_objective="topk_expected_pairwise")
    episode.metadata["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        synthetic_ranking_objective(logits, episode, Criterion(), ranking_objective="topk_expected_pairwise")


def test_visible_teacher_requires_explicit_synthetic_ranking_targets():
    with pytest.raises(ValueError, match="synthetic relevance targets"):
        sample_rating_episode(318, relevance_target="visible_item_frequency")
