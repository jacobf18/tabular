from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from tabimpute.prior import recsys_rating_episode as prior


def episode_arguments():
    return dict(seed=241, rows=24, cols=12, parent_rows=64, parent_cols=96,
                density_range=(.15, .15), ranking_query_mode="user_catalog", relevance_targets=True,
                ranking_max_users=0, negatives_per_user=100, query_limit=50, ordinal_targets=True,
                point_target="expected_rating", normalization="two_way_user_scale", exclude_unknown_queries=True)


def assert_episode_tensors_equal(original, changed, exceptions=()):
    for field in fields(original):
        value = getattr(original, field.name)
        if isinstance(value, torch.Tensor) and field.name not in exceptions:
            torch.testing.assert_close(value, getattr(changed, field.name), rtol=0, atol=0, equal_nan=True)


def test_default_ranking_user_scope_is_bitwise_compatible():
    arguments = episode_arguments()
    default = prior.sample_rating_episode(**arguments)
    explicit = prior.sample_rating_episode(**arguments, ranking_user_scope="local_positive")
    assert_episode_tensors_equal(default, explicit)
    assert default.metadata == explicit.metadata


def test_explicit_mask_ranks_users_without_local_positives_and_covers_catalog():
    rating_query = torch.tensor([[True, False, False], [False, False, False], [False, False, False]])
    ratings = torch.tensor([[5., 2., 3.], [2., 3., 1.], [3., 2., 2.]])
    eligible = torch.tensor([[True, True, False], [True, False, True], [True, True, True]])
    users = torch.tensor([True, True, False])
    selected = prior.matched_ranking_queries(rating_query, ratings, eligible, torch.Generator().manual_seed(1),
                                             100, user_mask=users)
    assert torch.equal(selected, eligible & users[:, None])
    legacy = prior.matched_ranking_queries(rating_query, ratings, eligible, torch.Generator().manual_seed(1), 100)
    assert legacy[0].any() and not legacy[1:].any()


def test_mask_excludes_empty_candidate_rows_before_user_cap():
    eligible = torch.zeros(6, 4, dtype=torch.bool)
    eligible[4:] = True
    selected = prior.matched_ranking_queries(torch.zeros_like(eligible), torch.ones(6, 4), eligible,
        torch.Generator().manual_seed(1), 10, max_users=2, user_mask=torch.ones(6, dtype=torch.bool))
    assert torch.equal(selected, eligible)


def test_explicit_user_cap_is_independent_of_local_ratings_and_queries():
    eligible = torch.ones(12, 4, dtype=torch.bool)
    users = torch.ones(12, dtype=torch.bool)
    positive = prior.matched_ranking_queries(eligible, torch.full((12, 4), 5.), eligible,
        torch.Generator().manual_seed(4), 10, max_users=3, user_mask=users)
    empty = prior.matched_ranking_queries(~eligible, torch.ones(12, 4), eligible,
        torch.Generator().manual_seed(4), 10, max_users=3, user_mask=users)
    assert torch.equal(positive, empty)
    assert empty.any(1).sum() == 3


@pytest.mark.parametrize("mask", [True, [True, True], torch.ones(2), torch.ones(2, 1, dtype=torch.bool),
                                  torch.ones(3, dtype=torch.bool), torch.ones(2, dtype=torch.bool, device="meta")])
def test_invalid_explicit_user_masks_are_rejected(mask):
    eligible = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="boolean ranking-user mask"):
        prior.matched_ranking_queries(eligible, torch.ones(2, 3), eligible,
                                     torch.Generator().manual_seed(1), 3, user_mask=mask)


@pytest.mark.parametrize("mode,relevance,scope", [("global", True, "parent_positive"),
    ("user_matched", True, "parent_positive"), ("user_catalog", False, "parent_positive"),
    ("user_catalog", True, "real_positive")])
def test_invalid_user_scope_combinations_are_rejected(mode, relevance, scope):
    arguments = {**episode_arguments(), "ranking_query_mode": mode, "relevance_targets": relevance}
    with pytest.raises(ValueError, match="synthetic"):
        prior.sample_rating_episode(**arguments, ranking_user_scope=scope)


def test_parent_scope_preserves_nonranking_tensors_and_includes_locally_empty_users():
    arguments = episode_arguments()
    local = prior.sample_rating_episode(**arguments)
    parent = prior.sample_rating_episode(**arguments, ranking_user_scope="parent_positive")
    assert_episode_tensors_equal(local, parent, exceptions=("query",))
    assert parent.metadata["ranking_users_without_local_positive"] > 0
    assert parent.metadata["ranking_users_without_local_context"] > 0
    assert local.metadata["ranking_users_without_sampled_positive"] == 0
    assert torch.all(parent.query[local.query])
    assert parent.metadata["gradient_data"] == "synthetic"
    assert not parent.relevance_probabilities.requires_grad


def test_parent_user_eligibility_ignores_rating_query_cap(monkeypatch):
    original = prior.sample_rating_episode(**episode_arguments(), ranking_user_scope="parent_positive")
    monkeypatch.setattr(prior, "cap_rating_queries", lambda query, *args: torch.zeros_like(query))
    uncapped_users = prior.sample_rating_episode(**episode_arguments(), ranking_user_scope="parent_positive")
    assert not uncapped_users.rating_query.any()
    assert uncapped_users.query.any()
    assert original.metadata["ranking_selected_users"] == uncapped_users.metadata["ranking_selected_users"]
    assert uncapped_users.metadata["ranking_users_without_sampled_positive"] == uncapped_users.metadata["ranking_selected_users"]


def test_full_parent_eligibility_matches_synthetic_known_holdout_positives():
    arguments = {**episode_arguments(), "rows": 64, "cols": 96, "query_limit": 10000}
    episode = prior.sample_rating_episode(**arguments, ranking_user_scope="parent_positive")
    raw = episode.target[0] * episode.row_scales[:, None] * episode.stds + episode.means + episode.row_offsets[:, None]
    expected = (episode.rating_query[0] & (raw.round() >= 4)).any(1)
    selected = (episode.query & ~episode.rating_query)[0].any(1)
    assert torch.equal(selected, expected)
    known = (episode.global_row_support > 0)[:, None] & (episode.global_col_support > 0)[None, :]
    candidates = torch.isnan(episode.source[0]) & known
    assert torch.equal(episode.query[0] & selected[:, None], candidates & selected[:, None])


def test_training_entrypoint_forwards_parent_scope_without_real_data(monkeypatch):
    import dwip_recsys_experiment as experiment

    captured = {}

    def sample(seed, **kwargs):
        captured.update(kwargs)
        return seed

    monkeypatch.setattr(experiment, "sample_rating_episode", sample)
    args = SimpleNamespace(seed=241, rows=24, cols=12, parent_rows=64, parent_cols=96,
        density_low=.15, density_high=.15, normalization="two_way_user_scale", point_target="expected_rating",
        item_popularity_low=.7, item_popularity_high=1.9, context_ordering="iid", query_sampling="user_balanced",
        mae_weight=.1, ranking_objective="topk_expected_pairwise", ranking_query_mode="user_catalog",
        ranking_user_scope="parent_positive")
    assert experiment.training_episode(args, 1) == 241000001
    assert captured["ranking_user_scope"] == "parent_positive"
    assert captured["relevance_targets"] is True
    assert captured["relevance_target"] == "conditional"


@pytest.mark.parametrize("negative_only,centered", [(False, False), (True, False), (True, True)])
def test_parent_scope_complete_objective_is_synthetic_only(negative_only, centered):
    from tabimpute.prior.recsys_ranking_objective import synthetic_ranking_objective

    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = prior.sample_rating_episode(**episode_arguments(), ranking_user_scope="parent_positive")
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    arguments = dict(ranking_objective="topk_expected_pairwise", negative_only_ranking=negative_only,
                     center_negative_only_ranking=centered)
    loss, diagnostics = synthetic_ranking_objective(logits, episode, Criterion(), **arguments)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert diagnostics["negative_only_ranking"] is negative_only
    assert diagnostics["center_negative_only_ranking"] is centered
    query_users = episode.query.nonzero()[:, 1]
    rating_users = episode.rating_query[0].any(1)
    locally_empty = ~rating_users[query_users]
    assert locally_empty.any() and logits.grad[locally_empty].abs().sum() > 0
    episode.metadata["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        synthetic_ranking_objective(logits, episode, Criterion(), **arguments)
