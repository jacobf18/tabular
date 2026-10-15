import pytest
import torch

from tabimpute.prior.recsys_clustered_factors import clustered_affinity, clustered_vectors
from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from preflight_dwip_recsys_training import preflight_preference_family


def test_clustered_affinity_is_seeded_finite_and_keeps_global_rng_untouched():
    state = torch.get_rng_state().clone()
    first, metadata = clustered_affinity(32, 24, 8, torch.Generator().manual_seed(61))
    second, repeated = clustered_affinity(32, 24, 8, torch.Generator().manual_seed(61))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert metadata == repeated
    assert first.shape == (32, 24) and torch.isfinite(first).all()
    assert 2 <= metadata["user_clusters"] <= 16
    assert 2 <= metadata["item_clusters"] <= 16
    assert .5 <= metadata["cluster_strength"] <= .95
    assert torch.equal(state, torch.get_rng_state())


def test_unit_strength_has_identical_vectors_within_clusters():
    vectors, assignments = clustered_vectors(512, 8, 5, 1.0, torch.Generator().manual_seed(62))
    for cluster in range(5):
        selected = vectors[assignments == cluster]
        assert len(selected) > 1
        torch.testing.assert_close(selected, selected[:1].expand_as(selected), rtol=0, atol=0)


def test_cluster_strength_controls_neighbor_similarity_without_variance_inflation():
    unstructured, labels = clustered_vectors(4096, 64, 8, 0., torch.Generator().manual_seed(63))
    structured, repeated = clustered_vectors(4096, 64, 8, .9, torch.Generator().manual_seed(63))
    assert torch.equal(labels, repeated)
    for vectors in (unstructured, structured):
        assert vectors.square().mean().item() == pytest.approx(1, abs=.2)
    unstructured_scatter = torch.stack([(unstructured[labels == cluster] - unstructured[labels == cluster].mean(0)).square().mean()
                                      for cluster in range(8)]).mean()
    structured_scatter = torch.stack([(structured[labels == cluster] - structured[labels == cluster].mean(0)).square().mean()
                                    for cluster in range(8)]).mean()
    assert structured_scatter < .15 * unstructured_scatter


@pytest.mark.parametrize("clusters,strength", [(1, .5), (4, -.1), (4, 1.1)])
def test_invalid_cluster_configuration_rejected(clusters, strength):
    with pytest.raises(ValueError, match="Require positive"):
        clustered_vectors(10, 3, clusters, strength, torch.Generator().manual_seed(64))


def test_default_gaussian_sampler_matches_explicit_legacy_options():
    implicit = sample_rating_episode(65, 24, 20, parent_rows=48)
    explicit = sample_rating_episode(65, 24, 20, parent_rows=48, preference_family="gaussian", exclude_unknown_queries=False)
    for name in ("source", "target", "query", "rating_query", "means", "stds"):
        torch.testing.assert_close(getattr(implicit, name), getattr(explicit, name), equal_nan=True, rtol=0, atol=0)
    assert implicit.metadata == explicit.metadata


def test_clustered_episode_keeps_synthetic_ordinal_probability_targets():
    episode = sample_rating_episode(66, 32, 24, parent_rows=64, preference_family="clustered",
                                     point_target="expected_rating", ordinal_targets=True, density_range=(.1, .1))
    assert episode.metadata["gradient_data"] == "synthetic"
    assert episode.metadata["preference_family"] == "clustered"
    assert torch.isnan(episode.source[episode.query]).all()
    probabilities = episode.ordinal_probabilities[episode.query]
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(len(probabilities)))
    torch.testing.assert_close((probabilities * torch.arange(1, 6)).sum(-1), episode.point_target[episode.query])


def test_mixed_family_samples_both_synthetic_preference_structures():
    families = {sample_rating_episode(seed, 16, 12, parent_rows=24, preference_family="mixed").metadata["preference_family"]
                for seed in range(10)}
    assert families == {"gaussian", "clustered"}


def test_known_only_queries_exclude_global_unknown_entities_but_not_local_coldness():
    unfiltered = sample_rating_episode(67, 48, 40, parent_rows=48, query_limit=4096, density_range=(.03, .03),
                                        normalization="global", context_ordering="first_observation")
    filtered = sample_rating_episode(67, 48, 40, parent_rows=48, query_limit=4096, density_range=(.03, .03),
                                      normalization="global", context_ordering="first_observation", exclude_unknown_queries=True)
    torch.testing.assert_close(unfiltered.source, filtered.source, equal_nan=True, rtol=0, atol=0)
    assert unfiltered.metadata["global_unknown_rating_queries"] > 0
    assert filtered.metadata["global_unknown_rating_queries"] == 0
    known = torch.isfinite(filtered.source).any(1)[:, None, :] & torch.isfinite(filtered.source).any(2)[:, :, None]
    assert not (filtered.query & ~known).any()
    assert filtered.rating_query.any()
    assert torch.isnan(filtered.source[filtered.query]).all()


def test_unknown_filter_uses_parent_support_not_only_crop_support():
    local_cold_queries = 0
    for seed in range(5):
        episode = sample_rating_episode(seed, 24, 20, parent_rows=96, parent_cols=80,
                                          density_range=(.03, .03), exclude_unknown_queries=True)
        assert episode.metadata["global_unknown_rating_queries"] == 0
        local_known = torch.isfinite(episode.source).any(1)[:, None, :] & torch.isfinite(episode.source).any(2)[:, :, None]
        local_cold_queries += int((episode.rating_query & ~local_known).sum())
    assert local_cold_queries > 0


def test_mixed_preflight_guarantees_both_families_instead_of_random_coverage():
    assert [preflight_preference_family("mixed", step) for step in range(4)] == ["gaussian", "clustered"] * 2
    for family in ("gaussian", "clustered"):
        assert all(preflight_preference_family(family, step) == family for step in range(4))


def test_global_support_fields_match_full_parent_and_bound_local_context():
    for parent_rows, parent_cols in ((24, 20), (96, 80)):
        episode = sample_rating_episode(68, 24, 20, parent_rows=parent_rows, parent_cols=parent_cols,
                                         density_range=(.1, .1), context_ordering="first_observation")
        support = torch.isfinite(episode.source[0])
        assert torch.all(episode.global_row_support >= support.sum(1))
        assert torch.all(episode.global_col_support >= support.sum(0))
        if (parent_rows, parent_cols) == (24, 20):
            assert torch.equal(episode.global_row_support, support.sum(1))
            assert torch.equal(episode.global_col_support, support.sum(0))
