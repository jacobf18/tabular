from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

import dwip_bpr_experiment as experiment
from tabimpute.prior.recsys_rating_episode import context_crop_indices, sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import synthetic_ranking_objective


def episode_arguments():
    return dict(seed=251, rows=24, cols=16, parent_rows=64, parent_cols=96, query_limit=50,
                density_range=(.15, .15), normalization="two_way_user_scale", point_target="expected_rating",
                ordinal_targets=True, exclude_unknown_queries=True, ranking_query_mode="user_catalog",
                ranking_user_scope="parent_positive", relevance_targets=True, ranking_max_users=8,
                negatives_per_user=16)


@pytest.mark.parametrize("ordering", ["iid", "first_observation", "first_observation_catalog"])
def test_default_pairing_setting_preserves_all_episode_tensors(ordering):
    arguments = {**episode_arguments(), "context_ordering": ordering}
    original = sample_rating_episode(**arguments)
    explicit = sample_rating_episode(**arguments, paired_window_cols=0)
    for field in fields(original):
        value = getattr(original, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(explicit, field.name), atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("width", [0, -1, 3, 4.0])
def test_pairing_rejects_invalid_window_geometry(width):
    with pytest.raises(ValueError, match="two equal-width"):
        context_crop_indices(torch.ones(12, 24, dtype=torch.bool), 6, 8, torch.Generator().manual_seed(1),
                             "first_observation_paired", paired_window_cols=width)


def test_paired_windows_preserve_anchor_and_use_disjoint_contiguous_blocks():
    observed = torch.ones(12, 24, dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[3, 4] = True
    observed[query] = False
    generator = torch.Generator().manual_seed(2)
    arrival = torch.rand(observed.shape, generator=generator).masked_fill(~observed, torch.inf)
    column_order = arrival.amin(0).argsort(stable=True)
    positions = torch.empty_like(column_order).scatter_(0, column_order, torch.arange(24))
    rows, columns = context_crop_indices(observed, 6, 8, torch.Generator().manual_seed(2),
        "first_observation_paired", query, paired_window_cols=4)
    assert 3 in rows and 4 in columns[:4]
    assert columns.unique().numel() == 8
    for window in columns.split(4):
        selected = positions[window]
        assert torch.equal(selected, torch.arange(selected[0], selected[0] + 4))
    repeated = context_crop_indices(observed, 6, 8, torch.Generator().manual_seed(2),
        "first_observation_paired", query, paired_window_cols=4)
    assert torch.equal(rows, repeated[0]) and torch.equal(columns, repeated[1])


def deterministic_arrival(shape, **kwargs):
    return torch.arange(shape[1], device=kwargs.get("device")).float()[None].expand(shape)


def test_partners_cover_distant_windows_instead_of_following_query_mass(monkeypatch):
    monkeypatch.setattr(torch, "rand", deterministic_arrival)
    observed = torch.ones(12, 16, dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[3, 0] = True
    observed[query] = False
    frequencies = torch.zeros(4, dtype=torch.long)
    for seed in range(360):
        _, columns = context_crop_indices(observed, 6, 8, torch.Generator().manual_seed(seed),
            "first_observation_paired", query, paired_window_cols=4)
        assert torch.equal(columns[:4], torch.arange(4))
        frequencies[columns[4] // 4] += 1
    assert frequencies[0] == 0
    assert torch.all((frequencies[1:] > 80) & (frequencies[1:] < 160))


def test_overlapping_tail_windows_cannot_be_paired_together(monkeypatch):
    monkeypatch.setattr(torch, "rand", deterministic_arrival)
    observed = torch.ones(12, 14, dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[3, 9] = True
    observed[query] = False
    for seed in range(30):
        _, columns = context_crop_indices(observed, 6, 8, torch.Generator().manual_seed(seed),
            "first_observation_paired", query, paired_window_cols=4)
        assert torch.equal(columns[:4], torch.arange(8, 12))
        assert columns[4].item() in (0, 4)
        assert columns.unique().numel() == 8


def test_paired_episode_keeps_parent_user_normalization_and_rating_queries():
    original = sample_rating_episode(**episode_arguments(), context_ordering="first_observation")
    paired = sample_rating_episode(**episode_arguments(), context_ordering="first_observation_paired", paired_window_cols=8)
    for name in ("row_offsets", "row_scales", "global_row_support", "global_query_counts"):
        torch.testing.assert_close(getattr(original, name), getattr(paired, name), atol=0, rtol=0)
    assert paired.rating_query.any()
    assert paired.metadata["paired_window_cols"] == 8
    assert paired.metadata["gradient_data"] == "synthetic"
    assert not torch.equal(torch.nan_to_num(original.source), torch.nan_to_num(paired.source))
    assert not paired.relevance_probabilities.requires_grad


@pytest.mark.parametrize("fraction,expected", [(0., "first_observation_paired"), (1., "first_observation_catalog")])
def test_training_wrapper_only_replaces_contiguous_component(monkeypatch, fraction, expected):
    captured = {}

    def generate(args, step, **kwargs):
        captured.update(vars(args))
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(experiment, "training_episode", generate)
    args = SimpleNamespace(seed=251, rows=24, cols=8, parent_cols=96, context_ordering="first_observation",
        training_item_windows=2, ranking_query_mode="user_catalog", negatives_per_user=16,
        catalog_fraction=fraction, paired_context_windows=True)
    episode = experiment.ranking_episode(args, 1)
    assert captured["context_ordering"] == expected
    assert captured["cols"] == 16
    assert captured.get("paired_window_cols", 0) == (8 if fraction == 0 else 0)
    assert episode.metadata["training_window_cols"] == 8


@pytest.mark.parametrize("windows,ordering", [(1, "first_observation"), (3, "first_observation"), (2, "iid")])
def test_training_wrapper_rejects_incompatible_pairing(windows, ordering):
    args = SimpleNamespace(catalog_fraction=0., training_item_windows=windows,
                           context_ordering=ordering, paired_context_windows=True)
    with pytest.raises(ValueError, match="two windows"):
        experiment.ranking_episode(args, 1)


@pytest.mark.parametrize("centered", [False, True])
def test_paired_context_objective_is_finite_and_synthetic_only(centered):
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(**episode_arguments(), context_ordering="first_observation_paired", paired_window_cols=8)
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    arguments = dict(ranking_objective="topk_expected_pairwise", negative_only_ranking=centered,
                     center_negative_only_ranking=centered, ranking_boundary_weight=1)
    loss, _ = synthetic_ranking_objective(logits, episode, Criterion(), **arguments)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    episode.metadata["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        synthetic_ranking_objective(logits, episode, Criterion(), **arguments)
