import numpy as np
import pytest
import torch

from dwip_recsys_experiment import inner_split, block_spans, selection_score, inference_scaling, inference_components
from tabimpute.prior.recsys_rating_episode import observed_scaling, sample_rating_episode, rating_objective, expected_ordinal_rating, fit_bias_normalization
from evaluate_dwip_recsys_candidate import remap_prediction, compare_metrics, verify_selection
from cornac.data import Dataset
import dwip_recsys_experiment as experiment


def test_inner_split_is_disjoint_reproducible_and_known_only():
    users = np.repeat(np.arange(10), 10)
    items = np.tile(np.arange(10), 10)
    ratings = np.full(100, 4.0)
    context, queries = inner_split(users, items, ratings)
    assert np.array_equal(queries, inner_split(users, items, ratings)[1])
    assert not np.intersect1d(context, queries).size
    assert np.isin(users[queries], users[context]).all()
    assert np.isin(items[queries], items[context]).all()


def test_scaling_ignores_unobserved_values_and_matches_population_std():
    observed = torch.tensor([[True, True], [True, False], [False, False]])
    ratings = torch.tensor([[1., 5.], [3., -999.], [999., 999.]])
    means, stds = observed_scaling(ratings, observed)
    assert torch.equal(means, torch.tensor([2., 5.]))
    assert torch.equal(stds, torch.tensor([1., 1.]))
    ratings[~observed] = 0
    assert torch.equal(observed_scaling(ratings, observed)[0], means)
    assert torch.equal(observed_scaling(ratings, observed)[1], stds)


def test_episode_has_hidden_exposed_queries_bounded_integer_ratings_and_global_scaling():
    episode = sample_rating_episode(42, rows=32, cols=24, parent_rows=100, query_limit=20, density_range=(.2, .2))
    repeated = sample_rating_episode(42, rows=32, cols=24, parent_rows=100, query_limit=20, density_range=(.2, .2))
    torch.testing.assert_close(episode.source, repeated.source, equal_nan=True)
    assert episode.rating_query.sum() <= 20
    assert episode.rating_query.any()
    assert torch.isnan(episode.source[episode.query]).all()
    assert (episode.query | ~episode.rating_query).all()
    raw_target = episode.target * episode.stds + episode.means
    torch.testing.assert_close(raw_target, raw_target.round())
    assert raw_target.min() >= 1 - 1e-5 and raw_target.max() <= 5 + 1e-5
    assert torch.isfinite(episode.target).all()


def test_block_spans_match_overlapping_tail_protocol():
    assert block_spans(943) == [(0, 943)]
    assert block_spans(1656) == [(0, 1000), (656, 1656)]


def test_selection_normalizes_each_dataset_against_mf():
    assert selection_score({"first": {"RMSE": .8}, "second": {"RMSE": 1.2}},
                           {"first": {"RMSE": 1.}, "second": {"RMSE": 1.}}) == pytest.approx(1.0)


def test_objective_backpropagates_without_exposing_query_targets():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(12, rows=32, cols=32, parent_rows=100, density_range=(.2, .2))
    logits = torch.zeros((episode.query.sum(), 1), requires_grad=True)
    loss, diagnostics = rating_objective(logits, episode, Criterion(), ranking_weight=.1)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert diagnostics["ranking"] > 0
    assert torch.isnan(episode.source[episode.query]).all()


def test_prediction_remapping_respects_cornac_inner_ids():
    train = Dataset.from_uir([(2, 3, 4.), (0, 1, 2.)], seed=123)
    matrix = np.arange(12).reshape(3, 4)
    mapped, users = remap_prediction(matrix, train)
    assert np.array_equal(users, [2, 0])
    assert np.array_equal(mapped, [[11, 9], [3, 1]])


def test_comparison_directions_and_ties():
    baseline = {metric: 1. for metric in ("MAE", "RMSE", "AUC", "MAP", "NDCG@10", "Precision@10", "Recall@10")}
    result = compare_metrics({**baseline, "MAE": .9, "AUC": 1.1}, baseline)
    assert result["MAE"]["better"] and result["AUC"]["better"]
    assert not result["RMSE"]["better"] and not result["MAP"]["better"]


def test_global_normalization_preserves_cold_item_rating_level_and_matches_inference():
    ratings = torch.tensor([[1., 5.], [3., 99.], [99., 99.]])
    observed = ratings != 99
    means, stds = observed_scaling(ratings, observed, "global")
    matrix = ratings.masked_fill(~observed, torch.nan).numpy()
    inference_means, inference_stds = inference_scaling(matrix, "global")
    np.testing.assert_allclose(inference_means, means.numpy())
    np.testing.assert_allclose(inference_stds, stds.numpy())
    assert ((ratings[0] - means) / stds)[0] < 0
    assert ((ratings[0] - means) / stds)[1] > 0
    ratings[~observed] = -999
    assert torch.equal(observed_scaling(ratings, observed, "global")[0], means)


@pytest.mark.parametrize("mode", ["column", "global", "shrunk_column"])
def test_empty_context_normalization_is_finite(mode):
    means, stds = observed_scaling(torch.ones(3, 4), torch.zeros(3, 4, dtype=torch.bool), mode)
    assert torch.equal(means, torch.zeros(4))
    assert torch.equal(stds, torch.ones(4))


def test_inner_loader_never_reads_outer_test_labels(monkeypatch):
    class TrainingOnlyArchive(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            assert not key.startswith("test_")
            return super().__getitem__(key)

    data = TrainingOnlyArchive(shape=np.array([10, 10]), train_users=np.repeat(np.arange(10), 10),
                               train_items=np.tile(np.arange(10), 10), train_ratings=np.full(100, 4.))
    monkeypatch.setattr(experiment.np, "load", lambda *args, **kwargs: data)
    observed, context, queries = experiment.load_split("unused", "validation")
    assert len(context[2]) == 90 and len(queries[2]) == 10
    assert np.isnan(observed[queries[0], queries[1]]).all()


def test_checkpoint_guard_accounts_for_atomic_temporary_bytes(tmp_path, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("Usage", "total used free")(100, 100, 18 * 1024**3)
    monkeypatch.setattr(experiment.shutil, "disk_usage", lambda path: usage)
    path = tmp_path / "model.pth"
    with pytest.raises(RuntimeError, match="Atomic checkpoint"):
        experiment.save_checkpoint(path, {"model": torch.ones(2)})
    assert not path.exists() and not path.with_suffix(".tmp").exists()


def test_checkpoint_guard_preserves_unexpected_temporary(tmp_path):
    path = tmp_path / "model.pth"
    path.with_suffix(".tmp").write_text("existing")
    with pytest.raises(ValueError, match="unexpected"):
        experiment.save_checkpoint(path, {"model": torch.ones(2)})
    assert path.with_suffix(".tmp").read_text() == "existing"


def test_shrunk_column_normalization_uses_global_context_without_cold_item_extremes():
    ratings = torch.tensor([[1., 5.], [3., 99.], [99., 99.]])
    observed = ratings != 99
    means, stds = observed_scaling(ratings, observed, "shrunk_column")
    assert 3 < means[1] < 5
    assert 1 < means[0] < 3
    matrix = ratings.masked_fill(~observed, torch.nan).numpy()
    inference_means, inference_stds = inference_scaling(matrix, "shrunk_column")
    np.testing.assert_allclose(inference_means, means.numpy())
    np.testing.assert_allclose(inference_stds, stds.numpy())


def test_expected_rating_matches_rounded_clipped_noise_expectation():
    preference = torch.tensor([.5, 1., 2., 3., 4., 5., 5.5])
    generator = torch.Generator().manual_seed(123)
    samples = (preference + .8 * torch.randn(100000, len(preference), generator=generator)).round().clamp(1, 5).mean(0)
    expected = expected_ordinal_rating(preference, .8)
    torch.testing.assert_close(expected, samples, atol=.01, rtol=0)
    assert expected[3] == 3
    assert torch.all((expected >= 1) & (expected <= 5))


def test_expected_target_changes_only_loss_target_not_context_or_queries():
    arguments = dict(seed=42, rows=32, cols=24, parent_rows=100, density_range=(.2, .2))
    observed = sample_rating_episode(**arguments)
    expected = sample_rating_episode(**arguments, point_target="expected_rating")
    torch.testing.assert_close(observed.source, expected.source, equal_nan=True)
    assert torch.equal(observed.target, expected.target)
    assert torch.equal(observed.query, expected.query)
    assert observed.point_target is None and expected.point_target.shape == expected.target.shape
    assert torch.isnan(expected.source[expected.query]).all()


def test_vectorized_ranking_matches_equal_user_reference():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(12, rows=32, cols=32, parent_rows=100, density_range=(.2, .2))
    logits = torch.linspace(-1, 1, int(episode.query.sum())).unsqueeze(1).requires_grad_()
    _, result = rating_objective(logits, episode, Criterion(), ranking_weight=.1)
    indices = episode.query.nonzero()
    rows, cols = indices[:, 1], indices[:, 2]
    prediction = logits[:, 0] * episode.stds[cols] + episode.means[cols]
    truth = episode.target[episode.query] * episode.stds[cols] + episode.means[cols]
    selected = episode.rating_query[episode.query]
    terms = []
    for row in rows.unique():
        positive = prediction[(rows == row) & selected & (truth.round() >= 4)]
        negative = prediction[(rows == row) & ~selected]
        if positive.numel() and negative.numel():
            terms.append(torch.nn.functional.softplus(negative[None, :] - positive[:, None]).mean())
    assert result["ranking"] == pytest.approx(torch.stack(terms).mean().item())


def test_two_way_normalization_matches_inference_and_ignores_hidden_values():
    generator = torch.Generator().manual_seed(42)
    ratings = torch.randint(1, 6, (20, 16), generator=generator).float()
    observed = torch.rand((20, 16), generator=generator) < .3
    components = fit_bias_normalization(ratings, observed)
    numpy_components = inference_components(ratings.masked_fill(~observed, torch.nan).numpy(), "two_way")
    for tensor, values in zip(components, numpy_components):
        np.testing.assert_allclose(tensor.numpy(), values, atol=1e-5, rtol=1e-5)
    ratings[~observed] = 999
    for actual, expected in zip(fit_bias_normalization(ratings, observed), components):
        assert torch.equal(actual, expected)


def test_two_way_episode_crops_both_axes_and_reconstructs_integer_targets():
    episode = sample_rating_episode(42, rows=24, cols=16, parent_rows=80, parent_cols=64,
                                    density_range=(.2, .2), normalization="two_way", point_target="expected_rating")
    assert episode.source.shape == (1, 24, 16)
    assert episode.metadata["parent_cols"] == 64
    assert episode.row_offsets.shape == (24,)
    assert torch.isnan(episode.source[episode.query]).all()
    raw = episode.target * episode.stds + episode.means + episode.row_offsets[None, :, None]
    torch.testing.assert_close(raw, raw.round(), atol=1e-5, rtol=0)
    assert raw.min() > .999 and raw.max() < 5.001
    assert episode.point_target.shape == raw.shape


def test_two_way_point_objective_restores_row_offsets_before_scoring():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(42, rows=24, cols=16, parent_rows=80, parent_cols=64,
                                    density_range=(.2, .2), normalization="two_way", point_target="expected_rating")
    indices = episode.query.nonzero()
    rows, cols = indices[:, 1], indices[:, 2]
    logits = ((episode.point_target[episode.query] - episode.means[cols] - episode.row_offsets[rows])
              / episode.stds[cols]).unsqueeze(1).requires_grad_()
    loss, _ = rating_objective(logits, episode, Criterion(), nll_weight=0)
    assert loss.item() == pytest.approx(0, abs=1e-9)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_initial_checkpoint_can_win_selection_and_nonminimal_selection_is_rejected(tmp_path):
    import json

    checkpoint = tmp_path / "best_validation_checkpoint.pth"
    torch.save({"model": {"weight": torch.ones(1)}, "step": 0}, checkpoint)
    (tmp_path / "status.json").write_text(json.dumps({"state": "complete", "best_score": .9}))
    (tmp_path / "validation_0.json").write_text(json.dumps({"step": 0, "score": .9, "improved": True}))
    (tmp_path / "validation_1000.json").write_text(json.dumps({"step": 1000, "score": .95, "improved": False}))
    assert verify_selection(checkpoint)["step"] == 0
    (tmp_path / "validation_1000.json").write_text(json.dumps({"step": 1000, "score": .89, "improved": True}))
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_selection(checkpoint)


def test_query_support_averages_overlapping_contexts_and_excludes_missing_queries():
    from diagnose_dwip_recsys_prior_fit import query_context_support

    observed = np.ones((4, 5))
    observed[1, 2] = np.nan
    rows, cols = query_context_support(observed, np.array([1]), np.array([2]), block=3)
    assert rows[0] == 2 and cols[0] == 2
    with pytest.raises(ValueError, match="outside"):
        query_context_support(observed, np.array([4]), np.array([2]), block=3)


def test_prior_diagnostic_weights_users_and_episodes_equally():
    from diagnose_dwip_recsys_prior_fit import user_weights, describe

    weights = user_weights(np.array([0, 1, 1, 1]))
    np.testing.assert_allclose(weights, [.5, 1/6, 1/6, 1/6])
    values = np.array([1., 10., 10., 10.])
    result = describe(values, values, weights)
    assert result["fraction_query_row_support_le2"] == pytest.approx(.5)
    assert describe(values, values)["fraction_query_row_support_le2"] == pytest.approx(.25)


def test_popularity_prior_defaults_preserve_reproducibility_and_validate_spread():
    arguments = dict(seed=42, rows=24, cols=16, parent_rows=80)
    default = sample_rating_episode(**arguments)
    explicit = sample_rating_episode(**arguments, item_popularity_sigma_range=(.7, 1.9))
    torch.testing.assert_close(default.source, explicit.source, equal_nan=True, rtol=0, atol=0)
    with pytest.raises(ValueError, match="item-popularity"):
        sample_rating_episode(**arguments, item_popularity_sigma_range=(2, 1))


def test_worst_rating_selection_cannot_hide_one_dataset_behind_another():
    baseline = {name: {"MAE": 1., "RMSE": 1.} for name in ("first", "second")}
    unbalanced = {"first": {"MAE": .8, "RMSE": .8}, "second": {"MAE": 1.05, "RMSE": 1.05}}
    all_winning = {name: {"MAE": .99, "RMSE": .99} for name in baseline}
    assert selection_score(unbalanced, baseline) < selection_score(all_winning, baseline)
    assert selection_score(unbalanced, baseline, "worst_rating") > 1
    assert selection_score(all_winning, baseline, "worst_rating") < 1
    with pytest.raises(ValueError, match="selection"):
        selection_score(all_winning, baseline, "unknown")


def test_context_ordering_uses_only_observation_mask_and_preserves_default_rng():
    from tabimpute.prior.recsys_rating_episode import context_crop_indices

    observed = torch.rand((40, 32), generator=torch.Generator().manual_seed(6)) < .2
    generator = torch.Generator().manual_seed(7)
    state = generator.get_state().clone()
    rows, cols = context_crop_indices(observed, 12, 8, generator)
    assert torch.equal(generator.get_state(), state)
    assert torch.equal(rows, torch.arange(12)) and torch.equal(cols, torch.arange(8))
    first = context_crop_indices(observed, 12, 8, generator, "first_observation")
    repeat = context_crop_indices(observed, 12, 8, torch.Generator().manual_seed(7), "first_observation")
    for actual, expected, width, maximum in zip(first, repeat, (12, 8), (40, 32)):
        assert torch.equal(actual, expected)
        assert actual.unique().numel() == width
        assert actual.min() >= 0 and actual.max() < maximum
    with pytest.raises(ValueError, match="ordering"):
        context_crop_indices(observed, 12, 8, generator, "unknown")


def test_ordered_crop_covers_a_known_user_heldout_anchor():
    from tabimpute.prior.recsys_rating_episode import context_crop_indices

    observed = torch.ones((40, 32), dtype=torch.bool)
    observed[35, 27] = False
    query = torch.zeros_like(observed)
    query[35, 27] = True
    rows, cols = context_crop_indices(observed, 12, 8, torch.Generator().manual_seed(7), "first_observation", query)
    assert 35 in rows and 27 in cols


@pytest.mark.parametrize("normalization", ["column", "global", "shrunk_column", "two_way"])
def test_ordered_episode_keeps_targets_and_context_aligned(normalization):
    episode = sample_rating_episode(51, rows=24, cols=16, parent_rows=80, parent_cols=64,
                                    density_range=(.3, .3), normalization=normalization,
                                    point_target="expected_rating", context_ordering="first_observation")
    assert episode.source.shape == (1, 24, 16)
    assert episode.metadata["context_ordering"] == "first_observation"
    assert torch.isnan(episode.source[episode.query]).all()
    observed = torch.isfinite(episode.source)
    assert torch.equal(episode.source[observed], episode.target[observed])
    raw = episode.target * episode.stds + episode.means
    if episode.row_offsets is not None:
        raw = raw + episode.row_offsets[None, :, None]
    torch.testing.assert_close(raw, raw.round(), atol=1e-5, rtol=0)
    assert raw.min() > .999 and raw.max() < 5.001
    assert episode.point_target.min() >= 1 and episode.point_target.max() <= 5


def test_activation_checkpointing_preserves_network_outputs_gradients_and_state_keys():
    import copy
    from tabimpute.model.model_new_stable import TabImputeModelStable

    torch.manual_seed(43)
    direct = TabImputeModelStable(embedding_size=32, num_attention_heads=4, mlp_hidden_size=48,
                                 num_layers=2, num_outputs=8, num_cls=2, attention_dropout=0,
                                 ffn_dropout=0, drop_path_rate=0, embedding_dropout=0).train()
    recomputed = copy.deepcopy(direct)
    source = torch.randn(1, 8, 6)
    query = torch.rand(1, 8, 6) < .3
    source[query] = torch.nan
    expected = experiment.training_logits(direct, source, query)
    actual = experiment.training_logits(recomputed, source, query, True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert direct.state_dict().keys() == recomputed.state_dict().keys()
    for original, repeated in zip(direct.parameters(), recomputed.parameters()):
        torch.testing.assert_close(original.grad, repeated.grad, rtol=0, atol=0)


def test_user_balanced_query_cap_keeps_sparse_users_and_is_reproducible():
    from tabimpute.prior.recsys_rating_episode import cap_rating_queries

    queries = torch.ones((8, 64), dtype=torch.bool)
    queries[4:, 1:] = False
    for limit in (4, 8, 20, 1000):
        actual = cap_rating_queries(queries, limit, torch.Generator().manual_seed(42), "user_balanced")
        repeat = cap_rating_queries(queries, limit, torch.Generator().manual_seed(42), "user_balanced")
        assert torch.equal(actual, repeat)
        assert not (actual & ~queries).any()
        assert actual.sum() == min(limit, queries.sum())
        assert actual.any(1).sum() == min(limit, queries.any(1).sum())
    with pytest.raises(ValueError, match="query sampling"):
        cap_rating_queries(queries, 8, torch.Generator(), "unknown")


def test_ema_preserves_optimizer_parameters_and_restores_weights_after_errors():
    model = torch.nn.Linear(2, 1)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    averaged = {name: torch.zeros_like(value) for name, value in original.items()}
    experiment.update_ema(averaged, model.state_dict(), .75)
    for name, value in averaged.items():
        torch.testing.assert_close(value, original[name] * .25)
    identities = [id(parameter) for parameter in model.parameters()]
    with experiment.candidate_parameters(model, averaged):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, averaged[name])
    assert [id(parameter) for parameter in model.parameters()] == identities
    with pytest.raises(RuntimeError, match="validation failure"):
        with experiment.candidate_parameters(model, averaged):
            raise RuntimeError("validation failure")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="EMA decay"):
        experiment.update_ema(averaged, model.state_dict(), 1.)


def test_ordinal_probabilities_are_valid_and_reproduce_expected_rating():
    from tabimpute.prior.recsys_rating_episode import ordinal_rating_probabilities

    preference = torch.linspace(-2, 8, 100)
    probabilities = ordinal_rating_probabilities(preference, .8)
    assert (probabilities >= 0).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(preference))
    torch.testing.assert_close((probabilities * torch.arange(1, 6)).sum(-1), expected_ordinal_rating(preference, .8))
    with pytest.raises(ValueError, match="positive rating noise"):
        ordinal_rating_probabilities(preference, 0)


def test_expected_mae_auxiliary_uses_raw_scale_and_equal_user_weights():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(57, rows=24, cols=16, parent_rows=80, parent_cols=64,
                                    density_range=(.3, .3), normalization="two_way", point_target="expected_rating",
                                    query_sampling="user_balanced", ordinal_targets=True)
    logits = torch.zeros((int(episode.query.sum()), 1), requires_grad=True)
    indices = episode.query.nonzero()
    prediction = episode.means[indices[:, 2]] + episode.row_offsets[indices[:, 1]]
    selected = episode.rating_query[episode.query]
    rows = indices[selected, 1]
    expected = ((prediction[selected, None] - torch.arange(1, 6)).abs()
                * episode.ordinal_probabilities[episode.query][selected]).sum(-1)
    manual = torch.stack([expected[rows == row].mean() for row in rows.unique()]).mean()
    original, _ = rating_objective(logits, episode, Criterion(), nll_weight=0)
    total, diagnostics = rating_objective(logits, episode, Criterion(), nll_weight=0, mae_weight=.1)
    torch.testing.assert_close(total - original, .1 * manual)
    assert diagnostics["expected_mae"] == pytest.approx(manual.item())
    total.backward()
    assert torch.isfinite(logits.grad).all()
