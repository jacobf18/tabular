from types import SimpleNamespace

import pytest
import torch

import dwip_bpr_experiment as experiment
from dwip_bpr_window_training import windowed_training_logits
from tabimpute.prior.recsys_ranking_objective import topk_expected_pairwise_loss


class ContextMeanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.))
        self.shapes = []

    def forward(self, source, decode_mask):
        self.shapes.append(tuple(source.shape))
        return (self.scale * (source + source.mean(-1, keepdim=True)))[decode_mask, None]


def test_joint_windows_restore_query_order_and_accumulate_shared_gradients():
    model = ContextMeanModel()
    source = torch.arange(12.).reshape(1, 3, 4)
    query = torch.tensor([[[True, False, False, True], [False, True, True, False], [True, True, False, True]]])
    logits = windowed_training_logits(model, source, query, window_cols=2)
    expected = torch.empty_like(source)
    for start in (0, 2):
        window = source[:, :, start:start + 2]
        expected[:, :, start:start + 2] = 2 * (window + window.mean(-1, keepdim=True))
    torch.testing.assert_close(logits[:, 0], expected[query])
    logits.sum().backward()
    torch.testing.assert_close(model.scale.grad, expected[query].sum() / 2)
    assert model.shapes == [(1, 3, 2), (1, 3, 2)]


def test_empty_query_window_is_skipped_and_partial_tail_keeps_alignment():
    model = ContextMeanModel()
    source = torch.arange(10.).reshape(1, 2, 5)
    query = torch.zeros_like(source, dtype=torch.bool)
    query[:, :, 4] = True
    logits = windowed_training_logits(model, source, query, window_cols=2)
    torch.testing.assert_close(logits[:, 0], 4 * source[query])
    assert model.shapes == [(1, 2, 1)]


@pytest.mark.parametrize("window_cols", [0, 4, 8])
def test_one_window_is_exactly_the_original_forward(window_cols):
    model = ContextMeanModel()
    source = torch.arange(8.).reshape(1, 2, 4)
    query = torch.ones_like(source, dtype=torch.bool)
    actual = windowed_training_logits(model, source, query, window_cols=window_cols)
    original = model(source, decode_mask=query)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert model.shapes == [(1, 2, 4), (1, 2, 4)]


@pytest.mark.parametrize("candidate_scope", ["all", "topk_union"])
def test_joint_loss_provides_cross_window_offset_gradients(candidate_scope):
    offsets = torch.nn.Parameter(torch.tensor([0., 2.]))
    prediction = torch.tensor([3., 2., 3., 2.]) + offsets.repeat_interleave(2)
    loss, _ = topk_expected_pairwise_loss(prediction, torch.tensor([.8, .6, .1, .05]),
        torch.zeros(4, dtype=torch.long), torch.arange(4), torch.tensor([True, False, False, False]), (1, 4), top_k=1,
        candidate_scope=candidate_scope)
    loss.backward()
    assert offsets.grad[0] < 0 and offsets.grad[1] > 0
    torch.testing.assert_close(offsets.grad.sum(), torch.tensor(0.), atol=1e-7, rtol=0)


@pytest.mark.parametrize("step", [1, 2])
def test_joint_episode_preserves_user_axis_and_single_parent_normalization(monkeypatch, step):
    captured = {}

    def sample(arguments, update, **kwargs):
        captured.update({"rows": arguments.rows, "cols": arguments.cols, "step": update, **kwargs})
        return SimpleNamespace(metadata={"gradient_data": "synthetic"})

    monkeypatch.setattr(experiment, "training_episode", sample)
    arguments = SimpleNamespace(seed=61, catalog_fraction=0, training_item_windows=2, cols=1000, rows=1000,
                                parent_cols=4096, ranking_query_mode="user_catalog", negatives_per_user=2000)
    episode = experiment.ranking_episode(arguments, step)
    assert captured == {"rows": 1000, "cols": 2000, "step": step, "preserve_user_item_axes": True}
    assert episode.metadata["training_item_windows"] == 2 and episode.metadata["training_window_cols"] == 1000
    assert arguments.cols == 1000


def test_joint_windows_refuse_incomplete_catalogs():
    arguments = SimpleNamespace(seed=61, catalog_fraction=0, training_item_windows=2, cols=1000, rows=1000,
                                parent_cols=4096, ranking_query_mode="user_catalog", negatives_per_user=1000)
    with pytest.raises(ValueError, match="complete"):
        experiment.ranking_episode(arguments, 1)
