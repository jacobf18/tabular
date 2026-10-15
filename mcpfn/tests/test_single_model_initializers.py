from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

import screen_single_matrixformer_initializers as screening
from single_model_distillation import Episode, episode_metrics


def states():
    return {
        "general": {"encoder.weight": torch.tensor([1., 2.]), "decoder.weight": torch.tensor([3., 4.]), "count": torch.tensor(1)},
        "specialist": {"encoder.weight": torch.tensor([3., 6.]), "decoder.weight": torch.tensor([9., 12.]), "count": torch.tensor(1)},
        "shared": {"encoder.weight": torch.tensor([0., 1.]), "decoder.weight": torch.tensor([2., 3.]), "count": torch.tensor(1)},
        "pilot": {"encoder.weight": torch.tensor([2., 3.]), "decoder.weight": torch.tensor([4., 5.]), "count": torch.tensor(1)},
    }


def test_all_recipes_materialize_one_state_without_mutating_sources():
    originals = states()
    before = deepcopy(originals)
    screening.validate_states(originals)
    for kind in ("interpolation", "task_addition", "decoder_interpolation"):
        result = screening.materialize({"kind": kind, "alpha": .25}, originals)
        assert tuple(result) == tuple(originals["general"])
        for name in ("encoder.weight", "decoder.weight"):
            if kind == "task_addition":
                expected = originals["general"][name] + .25 * (originals["specialist"][name] - originals["shared"][name])
            elif kind == "decoder_interpolation" and name.startswith("encoder"):
                expected = originals["general"][name]
            else:
                expected = .75 * originals["general"][name] + .25 * originals["specialist"][name]
            torch.testing.assert_close(result[name], expected)
    for role in originals:
        for name in originals[role]:
            torch.testing.assert_close(originals[role][name], before[role][name])


def test_exact_endpoints_and_checkpoint_control_are_independent_copies():
    originals = states()
    for alpha, expected_role in ((0., "general"), (1., "specialist")):
        result = screening.materialize({"kind": "interpolation", "alpha": alpha}, originals)
        for name in result:
            assert torch.equal(result[name], originals[expected_role][name])
    copied = screening.materialize({"kind": "checkpoint", "checkpoint": "pilot"}, originals)
    copied["decoder.weight"].zero_()
    assert torch.equal(originals["pilot"]["decoder.weight"], torch.tensor([4., 5.]))


@pytest.mark.parametrize("alpha", (-.1, 1.1, float("nan")))
def test_bad_coefficients_are_rejected(alpha):
    with pytest.raises(ValueError, match="coefficient"):
        screening.materialize({"kind": "interpolation", "alpha": alpha}, states())


def test_inconsistent_buffers_and_shapes_fail_closed():
    originals = states()
    originals["specialist"]["count"] = torch.tensor(2)
    with pytest.raises(ValueError, match="nonfloating"):
        screening.materialize({"kind": "task_addition", "alpha": .5}, originals)
    originals["specialist"]["encoder.weight"] = torch.ones(3)
    with pytest.raises(ValueError, match="tensor"):
        screening.validate_states(originals)


def test_registered_bank_is_disjoint_and_covers_both_rating_context_sizes():
    protocol = screening.read(screening.DEFAULT_PROTOCOL)
    cases = screening.bank_specs(protocol)
    assert len(cases) == 28
    assert len({case["seed"] for case in cases}) == 28
    assert min(case["seed"] for case in cases) > 10100546
    ratings = [case for case in cases if case["domain"] == "recsys"]
    assert sum(case["shape"] == (1000, 1000) for case in ratings) == 4
    assert sum(case["shape"] == (2000, 1000) for case in ratings) == 4
    assert len(protocol["candidates"]) == 13
    assert protocol["candidates"][:2] == [
        {"name": "general", "kind": "checkpoint", "checkpoint": "general"},
        {"name": "pilot150", "kind": "checkpoint", "checkpoint": "pilot"},
    ]


def test_worst_ratio_cannot_hide_a_domain_regression_in_an_average():
    metrics = {"imputation": {"rmse": .8}, "recsys": {"rmse": 1.1, "mae": .9}}
    references = {"imputation": {"rmse": 1.}, "recsys": {"rmse": 1., "mae": 1.}}
    score, ratios = screening.worst_score(metrics, references)
    assert score == 1.1 and sum(ratios.values()) / 3 < 1.
    metrics["recsys"]["mae"] = float("nan")
    with pytest.raises(ValueError):
        screening.worst_score(metrics, references)


def test_cached_episodes_are_safe_tensor_only_artifacts_with_identical_metrics(tmp_path):
    truth = torch.tensor([[[1., 2.], [3., 4.]]])
    mask = torch.tensor([[[True, False], [True, False]]])
    episode = Episode(truth.masked_fill(mask, torch.nan), truth, mask, {"domain": "recsys", "gradient_data": "synthetic"},
                      SimpleNamespace(means=torch.zeros(2), stds=torch.ones(2), row_offsets=torch.zeros(2), row_scales=torch.ones(2)))
    cached = screening.cache_episode(episode)
    path = tmp_path / "bank.pt"
    torch.save([cached], path)
    loaded = torch.load(path, weights_only=True)[0]
    restored = screening.restore_episode(loaded, "cpu")
    prediction = truth + 1.
    assert episode_metrics(prediction, episode) == episode_metrics(prediction, restored)
    restored.source.zero_()
    torch.testing.assert_close(cached["source"], episode.source, equal_nan=True)
    assert torch.isnan(loaded["source"][mask]).all()
    episode.metadata["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        screening.cache_episode(episode)


def test_campaign_wait_verifies_handle_without_reading_benchmark_scores(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"state": "evaluating", "pid": 123}))
    checked = []
    monkeypatch.setattr(screening.os, "kill", lambda process, signal: checked.append((process, signal)))
    assert screening.campaign_ready(tmp_path) is False and checked == [(123, 0)]
    (tmp_path / "status.json").write_text(json.dumps({"state": "complete"}))
    assert screening.campaign_ready(tmp_path) is True
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed"}))
    with pytest.raises(RuntimeError, match="stopped"):
        screening.campaign_ready(tmp_path)


def test_entire_screen_writes_one_selected_initialization_and_exact_bank(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.]))

    class Sampler:
        def sample(self, domain, step):
            truth = torch.full((1, 2, 2), 3. if domain == "imputation" else 5.)
            mask = torch.tensor([[[True, False], [False, True]]])
            scaling = None if domain == "imputation" else SimpleNamespace(
                means=torch.zeros(2), stds=torch.ones(2), row_offsets=torch.zeros(2), row_scales=torch.ones(2))
            return Episode(truth.masked_fill(mask, torch.nan), truth, mask,
                           {"domain": domain, "gradient_data": "synthetic"}, scaling)

    output = tmp_path / "output"
    output.mkdir()
    config_path, protocol_path, base_path = output / "config.json", tmp_path / "protocol.json", tmp_path / "base.json"
    for path in (config_path, protocol_path, base_path):
        path.write_text("{}")
    checkpoints = {}
    for role, value in (("general", 0.), ("specialist", 4.), ("shared", -1.), ("pilot", .25)):
        path = tmp_path / f"{role}.pth"
        torch.save({"model": {"weight": torch.tensor([value])}}, path)
        checkpoints[role] = {"path": path.name, "sha256": screening.sha256(path)}
    candidates = [{"name": "general", "kind": "checkpoint", "checkpoint": "general"},
                  {"name": "pilot150", "kind": "checkpoint", "checkpoint": "pilot"},
                  {"name": "interpolate_75", "kind": "interpolation", "alpha": .75}]
    protocol = {"checkpoints": checkpoints, "candidates": candidates}
    config = {"input_sha256": {str(tmp_path / entry["path"]): entry["sha256"] for entry in checkpoints.values()},
              "source_sha256": {}, "base_training_config": str(base_path), "base_config_sha256": screening.sha256(base_path),
              "protocol": str(protocol_path), "protocol_sha256": screening.sha256(protocol_path),
              "cases": [{"domain": "imputation"}, {"domain": "recsys"}]}
    monkeypatch.setattr(screening, "REPO", tmp_path)
    monkeypatch.setattr(screening, "storage_guard", lambda output: None)
    monkeypatch.setattr(screening, "setup", lambda config: (Model(), None, None, [Model(), Model()], Sampler()))
    monkeypatch.setattr(screening, "streamed_point_predictions", lambda model, source, criterion, **kwargs: torch.full_like(source, float(model.weight[0])))
    monkeypatch.setattr(screening, "teacher_target", lambda teachers, criterion, episode: (torch.full_like(episode.target, 4.), {}))
    original_restore = screening.restore_episode
    monkeypatch.setattr(screening, "restore_episode", lambda cached, device: original_restore(cached, "cpu"))
    monkeypatch.setattr(screening.torch.cuda, "empty_cache", lambda: None)
    screening.screen(output, protocol, config, {})
    selection = screening.read(output / "selection.json")
    assert selection["selected"]["name"] == "interpolate_75"
    assert selection["score"] == 2.
    assert selection["new_initialization_materialized"]
    assert not selection["training_performed"] and not selection["real_benchmark_inputs_used"]
    assert selection["candidates_completed"] == 3 and selection["cases_per_candidate"] == 2
    checkpoint = torch.load(output / "selected_initialization.pth", weights_only=True)
    assert checkpoint["initialization_only"] and checkpoint["step"] == 0
    torch.testing.assert_close(checkpoint["model"]["weight"], torch.tensor([3.]))
    assert len(list(output.glob("*.pth"))) == 1
    assert len(torch.load(output / "synthetic_bank.pt", weights_only=True)) == 2
    assert len(list((output / "candidates").glob("*.json"))) == 3
    assert screening.sha256(output / "synthetic_bank.pt") == selection["bank_sha256"]
