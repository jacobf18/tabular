from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from recsys_evaluation_common import sha256, write_json
from single_model_distillation import Episode
import single_model_column_objective as column
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from test_streamed_training import assert_gradients_match, example
import train_single_matrixformer_column_aligned as training


def exact_column_risk(prediction, truth, mask):
    terms = []
    for threshold in column.COLUMN_OBJECTIVE["thresholds"]:
        columns = []
        for index in range(truth.shape[2]):
            selected = mask[0, :, index]
            values = truth[0, selected, index]
            if values.numel() and values.std(correction=0) >= threshold:
                columns.append((prediction[0, selected, index] - values).square().mean().sqrt() / values.std(correction=0))
        terms.append(torch.stack(columns).mean() if columns else prediction.sum() * 0.)
    return torch.stack(terms).mean()


def test_majorizer_is_tangent_at_parent_and_upper_bounds_true_column_risk():
    torch.manual_seed(412)
    truth = torch.randn(1, 11, 7, dtype=torch.float64)
    teacher = (truth + .2 * torch.randn_like(truth)).requires_grad_()
    mask = torch.rand_like(truth) < .7
    mask[:, :, -1] = False
    weights, offset, info = column.column_majorizer(truth, mask, teacher)
    surrogate = (weights[mask] * (teacher[mask] - truth[mask]).square()).mean() + offset
    exact = exact_column_risk(teacher, truth, mask)
    torch.testing.assert_close(surrogate, exact, rtol=2e-7, atol=2e-7)
    surrogate_gradient = torch.autograd.grad(surrogate, teacher, retain_graph=True)[0]
    exact_gradient = torch.autograd.grad(exact, teacher)[0]
    torch.testing.assert_close(surrogate_gradient, exact_gradient, rtol=2e-7, atol=2e-7)
    assert not weights.requires_grad and not offset.requires_grad
    assert info["scored_columns"] == [6, 6]
    for scale in (.01, .5, 4.):
        candidate = truth + scale * torch.randn_like(truth)
        bound = (weights[mask] * (candidate[mask] - truth[mask]).square()).mean() + offset
        assert bound >= exact_column_risk(candidate, truth, mask) - 1e-6


def test_constant_near_constant_empty_and_singleton_columns_have_fixed_truth_coverage():
    truth = torch.tensor([[[-1., -.0005, -.00005, 2., 3., 1.],
                            [0., 0., 0., 2., 5., 3.],
                            [1., .0005, .00005, 2., 7., 5.]]])
    mask = torch.ones_like(truth, dtype=torch.bool)
    mask[:, 1:, 4] = False
    mask[:, :, 5] = False
    teacher = truth + .2
    weights, offset, info = column.column_majorizer(truth, mask, teacher)
    assert info["scored_columns"] == [1, 2]
    assert torch.count_nonzero(weights[:, :, 2:]) == 0
    assert torch.isfinite(offset) and torch.isfinite(weights).all()
    changed = truth - 8.
    assert column.column_majorizer(truth, mask, changed)[2]["scored_columns"] == info["scored_columns"]


def test_perfect_teacher_floor_remains_finite_and_all_constant_point_term_is_zero():
    truth = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    mask = torch.ones_like(truth, dtype=torch.bool)
    weights, offset, info = column.column_majorizer(truth, mask, truth)
    assert info["floor_active_columns"] == 3
    assert torch.isfinite(weights).all() and torch.isfinite(offset)
    constant = torch.ones_like(truth)
    weights, offset, info = column.column_majorizer(constant, mask, constant)
    assert info["scored_columns"] == [0, 0]
    assert not weights.any() and offset == 0


@pytest.mark.parametrize("failure", ["shape", "empty", "truth_nan", "teacher_nan", "mask_dtype", "weight"])
def test_majorizer_rejects_invalid_inputs(failure):
    truth = torch.randn(1, 5, 3)
    teacher = torch.zeros_like(truth)
    mask = torch.ones_like(truth, dtype=torch.bool)
    specification = deepcopy(column.COLUMN_OBJECTIVE)
    if failure == "shape":
        teacher = teacher[:, :, :-1]
    elif failure == "empty":
        mask.zero_()
    elif failure == "truth_nan":
        truth[0, 0, 0] = float("nan")
    elif failure == "teacher_nan":
        teacher[0, 0, 0] = float("nan")
    elif failure == "mask_dtype":
        mask = mask.float()
    else:
        specification["weight"] = .1
    with pytest.raises(ValueError):
        column.column_majorizer(truth, mask, teacher, specification)


@pytest.mark.parametrize("transposed", [False, True])
def test_streamed_majorizer_loss_and_gradients_match_dense_with_uneven_lookup_chunks(monkeypatch, transposed):
    reference, source, truth, mask = example(shape=(1, 9, 7))
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    model = deepcopy(reference)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teacher = torch.randn_like(truth, requires_grad=True)
    weights, offset, _ = column.column_majorizer(truth, mask, teacher)
    logits = reference(source, decode_mask=mask)
    expected = column.balanced.balanced_loss(criterion, logits, truth[mask], teacher[mask], domain="imputation",
                                           probability_scales=column.balanced.PROBABILITY_SCALES)
    expected = (expected + weights[mask] * (criterion.mean(logits) - truth[mask]).square()).mean() + offset
    monkeypatch.setattr(column, "TRAIN_OPTIONS", {"attention_batch_size": 2, "cell_batch_size": 13,
                                                 "encoder_row_batch_size": 3, "decoder_cell_batch_size": 5})
    episode = Episode(source, truth, mask, {"domain": "imputation", "gradient_data": "synthetic"})
    actual = column.backward_episode(model, criterion, episode, teacher, .5)
    (.5 * expected).backward()
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-6)
    assert_gradients_match(reference, model)
    assert teacher.grad is None and episode.metadata["column_objective"]["query_cells"] == int(mask.sum())


def test_rating_loss_and_gradients_are_unchanged():
    reference, source, truth, mask = example(shape=(1, 9, 7))
    model = deepcopy(reference)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teacher = torch.randn_like(truth)
    episode = Episode(source, truth, mask, {"domain": "recsys", "gradient_data": "synthetic"})
    expected = column.BASE_BACKWARD(reference, criterion, episode, teacher, .5,
                                   probability_scales=column.balanced.PROBABILITY_SCALES)
    actual = column.backward_episode(model, criterion, episode, teacher, .5)
    assert actual == expected
    assert_gradients_match(reference, model)
    assert "column_objective" not in episode.metadata


def test_real_data_rejected_before_any_model_execution():
    episode = Episode(None, None, None, {"domain": "imputation", "gradient_data": "real"})
    with pytest.raises(ValueError, match="synthetic"):
        column.backward_episode(None, None, episode, None, .5)
    with pytest.raises(ValueError, match="synthetic"):
        column.episode_metrics(None, episode)


def metrics(value):
    return {"imputation": {name: value for name in ("rmse", "mae", "missbench_nrmse", "uci_nrmse", "common_nrmse")},
            "recsys": {"rmse": value, "mae": value}}


@pytest.mark.parametrize("pair", column.RETENTION_METRICS)
def test_every_old_and_new_metric_must_retain_initialization(pair):
    student, initial, references = metrics(.9), metrics(1.), metrics(1.)
    assert column.selection_score(student, initial, references)[2]
    domain, name = pair
    student[domain][name] = 1.01
    assert not column.selection_score(student, initial, references)[2]


def test_confirmation_requires_general_retention_and_strict_primary_improvement():
    assert column.confirmation_gate(metrics(.9), metrics(1.), metrics(1.), metrics(1.))["passed"]
    assert not column.confirmation_gate(metrics(.9), metrics(1.), metrics(.8), metrics(1.))["passed"]
    assert not column.confirmation_gate(metrics(1.), metrics(1.), metrics(1.), metrics(1.))["passed"]
    assert {("imputation", "rmse"), ("recsys", "rmse"), ("recsys", "mae")} <= set(column.RETENTION_METRICS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_incomplete_or_nonfinite_metrics_cannot_enter_selection(bad):
    student = metrics(.9)
    student["imputation"]["common_nrmse"] = bad
    with pytest.raises(ValueError, match="finite"):
        column.selection_score(student, metrics(1.), metrics(1.))


@pytest.mark.parametrize("failure", [False, True])
def test_nested_adapters_preserve_reference_teachers_and_restore_every_callback(tmp_path, monkeypatch, failure):
    runner = column.continuation
    original_target = runner.teacher_target
    originals = (runner.episode_metrics, runner.aggregate, runner.selection_score, runner.backward_episode,
                 column.balanced.backward_episode, runner.setup)

    def bank(*arguments):
        assert runner.teacher_target is original_target
        assert runner.episode_metrics is column.episode_metrics
        if failure:
            raise RuntimeError("interrupted bank")
        return [], {"general": metrics(1.), "teacher": metrics(1.1),
                    "references": {"imputation": {"rmse": 1.}, "recsys": {"rmse": 1.1, "mae": 1.1}}}

    monkeypatch.setattr(runner, "prepare_bank", bank)
    try:
        with training.parent.training_teachers(), column.evaluation_metrics(), column.training_objective():
            _, reference = runner.prepare_bank(None, None, None, None, tmp_path)
            assert reference["references"]["imputation"]["uci_nrmse"] == 1.
            assert runner.teacher_target is not original_target
            assert runner.backward_episode is column.backward_episode
    except RuntimeError:
        assert failure
    assert (runner.episode_metrics, runner.aggregate, runner.selection_score, runner.backward_episode,
            column.balanced.backward_episode, runner.setup) == originals
    assert runner.prepare_bank is bank and runner.teacher_target is original_target


def test_actual_two_update_cpu_run_saves_one_changed_network_and_keeps_teachers_frozen(tmp_path, monkeypatch):
    runner = column.continuation
    student, source, truth, mask = example(shape=(1, 9, 7))
    initial = deepcopy(student.state_dict())
    teachers = [deepcopy(student).eval().requires_grad_(False) for _ in range(2)]
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    optimizer = torch.optim.SGD(student.parameters(), lr=.001)

    def sample(domain, step, **specification):
        return Episode(source, truth, mask, {"domain": domain, "gradient_data": "synthetic", "seed": step})

    episodes = [sample(kind, 0) for kind in ("imputation", "recsys")]
    bank = [runner.cache_episode(episode) for episode in episodes]
    torch.save(bank, tmp_path / "synthetic_bank.pt")
    controls = [{"domain": episode.metadata["domain"], "general": column.episode_metrics(torch.zeros_like(truth), episode)} for episode in episodes]
    control_metrics = column.aggregate(controls, "general")
    reference = {"bank_sha256": sha256(tmp_path / "synthetic_bank.pt"), "general": control_metrics,
                 "teacher": deepcopy(control_metrics), "references": deepcopy(control_metrics)}
    write_json(tmp_path / "reference.json", reference)
    config = {"version": 6, "steps": 2, "peak_lr": .001, "warmup_steps": 0, "validate_every": 1, "grad_clip": 1.,
              "validation_cases": [{}] * 56, "confirmation_cases": [{}] * 92,
              "rating_training_shapes": [[1000, 1000], [2000, 1000]],
              "column_objective": deepcopy(column.COLUMN_OBJECTIVE),
              "probability_loss_scales": dict(column.balanced.PROBABILITY_SCALES), "bank_replicates": 2,
              "training_teacher_policy": dict(training.parent.TEACHER_POLICY),
              "primary_metrics": [list(pair) for pair in column.PRIMARY_METRICS],
              "retention_metrics": [list(pair) for pair in column.RETENTION_METRICS]}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "preflight.json", {"config_sha256": sha256(tmp_path / "config.json")})
    monkeypatch.setattr(runner, "verify_run", lambda *args: None)
    monkeypatch.setattr(runner, "storage_guard", lambda *args: None)
    monkeypatch.setattr(runner, "setup", lambda config: (student, optimizer, criterion, teachers, SimpleNamespace(sample=sample)))
    monkeypatch.setattr(runner, "prepare_bank", lambda *args: (bank, reference))
    monkeypatch.setattr(runner, "rng_state", lambda: {"cpu": torch.get_rng_state(), "cuda": []})
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    evaluate_bank = runner.evaluate_bank
    monkeypatch.setattr(runner, "evaluate_bank", lambda *args: evaluate_bank(*args, device="cpu"))
    training.train(config, tmp_path)
    assert runner.read(tmp_path / "status.json")["state"] == "training_complete"
    latest = torch.load(tmp_path / "latest_checkpoint.pth", weights_only=True)
    assert latest["step"] == 2 and set(latest["model"]) == set(initial)
    assert any(not torch.equal(value, initial[name]) for name, value in latest["model"].items())
    assert all(torch.isfinite(value).all() for value in latest["model"].values())
    selected = torch.load(tmp_path / "best_validation_checkpoint.pth", weights_only=True)
    assert set(selected["model"]) == set(initial) and "teachers" not in selected
    for teacher in teachers:
        assert not any(parameter.grad is not None or parameter.requires_grad for parameter in teacher.parameters())
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    assert sha256(tmp_path / "synthetic_bank.pt") == reference["bank_sha256"]
