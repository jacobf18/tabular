from copy import deepcopy

import pytest
import torch

import single_model_structured_mixture as mixture
import train_single_matrixformer_structured_mixture as training


def imputation(value=1., *, common=True):
    return {"rmse": value, "mae": value, "missbench_nrmse": value, "uci_nrmse": value,
            "common_nrmse": value if common else None,
            "column_coverage": {"queried_columns": 3, "common_nonconstant_columns": int(common)}}


def records():
    result = [{"domain": "imputation", "student": imputation(2.)},
              {"domain": "recsys", "student": {"rmse": 3., "mae": 2.}}]
    for family in mixture.FAMILIES:
        result += [{"domain": "imputation", "structured": True, "family": family, "student": imputation(value)}
                   for value in (1., 3.)]
    return result


def metrics():
    return mixture.aggregate(records(), "student")


def test_structured_scores_never_dilute_legacy_retention_metrics():
    cases = records()
    baseline = mixture.aggregate(cases, "student")
    cases[0]["student"] = imputation(20.)
    for row in cases[2:]:
        row["student"] = imputation(.01)
    updated = mixture.aggregate(cases, "student")
    assert updated["imputation"]["rmse"] == 20.
    assert updated["structured"]["overall"]["rmse"] == .01
    assert not mixture.selection_score(updated, baseline, baseline)[2]


@pytest.mark.parametrize("kind", ["missing_family", "unbalanced_family", "missing_legacy", "bad_common", "all_undefined", "nan_canonical"])
def test_full_coverage_and_true_undefined_semantics_are_required(kind):
    cases = records()
    if kind == "missing_family":
        cases = cases[:-2]
    elif kind == "unbalanced_family":
        cases = cases[:-1]
    elif kind == "missing_legacy":
        cases = cases[2:]
    elif kind == "bad_common":
        cases[2]["student"]["common_nrmse"] = None
    elif kind == "all_undefined":
        for row in cases[2:4]:
            row["student"] = imputation(common=False)
    else:
        cases[2]["student"]["missbench_nrmse"] = float("nan")
    with pytest.raises(ValueError):
        mixture.aggregate(cases, "student")


def test_common_guard_undefined_case_remains_in_every_canonical_metric():
    cases = records()
    cases[2]["student"] = imputation(20., common=False)
    result = mixture.aggregate(cases, "student")["structured"]["families"][mixture.FAMILIES[0]]
    assert result["missbench_nrmse"] == 11.5 and result["common_nrmse"] == 3.
    assert result["canonical_metric_cases"] == 2 and result["common_nrmse_cases"] == 1


@pytest.mark.parametrize("name", mixture.STRUCTURED_METRICS)
def test_every_structured_overall_metric_is_a_retention_constraint(name):
    baseline = metrics()
    changed = deepcopy(baseline)
    changed["structured"]["overall"][name] *= 1.001
    assert not mixture.selection_score(changed, baseline, baseline)[2]


@pytest.mark.parametrize("family", mixture.FAMILIES)
def test_no_single_family_loss_is_hidden_by_other_improvements(family):
    baseline = metrics()
    changed = deepcopy(baseline)
    for name in mixture.FAMILIES:
        changed["structured"]["families"][name]["common_nrmse"] *= .5
    changed["structured"]["families"][family]["common_nrmse"] = 2.001
    score, ratios, eligible = mixture.selection_score(changed, baseline, baseline)
    assert not eligible and len(ratios) == 16 and score == ratios[f"structured_{family}"]


def test_confirmation_requires_strict_improvement_and_general_retention():
    baseline = metrics()
    assert not mixture.confirmation_gate(baseline, baseline, baseline, baseline)["passed"]
    improved = deepcopy(baseline)
    for domain, name in mixture.column.RETENTION_METRICS:
        improved[domain][name] *= .9
    for name in mixture.STRUCTURED_METRICS:
        improved["structured"]["overall"][name] *= .9
    for family in mixture.FAMILIES:
        improved["structured"]["families"][family]["common_nrmse"] *= .9
    assert mixture.confirmation_gate(improved, baseline, baseline, baseline)["passed"]
    general = deepcopy(baseline)
    general["structured"]["overall"]["rmse"] = .1
    assert not mixture.confirmation_gate(improved, baseline, general, baseline)["passed"]


def test_balanced_training_cycles_and_explicit_neural_shapes():
    cases = mixture.training_cases()
    assert len(cases) == 200 and [row["step"] for row in cases] == list(range(2, 401, 2))
    assert len({row["case"]["seed"] for row in cases}) == 200
    for cycle in range(4):
        assert {(row["case"]["family"], row["case"]["component"]) for row in cases[cycle * 48:(cycle + 1) * 48]} == {
            (family, component) for family in mixture.FAMILIES for component in mixture.COMPONENTS}
    for row in cases:
        case = row["case"]
        shapes = mixture.COMPACT_SHAPES if case["family"] in mixture.NEURAL_FAMILIES else mixture.SPECTRAL_SHAPES
        assert tuple(case["shape"]) in shapes


@pytest.mark.parametrize("confirmation,count", [(False, 96), (True, 192)])
def test_all_structured_validation_cases_are_prospectively_fixed(confirmation, count):
    cases = mixture.structured_bank(99101, confirmation=confirmation)
    assert len(cases) == count and len({case["structured"]["seed"] for case in cases}) == count
    assert {sum(case["structured"]["family"] == family for case in cases) for family in mixture.FAMILIES} == {count // 12}


def test_only_truth_defined_coverage_is_required_to_match():
    first, second = imputation(), imputation()
    first["column_coverage"]["missbench_scored_columns"] = 2
    second["column_coverage"]["missbench_scored_columns"] = 3
    mixture.require_truth_coverage([{"domain": "imputation", "general": first, "student": second}], ("general", "student"))
    second["column_coverage"]["common_nonconstant_columns"] = 2
    with pytest.raises(ValueError, match="Truth-only"):
        mixture.require_truth_coverage([{"domain": "imputation", "general": first, "student": second}], ("general", "student"))


def test_runtime_callbacks_restore_even_when_training_fails(tmp_path):
    names = ("setup", "training_episode", "teacher_target", "backward_episode", "episode_metrics", "aggregate",
             "selection_score", "prepare_bank", "evaluate_bank")
    originals = {name: getattr(mixture.backend, name) for name in names}
    with pytest.raises(RuntimeError):
        with mixture.runtime({"query_seed_offset": 3}, tmp_path):
            assert mixture.backend.aggregate is mixture.aggregate
            assert mixture.backend.selection_score is mixture.selection_score
            raise RuntimeError("deliberate failure")
    assert all(getattr(mixture.backend, name) is callback for name, callback in originals.items())


def test_runtime_never_changes_rating_sampler_arguments(tmp_path):
    class Recorder:
        def sample(self, *arguments, **keywords):
            return arguments, keywords

    config = {"query_seed_offset": 3, "rating_training_shapes": [[1000, 1000], [2000, 1000]],
              "training_structured_cases": mixture.training_cases(), **mixture.SEEDS}
    with mixture.runtime(config, tmp_path):
        assert mixture.backend.training_episode(Recorder(), config, "recsys", 1) == (("recsys", 1), {"shape": (1000, 1000)})
        assert mixture.backend.training_episode(Recorder(), config, "recsys", 2) == (("recsys", 2), {"shape": (2000, 1000)})
        arguments, keywords = mixture.backend.training_episode(Recorder(), config, "imputation", 2)
        assert arguments == ("imputation", 2) and keywords == {"structured": config["training_structured_cases"][0]["case"]}


def configuration():
    config = dict(version=9, **mixture.SEEDS, steps=400, validate_every=50, peak_lr=1e-5, warmup_steps=20,
        weight_decay=.02, grad_clip=.5, memory_fraction=.8,
        parent_checkpoint=str(training.ROOT / "query_budget_v8/best_validation_checkpoint.pth"),
        teachers=[str(training.GENERAL), str(training.SPECIALIST)],
        rating_training_shapes=[[1000, 1000], [2000, 1000]], training_teacher_policy={"imputation": "initialization", "recsys": "specialist"},
        query_budget_objective=deepcopy(training.previous.SPECIFICATION),
        spectral_amplitude_mixture=deepcopy(training.previous.area.SPECIFICATION),
        probability_loss_scales=deepcopy(training.previous.balanced.PROBABILITY_SCALES),
        column_objective=deepcopy(mixture.column.COLUMN_OBJECTIVE),
        phases=[{"name": "structured_mixture", "end_step": 400, "rows_low": 256, "rows_high": 2000,
                 "cols_low": 128, "cols_high": 1000, "batch_size": 1}],
        structured_policy=deepcopy(mixture.POLICY), structured_prior=deepcopy(training.PRIOR_SPECIFICATION),
        training_structured_cases=mixture.training_cases(),
        validation_cases=training.repeated_bank(mixture.SEEDS["validation_seed"]) + mixture.structured_bank(mixture.SEEDS["structured_validation_seed"]),
        confirmation_cases=training.repeated_bank(mixture.SEEDS["confirmation_seed"], confirmation=True)
            + mixture.structured_bank(mixture.SEEDS["structured_confirmation_seed"], confirmation=True))
    return config


def test_complete_registration_specification_counts_and_seeds():
    config = configuration()
    training.validate_recipe(config)
    assert len(config["validation_cases"]) == 152 and len(config["confirmation_cases"]) == 284


@pytest.mark.parametrize("change", ["missing_case", "changed_policy", "old_seed", "rating_shape", "smoke_init", "context_shape", "lr"])
def test_registered_recipe_rejects_scope_or_training_drift(change):
    config = configuration()
    if change == "missing_case":
        config["confirmation_cases"].pop()
    elif change == "changed_policy":
        config["structured_policy"]["retention"] = "ignore failures"
    elif change == "old_seed":
        config["data_seed"] = 52100501
    elif change == "rating_shape":
        config["rating_training_shapes"] = [[1000, 1000]]
    elif change == "smoke_init":
        config["parent_checkpoint"] = str(training.ROOT / "complete_missingness_training_preflight_v1/smoke_test_checkpoint.pth")
    elif change == "context_shape":
        config["phases"][0]["rows_high"] = 100
    else:
        config["peak_lr"] = 1e-3
    with pytest.raises(ValueError):
        training.validate_recipe(config)


def test_cpu_bank_and_evaluation_retain_full_raw_predictions(tmp_path, monkeypatch):
    from single_model_distillation import Episode
    import numpy as np

    class Sampler:
        def sample(self, domain, step, structured=None):
            target = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, :2] = True
            metadata = {"domain": domain, "gradient_data": "synthetic", "seed": 1}
            if structured:
                metadata.update(structured=True, family=structured["family"])
            return Episode(target.masked_fill(mask, torch.nan), target, mask, metadata)

    def prediction(model, source, criterion, **options):
        return source.nan_to_num() + .25

    monkeypatch.setattr(mixture, "storage_guard", lambda output: None)
    monkeypatch.setattr(mixture, "streamed_point_predictions", prediction)
    monkeypatch.setattr(mixture, "reference_teacher_target", lambda teachers, criterion, episode: (
        episode.source.nan_to_num() + .5, {"target": "test-reference"}))
    specifications = [{"domain": "imputation"}, {"domain": "recsys"}] + [
        {"domain": "imputation", "structured": {"family": family}} for family in mixture.FAMILIES]
    with mixture.runtime({"query_seed_offset": 3}, tmp_path):
        bank, reference = mixture.backend.prepare_bank(Sampler(), [object(), object()], None, specifications, tmp_path)
        model = torch.nn.Linear(1, 1).train()
        metrics, cases = mixture.backend.evaluate_bank(model, None, bank, device="cpu")
        repeated_metrics, repeated = mixture.backend.evaluate_bank(model, None, bank, device="cpu")
    assert len(bank) == len(cases) == 14 and model.training
    assert metrics == repeated_metrics and len(reference["cases"]) == 14
    assert metrics["imputation"]["canonical_metric_cases"] == 1
    assert metrics["structured"]["overall"]["canonical_metric_cases"] == 12
    assert cases[0]["prediction_file"] != repeated[0]["prediction_file"]
    for row in reference["cases"]:
        with np.load(row["prediction_file"], allow_pickle=False) as stored:
            assert set(stored.files) == {"general", "teacher"} and stored["general"].shape == (1, 4, 2)
    for row in cases + repeated:
        with np.load(row["prediction_file"], allow_pickle=False) as stored:
            assert set(stored.files) == {"student"} and np.isfinite(stored["student"]).all()


def test_confirmation_cannot_pass_if_only_structured_metrics_improve():
    baseline = metrics()
    changed = deepcopy(baseline)
    for family in mixture.FAMILIES:
        changed["structured"]["families"][family]["common_nrmse"] *= .5
    for name in mixture.STRUCTURED_METRICS:
        changed["structured"]["overall"][name] *= .5
    changed["recsys"]["mae"] *= 1.01
    assert not mixture.confirmation_gate(changed, baseline, baseline, baseline)["passed"]
