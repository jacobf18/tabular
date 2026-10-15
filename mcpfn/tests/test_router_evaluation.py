import csv
import ast

import numpy as np
import pytest

from evaluate_two_expert_router import Replay, TabularExpert, VARIANTS, summarize_tabular
from evaluate_dwip_missbench_uci import standardize_uci
from tabimpute.routing import RoutingResult, TwoExpertRouter


class NormalizedMeanExpert:
    def impute(self, matrix, return_full=False):
        mean = np.nanmean(matrix, axis=0)
        raw = np.broadcast_to(mean, matrix.shape).copy()
        completed = np.where(np.isnan(matrix), raw, matrix)
        return completed, raw


def test_uci_preprocessing_preserves_original_float32_reduction_layout():
    matrix = np.random.default_rng(42).normal(size=(1503, 5)).astype(np.float32, order="F")
    matrix[::3, 1] = np.nan
    normalized, mean, scale, _ = standardize_uci(matrix)
    expected = NormalizedMeanExpert().impute(normalized.copy(), return_full=True)[0] * scale + mean
    experts = [TabularExpert(NormalizedMeanExpert()) for _ in range(2)]
    for expert in experts:
        expert.suite = "uci"
    result = TwoExpertRouter(*experts).predict_all(matrix, include_holdout=False)
    np.testing.assert_array_equal(result.predictions["general"][np.isnan(matrix)], expected[np.isnan(matrix)])


def test_replay_rejects_different_context():
    matrix = np.array([[1., np.nan]])
    result = RoutingResult({"general": np.array([[1., 2.]])}, {}, {})
    replay = Replay([(matrix, result)], "general")
    with pytest.raises(AssertionError):
        replay.impute(np.array([[2., np.nan]]))
    np.testing.assert_array_equal(replay.impute(matrix.copy()), [[1., 2.]])


def test_aggregate_equal_dataset_weight_and_suppresses_partial_groups(tmp_path):
    cases = [{"case_id": str(index), "suite": "uci", "dataset": "first" if index < 2 else "second",
              "pattern": "MCAR", "missingness": .1, "repeat": index} for index in range(3)]
    records = [{**case, "nrmse": {variant: 2. if index < 2 else 4. for variant in VARIANTS},
                "decisions": {mode: {"selected": "general", "specialist_weight": .2} for mode in ("observed", "holdout")}}
               for index, case in enumerate(cases)]
    summarize_tabular(tmp_path, records[:2], cases)
    with (tmp_path / "summary.csv").open() as handle:
        assert next(csv.DictReader(handle))["observed_hard"] == ""
    summarize_tabular(tmp_path, records, cases)
    with (tmp_path / "summary.csv").open() as handle:
        assert float(next(csv.DictReader(handle))["observed_hard"]) == 3.


def test_historical_llm_scorer_keeps_outer_labels_out_of_router():
    from evaluate_two_expert_router import BENCH, BatchCollector

    source = BENCH / "code/eval_tabimpute.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "impute_tabimpute")
    namespace = {"np": np, "MAX_ROWS": 10, "TEST_BATCH": 1}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    scorer = namespace["impute_tabimpute"]
    train = np.array([[1., 2., 3.], [3., 4., 5.]])
    test = np.array([[2., 100., 200.], [2., 300., 400.]])
    router = TwoExpertRouter(NormalizedMeanExpert(), NormalizedMeanExpert())
    collector = BatchCollector(router)
    arguments = (train, np.ones_like(train), test, np.ones_like(test), [0], train.mean(0), train.std(0), 3)
    expected = scorer(collector, *arguments)
    assert len(collector.batches) == 2
    for matrix, result in collector.batches:
        assert np.isnan(matrix[-1, 1:]).all()
        assert result.decisions["observed"]["evidence_cells"] == 7
    replay = Replay(collector.batches, "observed_hard")
    assert scorer(replay, *arguments) == expected
    modified = test.copy()
    modified[:, 1:] = -123.
    changed_collector = BatchCollector(router)
    scorer(changed_collector, train, np.ones_like(train), modified, np.ones_like(test), [0], train.mean(0), train.std(0), 3)
    for (before, old_result), (after, new_result) in zip(collector.batches, changed_collector.batches):
        np.testing.assert_array_equal(before, after)
        assert old_result.decisions == new_result.decisions


@pytest.mark.parametrize("stage", ["tabular", "recsys"])
def test_reused_stages_cannot_be_overwritten(monkeypatch, tmp_path, stage):
    import sys
    from evaluate_two_expert_router import main

    output = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["router", "--stage", stage, "--reuse-completed", str(tmp_path / "previous"),
                                    "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()
