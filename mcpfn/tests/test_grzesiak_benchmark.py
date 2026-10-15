import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from grzesiak_benchmark import RSCRIPT
from grzesiak_evaluate import distance_mean, existing_valid, metrics, stable_seed, standardized_energy
from grzesiak_summarize import dataset_balanced, rank_scores


def test_energy_matches_direct_distance_formula():
    truth = np.array([[1.0, 20.0], [2.0, 30.0], [4.0, 10.0]])
    prediction = np.array([[2.0, 20.0], [2.0, 30.0], [4.0, 20.0]])
    center, scale = truth.mean(axis=0), truth.std(axis=0, ddof=1)
    original = (truth - center) / scale
    imputed = (prediction - center) / scale
    expected = 2 * cdist(original, imputed).mean() - cdist(original, original).mean() - cdist(imputed, imputed).mean()
    result = standardized_energy(truth, prediction)
    assert result["energy_distance_std"] == pytest.approx(expected)
    assert result["energy_statistic_std"] == pytest.approx(expected * 1.5)
    assert distance_mean(original, imputed, block_size=1) == pytest.approx(cdist(original, imputed).mean())


@pytest.mark.skipif(not RSCRIPT.exists(), reason="isolated benchmark R environment not installed")
def test_energy_matches_upstream_r_implementation(tmp_path):
    generator = np.random.default_rng(43)
    truth = generator.normal(size=(15, 4))
    prediction = truth.copy()
    prediction[3:8, 2] = 0.0
    pd.DataFrame(truth).to_csv(tmp_path / "truth.csv", index=False)
    pd.DataFrame(prediction).to_csv(tmp_path / "prediction.csv", index=False)
    code = ('arguments <- commandArgs(TRUE); truth <- as.matrix(read.csv(arguments[1])); '
            'prediction <- as.matrix(read.csv(arguments[2])); standardized <- scale(truth); '
            'prediction <- scale(prediction,center=attr(standardized,"scaled:center"),scale=attr(standardized,"scaled:scale")); '
            'cat(energy::eqdist.e(rbind(standardized,prediction),c(nrow(truth),nrow(truth))))')
    result = subprocess.check_output([str(RSCRIPT), "--vanilla", "-e", code, str(tmp_path / "truth.csv"), str(tmp_path / "prediction.csv")], text=True)
    assert standardized_energy(truth, prediction)["energy_statistic_std"] == pytest.approx(float(result), rel=1e-6)


def test_identical_matrices_have_zero_energy():
    truth = np.arange(15, dtype=float).reshape(5, 3)
    assert standardized_energy(truth, truth)["energy_distance_std"] == pytest.approx(0.0, abs=1e-12)


def test_constant_column_energy_is_not_silently_repaired():
    with pytest.raises(ValueError, match="constant"):
        standardized_energy(np.ones((3, 2)), np.ones((3, 2)))


def test_both_nrmse_definitions_are_distinguished():
    truth = np.array([[1.0, 100.0], [3.0, 200.0], [5.0, 300.0]])
    missing = truth.copy()
    missing[:2] = np.nan
    prediction = truth.copy()
    prediction[:2] += np.array([1.0, 50.0])
    result = metrics(truth, missing, prediction, "grzesiak")
    assert result["nrmse"] == pytest.approx(1.0)
    expected = np.sqrt(np.mean((truth[:2] - prediction[:2]) ** 2)) / np.std(truth[:2], ddof=1)
    assert result["nrmse_upstream_pooled"] == pytest.approx(expected)
    assert result["nrmse_upstream_pooled"] != pytest.approx(result["nrmse"])


def test_observed_value_changes_fail_validation():
    truth = np.arange(12, dtype=float).reshape(6, 2)
    missing = truth.copy()
    missing[:2] = np.nan
    prediction = truth.copy()
    prediction[-1] += 1
    with pytest.raises(ValueError, match="Observed"):
        metrics(truth, missing, prediction, "uci")


def test_case_seeds_are_order_independent():
    first = stable_seed("dataset/MCAR/0.1/1")
    assert first == stable_seed("dataset/MCAR/0.1/1")
    assert first != stable_seed("dataset/MAR/0.1/1")
    assert 0 < first < 2 ** 31


def test_failure_ranks_follow_paper_rule_and_ties():
    frame = pd.DataFrame({"case": ["first"] * 4 + ["second"] * 2,
                          "score": [2.0, 2.0, 4.0, np.nan, np.nan, np.nan]})
    assert rank_scores(frame, ["case"])["rank"].tolist() == [1.5, 1.5, 3.0, 4.0, 1.0, 1.0]


def test_summary_weights_datasets_equally():
    frame = pd.DataFrame({"dataset": ["first", "first", "second"], "score": [1.0, 3.0, 10.0]})
    assert dataset_balanced(frame, "score") == pytest.approx(6.0)


@pytest.mark.skipif(not RSCRIPT.exists(), reason="isolated benchmark R environment not installed")
def test_natural_mean_accepts_tibble_inputs(tmp_path):
    from grzesiak_benchmark import REPO, write_json

    upstream = tmp_path / "upstream/R"
    upstream.mkdir(parents=True)
    (upstream / "mice.R").write_text("")
    source = tmp_path / "missing.RDS"
    subprocess.run([str(RSCRIPT), "--vanilla", "-e",
                    'saveRDS(tibble::tibble(first=c(rep(NA_real_,12),seq_len(18)),second=seq_len(30),third=seq_len(30)^2),commandArgs(TRUE)[1])',
                    str(source)], check=True, capture_output=True)
    request = {"upstream": str(upstream.parent), "input": str(source), "method": "column_mean", "seed": 42,
               "output": str(tmp_path / "prediction.csv"), "result": str(tmp_path / "result.json"),
               "columns": str(tmp_path / "columns.csv"), "url": "", "token": ""}
    write_json(tmp_path / "request.json", request)
    subprocess.run([str(RSCRIPT), "--vanilla", str(REPO / "scripts/grzesiak_natural.R"), str(tmp_path / "request.json")],
                   check=True, capture_output=True)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "ok"
    assert result["energy_iscore"] == pytest.approx(4.5)
    assert result["imputations_per_projection"] == 1
