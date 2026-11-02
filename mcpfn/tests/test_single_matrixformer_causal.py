import numpy as np
import pandas as pd
import pytest

from evaluate_single_matrixformer_causal import (
    CANDIDATES, METRICS, REFERENCES, average_predictions, causal_module, complete, load_draw, summarize,
)
from export_single_matrixformer_causal_results import paper_scores, verify_archive


def fixture_frame():
    rows = []
    for path, values in (("first", [0.0, 0.5]), ("second", [1.0, 0.5, 0.5])):
        for draw, value in enumerate(values):
            for index, method in enumerate(REFERENCES):
                score = value if index == 0 else (index - 1) / 5
                rows.append({"family": "test", "path": path, "draw_name": str(draw), "method": method,
                             **{metric: score for metric in METRICS}})
    return pd.DataFrame(rows)


def test_fixed_scale_equal_design_weighting_and_ci():
    reference = fixture_frame()
    expected, _ = summarize(reference)
    candidate = reference[reference.method == REFERENCES[0]].copy()
    candidate["method"] = CANDIDATES[0]
    candidate[list(METRICS)] = -1.0
    scores, _ = summarize(pd.concat([reference, candidate]))
    pd.testing.assert_frame_equal(scores[scores.method.isin(REFERENCES)].reset_index(drop=True), expected)
    selected = scores[(scores.method == REFERENCES[0]) & (scores.metric == METRICS[0])].iloc[0]
    assert selected.score == pytest.approx((0.25 + 2 / 3) / 2)
    expected_se = np.sqrt((np.std([0, 0.5], ddof=1) / np.sqrt(2)) ** 2 +
                          (np.std([1, 0.5, 0.5], ddof=1) / np.sqrt(3)) ** 2) / 2
    assert selected.ci95 == pytest.approx(1.96 * expected_se)
    assert (scores.loc[scores.method == CANDIDATES[0], "score"] == -1).all()


def test_invalid_score_rows_rejected():
    frame = fixture_frame()
    for invalid in (pd.concat([frame, frame.iloc[:1]]), frame.iloc[1:]):
        with pytest.raises(ValueError):
            summarize(invalid)


def test_archive_comparison_allows_only_saved_rounding():
    scores, _ = summarize(fixture_frame())
    historical = scores.copy()
    historical[["score", "se", "ci95"]] = historical[["score", "se", "ci95"]].round(6)
    assert verify_archive(scores, historical) <= 5.01e-7
    historical.loc[0, "ci95"] += 0.001
    with pytest.raises(AssertionError):
        verify_archive(scores, historical)


def test_paper_selects_original_primary_without_changing_scores_or_archive():
    reference = fixture_frame()
    template = reference[reference.method == REFERENCES[0]]
    candidates = [template.assign(method=method) for method in CANDIDATES]
    original, _ = summarize(pd.concat([reference, *candidates]))
    frozen = original.copy(deep=True)
    selected = paper_scores(original)
    assert set(selected.loc[selected.shown, "method"]) == {*REFERENCES, "matrixformer"}
    pd.testing.assert_frame_equal(selected.drop(columns="shown"), original.drop(columns="shown"))
    pd.testing.assert_frame_equal(original, frozen)
    with pytest.raises(ValueError, match="orientations"):
        paper_scores(original[original.method != "matrixformer-avg"])


def test_paper_does_not_select_the_best_scoring_orientation():
    reference = fixture_frame()
    template = reference[reference.method == REFERENCES[0]]
    candidates = [template.assign(method=method, **{metric: -index for metric in METRICS})
                  for index, method in enumerate(CANDIDATES)]
    scores, _ = summarize(pd.concat([reference, *candidates]))
    selected = paper_scores(scores)
    assert (selected.loc[selected.method == "matrixformer-avg", "score"] <
            selected.loc[selected.method == "matrixformer", "score"].to_numpy()).all()
    assert selected.loc[selected.method == "matrixformer", "shown"].all()
    assert not selected.loc[selected.method.isin(["matrixformer-T", "matrixformer-avg"]), "shown"].any()


def test_tied_reference_draw_is_skipped_for_every_method():
    frame = fixture_frame()
    frame.loc[(frame.path == "second") & (frame.draw_name == "0"), list(METRICS)] = 0.5
    _, designs = summarize(frame)
    selected = designs[designs.path == "second"]
    assert (selected.n_tied_draws == 1).all()
    assert (selected.n_draws == 2).all()


def test_completion_and_metrics_match_original_protocol():
    causal = causal_module()
    observed = np.array([[1, 10, np.nan], [3, np.nan, 8], [5, 14, 6]], dtype=np.float32)
    calls = []

    def impute(panel):
        calls.append(panel.copy())
        return np.nan_to_num(panel, nan=0.0) + 0.125

    standard = complete(impute, observed, causal.standardize_columns)
    transposed = complete(impute, observed, causal.standardize_columns, transpose=True)
    np.testing.assert_allclose(np.nanmean(calls[0], axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(np.nanmean(calls[1], axis=0), 0, atol=1e-6)
    mask = np.isnan(observed)
    average = average_predictions(standard, transposed, observed)
    for prediction in (standard, transposed, average):
        np.testing.assert_array_equal(prediction[~mask], observed[~mask])
    np.testing.assert_array_equal(average[mask], ((standard + transposed) * 0.5)[mask])
    assert standard[1, 1] == 12.25
    true = np.nan_to_num(observed, nan=4.0)
    metrics = causal.compute_metrics(true, average, mask)
    assert metrics["treated_block_rmse"] == float(np.sqrt(np.mean((true[mask] - average[mask]) ** 2)))
    assert metrics["tau_abs_error"] == float(abs(np.mean(true[mask]) - np.mean(average[mask])))


def test_draw_audit_detects_mask_mismatch(tmp_path):
    np.save(tmp_path / "observed.npy", np.array([[1, np.nan]], dtype=np.float32))
    np.save(tmp_path / "true.npy", np.array([[1, 2]], dtype=np.float32))
    np.save(tmp_path / "treatment_mask.npy", np.array([[False, True]]))
    load_draw(tmp_path)
    np.save(tmp_path / "treatment_mask.npy", np.array([[True, False]]))
    with pytest.raises(ValueError, match="hidden block"):
        load_draw(tmp_path)
