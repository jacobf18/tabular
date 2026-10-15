import numpy as np
import pytest

from diagnose_single_model_column_failures import ROLES, column_rows, contribution_summary, metric


def fixture_arrays():
    truth = np.array([[0., 0., 1.], [2., 0., 1.0004], [4., 0., 1.0008]], dtype=np.float32)
    mask = np.ones_like(truth, dtype=bool)
    source = np.full_like(truth, np.nan)
    arrays = {role: truth.copy() for role in ROLES}
    arrays["initialization"][:, 0] += .5
    arrays["candidate"][:, 0] += 1.
    arrays["candidate"][:, 1] += .1
    arrays["candidate"][:, 2] += .01
    return truth, mask, source, arrays


def test_column_contributions_keep_original_threshold_and_skipping_rules():
    rows = column_rows(*fixture_arrays())
    assert len(rows) == 3
    assert rows[1]["general_missbench_nrmse"] == 0.
    assert rows[1]["candidate_missbench_nrmse"] is None
    assert rows[2]["candidate_missbench_nrmse"] is None
    assert rows[2]["candidate_uci_nrmse"] > 1
    assert not rows[2]["common_eligible"]
    assert rows[0]["initialization_common_nrmse"] > 0
    assert rows[0]["candidate_common_nrmse"] == pytest.approx(2 * rows[0]["initialization_common_nrmse"])
    assert all(row["observations"] == 0 and row["queries"] == 3 for row in rows)


def test_each_group_partition_reconstructs_aggregate_contribution():
    rows = column_rows(*fixture_arrays())
    rows = [{"index": index, "replica": index + 1, "prior": "synthetic", "context": "small", **row}
            for index in range(2) for row in rows]
    summary = contribution_summary(rows, cases=2)
    for suite in ("missbench", "uci", "common"):
        expected = sum(row[f"candidate_{suite}_contribution"] - row[f"initialization_{suite}_contribution"] for row in rows) / 2
        for kind in ("scale_band", "support_band", "context", "prior", "replica"):
            actual = sum(row["candidate_minus_initialization"] for row in summary if row["metric"] == suite and row["grouping"] == kind)
            assert actual == pytest.approx(expected)


def test_column_diagnostic_rejects_missing_role_or_nonfinite_prediction():
    truth, mask, source, arrays = fixture_arrays()
    with pytest.raises(ValueError):
        column_rows(truth, mask, source, {"candidate": arrays["candidate"]})
    arrays["candidate"][0, 0] = np.nan
    with pytest.raises(ValueError):
        column_rows(truth, mask, source, arrays)


def test_column_contributions_reproduce_both_frozen_original_scorers():
    truth, mask, source, arrays = fixture_arrays()
    rows = column_rows(truth, mask, source, arrays)
    scorers = {suite: metric.load_original(path, "compute_normalized_rmse_columnwise")
               for suite, path in metric.SCORERS.items()}
    for role in ROLES:
        scores = metric.score_queries(truth, arrays[role], mask, scorers)
        for suite, name in (("missbench", "missbench_nrmse"), ("uci", "uci_nrmse"),
                            ("common", "common_nonconstant_nrmse")):
            assert sum(row[f"{role}_{suite}_contribution"] for row in rows) == pytest.approx(scores[name], rel=1e-6)
