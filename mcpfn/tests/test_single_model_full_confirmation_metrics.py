from copy import deepcopy

import pytest

import diagnose_single_model_full_confirmation_metrics as diagnostic
from test_single_model_metric_alignment import episode
from test_single_model_confirmation_metrics import cases


def proposal():
    return {"scope": {"case_indices": list(range(30)), "expected_total": 30, "small_imputation_cases": 24,
                       "large_imputation_cases": 6, "cropping": False, "normalization_refitting": False},
            "roles": list(diagnostic.cpu.ROLES), "pooled_rmse_parity": {"rtol": 1e-6, "atol": 1e-6},
            "training_performed": False, "checkpoint_selection_performed": False,
            "real_benchmark_data_used": False, "original_gate_changed": False}


def test_registered_scope_and_tolerance_are_preserved():
    original = proposal()
    diagnostic.validate_proposal(original)
    for kind in ("subset", "cropped", "tolerance", "gate", "missing_role"):
        changed = deepcopy(original)
        if kind == "subset":
            changed["scope"]["case_indices"] = list(range(24))
        elif kind == "cropped":
            changed["scope"]["cropping"] = True
        elif kind == "tolerance":
            changed["pooled_rmse_parity"]["atol"] = .01
        elif kind == "gate":
            changed["original_gate_changed"] = True
        else:
            changed["roles"].remove("initialization")
        with pytest.raises(ValueError, match="fixed 30-case"):
            diagnostic.validate_proposal(changed)


@pytest.mark.parametrize("shape,transposed", [((2000, 1000), False), ((1000, 2000), True)])
def test_full_large_shapes_are_valid_without_cropping(shape, transposed):
    diagnostic.validate_episode(episode(shape, "low_rank_mcar", transposed), 24)


def test_small_anchor_cases_remain_valid():
    diagnostic.validate_episode(episode((24, 7), "anchor_latent_mcar"), 6)


@pytest.mark.parametrize("failure", ["crop", "observed_query", "real", "metadata", "nonfinite_target", "wrong_index"])
def test_large_case_guard_fails_closed(failure):
    case = episode((2000, 1000), "low_rank_mcar")
    index = 24
    if failure == "crop":
        case = episode((1000, 1000), "low_rank_mcar")
    elif failure == "observed_query":
        case.source[case.mask] = 0.
    elif failure == "real":
        case.metadata["gradient_data"] = "real"
    elif failure == "metadata":
        case.metadata["rows"] = 1000
    elif failure == "nonfinite_target":
        case.target[case.mask] = float("nan")
    else:
        index = 30
    with pytest.raises(ValueError):
        diagnostic.validate_episode(case, index)


def test_original_bf16_score_parity_is_strict_for_every_role():
    scores = {role: {"pooled_rmse": 1.} for role in diagnostic.cpu.ROLES}
    original = {role: 1. for role in diagnostic.cpu.ROLES}
    diagnostic.verify_pooled_parity(scores, original)
    for role in diagnostic.cpu.ROLES:
        changed = deepcopy(scores)
        changed[role]["pooled_rmse"] = 1.0001
        with pytest.raises(AssertionError, match="BF16"):
            diagnostic.verify_pooled_parity(changed, original)
    with pytest.raises(ValueError, match="four matched"):
        diagnostic.verify_pooled_parity(scores, {"candidate": 1.})
    scores["candidate"]["pooled_rmse"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        diagnostic.verify_pooled_parity(scores, original)


def test_context_groups_keep_small_and_both_large_orientations():
    data = cases()
    data[0].update(rows=240, columns=80)
    data[1].update(index=24, rows=2000, columns=1000)
    extra = deepcopy(data[1])
    extra.update(index=25, seed=102, rows=1000, columns=2000)
    data.append(extra)
    rows = diagnostic.context_summaries(data)
    counts = {row["context_group"]: row["cases"] for row in rows if row["role"] == "candidate"}
    assert counts == {"small": 1, "large": 2, "large_2000x1000": 1, "large_1000x2000": 1}
