from copy import deepcopy
import ctypes
import inspect

import pytest
import torch

import recover_single_model_family_projection_pipeline as recovery
from test_single_model_family_projection_training_audit import episode_fixture


def test_sparse_context_can_have_unscored_missing_cells_without_changing_original():
    original = inspect.getsource(recovery.audit.check_episode)
    payload, task, logged = episode_fixture()
    payload["episode"]["source"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="Incomplete or invalid"):
        recovery.audit.check_episode(payload, task, logged)
    recovery.corrected_checker()(payload, task, logged)
    assert inspect.getsource(recovery.audit.check_episode) == original


@pytest.mark.parametrize("kind", ["infinite_source", "no_context", "no_queries", "nonfinite_target", "nonfinite_teacher", "changed_seed", "real_data", "nonfinite_loss"])
def test_recovery_preserves_all_other_data_validity_checks(kind):
    payload, task, logged = episode_fixture()
    payload["episode"]["source"][0, 0] = torch.nan
    if kind == "infinite_source":
        payload["episode"]["source"][0, 0] = torch.inf
    elif kind == "no_context":
        payload["episode"]["source"].fill_(torch.nan)
    elif kind == "no_queries":
        payload["episode"]["mask"].fill_(False)
    elif kind == "nonfinite_target":
        payload["episode"]["target"][0, 0] = torch.nan
    elif kind == "nonfinite_teacher":
        payload["teacher"][0, 0] = torch.nan
    elif kind == "changed_seed":
        payload["episode"]["metadata"]["seed"] += 1
    elif kind == "real_data":
        payload["episode"]["metadata"]["gradient_data"] = "real"
    else:
        logged["losses"]["control"] = float("nan")
    with pytest.raises(ValueError):
        recovery.corrected_checker()(payload, task, logged)


def test_recovery_uses_distinct_destinations_and_restores_original_callbacks():
    checker = recovery.audit.check_episode
    audit_file, pipeline_file = recovery.audit.__file__, recovery.pipeline.__file__
    original = {name: deepcopy(getattr(recovery.pipeline, name)) for name in ("OUTPUT", "AUDIT", "CONFIRMATIONS", "CAMPAIGNS", "REVIEWS")}
    with recovery.runtime():
        assert recovery.audit.check_episode is not checker
        assert recovery.audit.__file__ == recovery.pipeline.__file__ == str(recovery.Path(recovery.__file__).resolve())
        assert recovery.pipeline.OUTPUT != original["OUTPUT"]
        assert recovery.pipeline.AUDIT != original["AUDIT"]
        for field in ("CONFIRMATIONS", "CAMPAIGNS", "REVIEWS"):
            assert set(getattr(recovery.pipeline, field).values()).isdisjoint(original[field].values())
    assert recovery.audit.check_episode is checker
    assert recovery.audit.__file__ == audit_file and recovery.pipeline.__file__ == pipeline_file
    assert {name: getattr(recovery.pipeline, name) for name in original} == original


@pytest.mark.parametrize("squared,beta", [(False, .9), (True, .999)])
def test_fused_moments_match_independent_scalar_fmaf_including_underflow(squared, beta):
    function = ctypes.CDLL("libm.so.6").fmaf
    function.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    function.restype = ctypes.c_float
    beta = ctypes.c_float(beta).value
    applied = torch.tensor([.1, -.1, 1e-27, 1e-19, 1., -3.], dtype=torch.float32).double()
    previous = torch.tensor([.3, -.2, 0., 1e-32, .4, .1], dtype=torch.float32).double()
    if squared:
        previous = previous.abs()
    expected = []
    for old, gradient in zip(previous.tolist(), applied.tolist()):
        value = ctypes.c_float(gradient * gradient).value if squared else gradient
        expected.append(function(beta, old, function(-beta, value, value)))
    actual = recovery.fused_moment(previous, applied, beta, squared=squared)
    assert torch.equal(actual, torch.tensor(expected, dtype=torch.float32).double())


def test_nondeterministic_backward_is_reported_not_falsely_called_bitwise_matched():
    original = {"before": torch.ones(2), "combined_gradient": torch.ones(2), "applied_gradient": torch.ones(2),
                "tentative": torch.ones(2), "task_gradients": torch.ones((2, 2))}
    projected = deepcopy(original)
    projected["task_gradients"][0, 0] += .01
    projected["combined_gradient"][0] += .001
    report = recovery.pair_diagnostics({"control": original, "projected": projected})
    assert report["initial_parameters_exactly_equal"] and not report["all_task_gradients_exactly_equal"]
    assert report["tasks"][0]["unequal_coordinates"] == 1
    assert report["tasks"][1]["exactly_equal"]
    assert not report["vectors"]["combined_gradient"]["exactly_equal"]
    projected["before"][0] += .1
    with pytest.raises(ValueError, match="exactly the same"):
        recovery.pair_diagnostics({"control": original, "projected": projected})
