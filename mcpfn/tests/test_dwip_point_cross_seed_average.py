from copy import deepcopy
import json

import pytest

import evaluate_dwip_point_cross_seed_average as runner
from evaluate_dwip_point_cross_seed_average import QUEUES, WEIGHTS, verify_registration
from evaluate_dwip_cross_seed_average import select_variant


def test_point_average_registration_freezes_replicas_weights_and_sources():
    sources, manifests = {"wrapper": "hash"}, {"seed1": "first", "seed2": "second"}
    registered = {"queues": QUEUES, "weights": WEIGHTS, "source_sha256": sources, "queue_manifest_sha256": manifests}
    verify_registration(registered, sources, manifests)
    changed = deepcopy(registered)
    changed["weights"]["seed1"] = .6
    with pytest.raises(ValueError, match="weights changed"):
        verify_registration(changed, sources, manifests)
    with pytest.raises(ValueError, match="source or training manifest"):
        verify_registration(registered, {"wrapper": "tampered"}, manifests)


def test_point_average_uses_existing_strict_synthetic_gate_not_benchmark_score():
    scores = {"seed1": {"macro_rmse": .98}, "seed2": {"macro_rmse": .97},
              "average": {"macro_rmse": .975, "mteb": 1.}}
    assert select_variant(scores) == "seed2"
    scores["average"]["macro_rmse"] = .97
    assert select_variant(scores) == "seed2"
    scores["average"]["macro_rmse"] = .969
    assert select_variant(scores) == "average"


def test_gpu_wait_handles_the_existing_string_pid_interface(tmp_path, monkeypatch):
    outputs = iter(("724908\n724909", ""))
    waits = []
    monkeypatch.setattr(runner, "gpu_processes", lambda: next(outputs))
    monkeypatch.setattr(runner.time, "sleep", waits.append)
    runner.wait_for_idle_gpu(tmp_path)
    assert waits == [120]
    assert json.loads((tmp_path / "status.json").read_text())["observed_gpu_pids"] == "724908\n724909"
