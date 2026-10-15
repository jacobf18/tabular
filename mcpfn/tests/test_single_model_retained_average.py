from copy import deepcopy

import pytest
import torch

import average_single_model_retained_tail as averaging
from recsys_evaluation_common import sha256, write_json
from test_single_model_query_registration import fixture_config


@pytest.fixture(autouse=True)
def allow_tiny_test_artifacts(monkeypatch):
    monkeypatch.setattr(averaging, "storage_guard", lambda output: None)


def test_qualification_plan_preserves_six_metric_gate_and_uses_fresh_full_cases():
    config = fixture_config()
    plan = averaging.qualification_plan(config)
    assert len(plan["cases"]) == len({case["seed"] for case in plan["cases"]}) == 92
    old = {case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]}
    assert not old & {case["seed"] for case in plan["cases"]}
    assert len(plan["primary_metrics"]) == 4 and len(plan["retention_metrics"]) == 6
    assert plan["initialization_checkpoint"] == config["parent_checkpoint"]
    assert plan["prior_regime"] == "legacy" and not plan["query_budget_applied"]
    assert not plan["fallback_to_producer_or_partial_average"]


@pytest.mark.parametrize("failure", [None, "late", "complete", "campaign"])
def test_average_registration_is_before_all_contributions_and_keeps_original_producer(tmp_path, monkeypatch, failure):
    producer, campaign, output = [tmp_path / name for name in ("producer", "campaign", "average")]
    config = fixture_config()
    write_json(producer / "config.json", config)
    producer_digest = sha256(producer / "config.json")
    write_json(producer / "registration.json", {"config_sha256": sha256(producer / "config.json")})
    write_json(producer / "status.json", {"state": "training_complete" if failure == "complete" else "training",
                                          "step": 300 if failure == "late" else 210, "pid": 1})
    write_json(campaign / "cache_launcher_config.json", {"run": "different producer" if failure == "campaign" else str(producer)})
    monkeypatch.setattr(averaging.base, "require_live", lambda status: None)
    output.mkdir()
    if failure:
        with pytest.raises(ValueError):
            averaging.prepare(producer, campaign, output)
    else:
        plan = averaging.prepare(producer, campaign, output)
        assert plan["steps"] == list(range(300, 401, 10))
        assert plan["registered_at_producer_step"] == 210
        assert plan["model"] == config["model"]
        assert sha256(producer / "config.json") == producer_digest
        assert str(averaging.Path(averaging.__file__).resolve()) in plan["source_sha256"]


def test_retained_model_is_exact_and_binds_original_checkpoint_receipt(tmp_path):
    payload = {"step": 300, "config_sha256": "configuration", "model": {"weight": torch.tensor([1., 2.])},
               "optimizer": {"not_retained": True}}
    config = {"steps": [300, 310], "producer_config_sha256": "configuration"}
    original = deepcopy(payload)
    records = []
    averaging.retain_payload(payload, "checkpoint_digest", config, tmp_path, records)
    saved = torch.load(records[0]["path"], map_location="cpu", weights_only=True)
    torch.testing.assert_close(saved["model"]["weight"], original["model"]["weight"], rtol=0, atol=0)
    assert "optimizer" not in saved and saved["producer_checkpoint_sha256"] == "checkpoint_digest"
    assert sha256(averaging.Path(records[0]["path"])) == records[0]["sha256"]
    assert payload["optimizer"] == original["optimizer"]
    with pytest.raises(ValueError):
        averaging.retain_payload(payload, "checkpoint_digest", config, tmp_path, records)


def test_partial_or_wrong_producer_state_cannot_be_collected(tmp_path):
    config = {"steps": [300], "producer_config_sha256": "expected"}
    for step, fingerprint in ((310, "expected"), (300, "wrong")):
        with pytest.raises(ValueError):
            averaging.retain_payload({"step": step, "config_sha256": fingerprint}, "hash", config, tmp_path, [])


@pytest.mark.parametrize("failure", [False, True])
def test_completion_is_published_only_after_retention_checks_and_callbacks_restore(tmp_path, monkeypatch, failure):
    config = {"steps": [300], "producer_config_sha256": "expected"}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "registration.json", {"config_sha256": sha256(tmp_path / "config.json")})
    write_json(tmp_path / "status.json", {"state": "waiting_for_step"})
    payload = {"step": 300, "config_sha256": "expected", "model": {"weight": torch.ones(2)}}
    loader = lambda path: (payload, "source_hash")
    monkeypatch.setattr(averaging.base, "load_stable", loader)
    writer = averaging.base.write_json

    def collect(current, output):
        averaging.base.load_stable(output / "fixture.pth")
        averaging.base.write_json(output / "receipts.json", {"snapshots": [{"checkpoint_sha256": "source_hash"}]})
        averaging.base.write_json(output / "status.json", {"state": "complete"})
        assert averaging.base.read(output / "status.json")["state"] == "waiting_for_step"
        if failure:
            raise RuntimeError("collector failed before complete retention")

    monkeypatch.setattr(averaging.base, "execute", collect)
    if failure:
        with pytest.raises(RuntimeError):
            averaging.execute(config, tmp_path)
        assert not (tmp_path / "retention_result.json").exists()
    else:
        averaging.execute(config, tmp_path)
        assert averaging.base.read(tmp_path / "retention_result.json")["complete"]
        assert averaging.base.read(tmp_path / "status.json")["state"] == "complete"
    assert averaging.base.load_stable is loader and averaging.base.write_json is writer
