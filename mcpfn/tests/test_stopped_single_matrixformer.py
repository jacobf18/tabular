import inspect
import pytest

import evaluate_stopped_single_matrixformer as evaluation


@pytest.mark.parametrize("stage", evaluation.STAGES)
def test_only_authorization_changes(stage):
    module, function, source = evaluation.evaluator(stage)
    original = inspect.getsource(module.evaluate)
    boundary = '    manifest_path = output / "manifest.json"\n'
    assert source.split(boundary)[1] == original.split(boundary)[1]
    assert "gate.json" not in source
    assert "report_only_manifest(" in source
    assert function.__globals__["audit"] is module.audit
    assert function.__globals__["load_protocol"] is module.load_protocol
    assert function.__globals__["make_model"] is module.make_model


def test_adapter_rejects_unknown_boundary():
    with pytest.raises(ValueError, match="boundary changed"):
        evaluation.adapted_source("def different():\n    pass\n", "single_checkpoint")


def test_protocol_symlinks_match_cache_import_resolution(tmp_path):
    backing = tmp_path / "storage" / "recsys"
    backing.mkdir(parents=True)
    (tmp_path / "recsys_protocol_v1").symlink_to(backing, target_is_directory=True)
    protocol = evaluation.canonical_protocol(tmp_path, "recsys")
    assert protocol == backing
    assert protocol == protocol.resolve()


@pytest.fixture
def authorization(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixed checkpoint fixture")
    config = tmp_path / "config.json"
    config.write_text('{}\n')
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{}\n')
    record = {"evaluation_only": True, "training_authorized": False, "independent_confirmation_passed": False,
              "selected_step": 25, "checkpoint": str(checkpoint), "checkpoint_sha256": evaluation.sha256(checkpoint),
              "input_sha256": {str(checkpoint): evaluation.sha256(checkpoint), str(config): evaluation.sha256(config)},
              "source_sha256": {}, "protocol_sha256": {str(protocol): evaluation.sha256(protocol)}}
    evaluation.write_json(tmp_path / "evaluation_authorization.json", record)
    return tmp_path, checkpoint, record


def test_manifest_never_claims_confirmation(authorization):
    directory, checkpoint, _ = authorization
    manifest = evaluation.report_only_manifest(directory, checkpoint, directory)
    assert manifest["evaluation_only"] is True
    assert manifest["independent_confirmation_passed"] is False
    assert manifest["student_has_one_checkpoint"] is True
    assert "confirmation" not in manifest
    assert not (directory / "gate.json").exists()


@pytest.mark.parametrize("key,value", [("training_authorized", True), ("evaluation_only", False),
                                      ("independent_confirmation_passed", True), ("selected_step", 0)])
def test_invalid_authorization_rejected(authorization, key, value):
    directory, checkpoint, record = authorization
    record[key] = value
    evaluation.write_json(directory / "evaluation_authorization.json", record)
    with pytest.raises(ValueError, match="report-only authorization"):
        evaluation.report_only_manifest(directory, checkpoint, directory)


@pytest.mark.parametrize("filename", ["model.pth", "config.json", "protocol.json"])
def test_changed_inputs_rejected(authorization, filename):
    directory, checkpoint, _ = authorization
    (directory / filename).write_bytes(b"changed")
    with pytest.raises(ValueError):
        evaluation.report_only_manifest(directory, checkpoint, directory)


def test_unknown_protocol_rejected(authorization, tmp_path):
    directory, checkpoint, _ = authorization
    other = tmp_path / "other"
    other.mkdir()
    (other / "protocol.json").write_text('{}\n')
    with pytest.raises(ValueError, match="Unregistered"):
        evaluation.report_only_manifest(other, checkpoint, directory)
