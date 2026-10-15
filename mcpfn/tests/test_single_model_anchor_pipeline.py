import ast
import os

import pytest

import run_single_model_anchor_rehearsal_pipeline as pipeline
import train_single_matrixformer_anchor_rehearsal as training


def identity():
    return {"pid": os.getpid(), "start_ticks": 42,
            "command": "bash -c set -e; python -u scripts/train_single_matrixformer_anchor_rehearsal.py "
                       "--stage train; " + pipeline.AUDITOR_COMMAND}


def test_original_chain_and_exact_final_auditor_exec_remain_valid(monkeypatch):
    expected = identity()
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: expected)
    assert pipeline.chain_is_live(expected)
    observed = {**expected, "command": pipeline.AUDITOR_COMMAND}
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: observed)
    monkeypatch.chdir(pipeline.REPO)
    assert pipeline.chain_is_live(expected)


@pytest.mark.parametrize("mutation", ["missing", "pid", "start", "auditor", "cwd", "suffix", "trainer"])
def test_dead_reused_or_unexpected_chain_is_rejected(monkeypatch, tmp_path, mutation):
    expected = identity()
    observed = {**expected, "command": pipeline.AUDITOR_COMMAND}
    monkeypatch.chdir(pipeline.REPO)
    if mutation == "missing":
        observed = None
    elif mutation == "pid":
        observed["pid"] += 1
    elif mutation == "start":
        observed["start_ticks"] += 1
    elif mutation == "auditor":
        observed["command"] += " --changed"
    elif mutation == "cwd":
        monkeypatch.chdir(tmp_path)
    elif mutation == "suffix":
        expected["command"] += "; unexpected-command"
    else:
        expected["command"] = expected["command"].replace("anchor_rehearsal.py --stage train", "different.py --stage train")
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: observed)
    assert not pipeline.chain_is_live(expected)


def test_all_scientific_gates_case_counts_and_complete_campaign_remain_unchanged():
    assert pipeline.training is training and pipeline.training_audit.training is training
    assert pipeline.REPLAY.name == "anchor_rehearsal_v28_endpoint_replay"
    assert pipeline.CONFIRMATION.name == "anchor_rehearsal_v28_confirmation"
    assert pipeline.AUDITOR_COMMAND == "python scripts/audit_single_model_anchor_rehearsal.py"
    assert pipeline.STAGE_WRITE_BUDGETS == pipeline.template.STAGE_WRITE_BUDGETS
    assert pipeline.MINIMUM_FREE == pipeline.template.MINIMUM_FREE
    source = pipeline.ADAPTED_SOURCE
    assert 'audit["through_step"] != 64' in source
    assert 'audit["training_episodes"] != 1792' in source
    assert 'audit["prediction_grids"] != 1368' in source
    assert 'result["frozen"]["frozen_tensors"] != 192' in source
    trees = [ast.parse(module.ADAPTED_SOURCE) for module in (pipeline, pipeline.template)]
    for name in ("replay", "audit_replay", "confirm", "audit_confirmation", "run"):
        functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name) for tree in trees]
        assert ast.dump(functions[0]) == ast.dump(functions[1])
    for name in ("register_chain_identity", "wait_for_training_audit", "require_compact_audit", "materialize_endpoint_tasks"):
        assert getattr(pipeline, name).__globals__ is pipeline.__dict__
    assert "chain_is_live" in pipeline.wait_for_training_audit.__code__.co_names
    assert "teacher_policy" in pipeline.materialize_endpoint_tasks.__code__.co_names
    assert "teacher_policy" in pipeline.require_compact_audit.__code__.co_names
