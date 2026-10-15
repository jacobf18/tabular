import ast
import builtins
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_single_model_full_network_pipeline as pipeline
import test_single_model_legacy_point_pipeline as inherited
from recover_single_model_legacy_strata_replay import code_globals


def test_qualification_keeps_original_gates_and_resolves_full_scope_and_storage():
    assert pipeline.training.OUTPUT.name == "full_network_v34"
    assert pipeline.OUTPUT.name == "full_network_v34_pipeline"
    assert pipeline.REPLAY.name == "full_network_v34_endpoint_replay"
    assert pipeline.CONFIRMATION.name == "full_network_v34_confirmation"
    assert len(pipeline.training.confirmation_cases()) == 284
    assert pipeline.PARENT_CACHE == pipeline.template.PARENT_CACHE
    assert pipeline.STAGE_WRITE_BUDGETS == pipeline.template.STAGE_WRITE_BUDGETS
    assert pipeline.diagnostics is pipeline.training.diagnostics
    assert pipeline.MINIMUM_FREE == 18 * 1024**3 and pipeline.deepcopy is deepcopy
    for module in (pipeline, pipeline.training, pipeline.training_audit, pipeline.diagnostics):
        for name, function in vars(module).items():
            if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(module):
                assert not code_globals(inspect.unwrap(function).__code__) - vars(module).keys() - vars(builtins).keys(), name
        if not hasattr(module, "ADAPTED_SOURCE"):
            continue
        assert 'top_three.' not in module.ADAPTED_SOURCE and 'top_two.' not in module.ADAPTED_SOURCE
        for node in ast.walk(ast.parse(module.ADAPTED_SOURCE)):
            if isinstance(node, ast.Attribute) and ast.unparse(node).startswith(("full_network.", "training.paired.", "paired.")):
                eval(compile(ast.Expression(node), "<runtime-attribute>", "eval"), vars(module))
    for text in ('"prediction_grids": count, "cases": 284', 'run_single_matrixformer_cached_benchmarks.py',
                 'review_single_matrixformer_campaign.py', 'len(report["cases"]) != 152', '"research_selection_tolerance": 0',
                 'training.paired.paired_displacement(', 'require_compact_audit(audit)'):
        assert text in pipeline.ADAPTED_SOURCE
    assert 'training.paired.strata.paired_displacement(' not in pipeline.ADAPTED_SOURCE
    assert pipeline.no_confirmation_reason.__code__.co_code == pipeline.template.no_confirmation_reason.__code__.co_code
    assert 'result["frozen"]["frozen_tensors"] != 16' in pipeline.ADAPTED_SOURCE
    assert pipeline.endpoint_setup.__globals__["full_network"] is pipeline.training.full_network


def test_source_and_confirmation_identities_survive_fresh_interpreter():
    expected = {"pipeline": pipeline.ADAPTED_SOURCE_SHA256, "training": pipeline.training.ADAPTED_SOURCE_SHA256,
                "audit": pipeline.training_audit.ADAPTED_SOURCE_SHA256, "cases": pipeline.training.confirmation_cases()}
    command = ("import json, runpy; module=runpy.run_path('scripts/run_single_model_full_network_pipeline.py', run_name='identity_test'); "
               "print(json.dumps({'pipeline':module['ADAPTED_SOURCE_SHA256'], 'training':module['training'].ADAPTED_SOURCE_SHA256, "
               "'audit':module['training_audit'].ADAPTED_SOURCE_SHA256, 'cases':module['training'].confirmation_cases()}))")
    actual = subprocess.check_output([sys.executable, "-c", command], cwd=pipeline.REPO,
        env={**os.environ, "PYTHONHASHSEED": "239", "CUDA_VISIBLE_DEVICES": ""}, text=True)
    assert json.loads(actual) == expected


def test_actual_endpoint_materialization_executes_all28_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_actual_endpoint_materialization_executes_all28_tasks(tmp_path, monkeypatch)


def test_actual_endpoint_can_precede_terminal_attempt(monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_actual_endpoint_can_precede_terminal_attempt(monkeypatch)


def test_empty_endpoint_does_not_infer_or_select(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_empty_endpoint_does_not_infer_or_select(tmp_path, monkeypatch)


@pytest.mark.parametrize("mutation", [None, "fixed_task_count", "omitted_admission", "rating_count", "old_scope"])
def test_independent_admission_counts_and_full_optimizer_scope_are_mandatory(monkeypatch, mutation):
    counts = {"scheduled_attempts": 64, "training_tasks": 1736, "rating_contexts": 124, "accepted_updates": 60}
    report = {"admission_attempts": counts, "accepted_updates": 60, "teacher_identity_checked_tasks": 1736,
              "training_teacher_policy": pipeline.training.teacher_policy.POLICY,
              "independently_regenerated_training_tasks": 1736, "independently_regenerated_rating_contexts": 124,
              "independently_regenerated_structured_draws": 1536, "exact_complete_task_and_teacher_digest_checks": True,
              "decoder_only_training": False, "top_block_training": True, "full_network_training": True,
              "autograd_replayed": False, "full_optimizer_trajectory_replayed": False}
    if mutation == "fixed_task_count":
        report["independently_regenerated_training_tasks"] = 1792
    elif mutation == "omitted_admission":
        report["independently_regenerated_structured_draws"] = 24 * 62
    elif mutation == "rating_count":
        report["independently_regenerated_rating_contexts"] = 128
    elif mutation == "old_scope":
        report.pop("full_network_training")
        report["top_three_block_training"] = True
    monkeypatch.setattr(pipeline, "sha256", lambda path: "audit")
    monkeypatch.setattr(pipeline, "read", lambda path: {"complete": True, "tasks": 1736, "audit_sha256": "audit"}
                        if path.name == "result.json" else {"state": "complete"} if path.name == "status.json"
                        else {"source_sha256": {}})
    monkeypatch.setattr(pipeline.training_audit, "read_prefix", lambda *args: ([], ""))
    monkeypatch.setattr(pipeline.training_audit, "admission_inventory", lambda *args: counts)
    if mutation:
        with pytest.raises(ValueError):
            pipeline.require_compact_audit(report)
    else:
        pipeline.require_compact_audit(report)


def test_only_exact_registered_shell_exec_handoff_is_accepted(monkeypatch, tmp_path):
    expected = {"pid": os.getpid(), "start_ticks": 42, "command":
                "bash -c python scripts/train_single_matrixformer_full_network.py --stage train; " + pipeline.AUDITOR_COMMAND}
    observed = {**expected, "command": pipeline.AUDITOR_COMMAND}
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: observed)
    monkeypatch.setattr(pipeline, "REPO", Path.cwd())
    assert pipeline.chain_is_live(expected)
    observed["start_ticks"] += 1
    assert not pipeline.chain_is_live(expected)
    observed["start_ticks"] -= 1
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    assert not pipeline.chain_is_live(expected)
