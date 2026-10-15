import builtins
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_single_model_point_only_pipeline as pipeline
import test_single_model_legacy_point_pipeline as inherited
from recover_single_model_legacy_strata_replay import code_globals


def test_complete_qualification_preserves_scope_gates_and_recursive_dependencies():
    assert pipeline.training.OUTPUT.name == "point_only_v32"
    assert pipeline.OUTPUT.name == "point_only_v32_pipeline"
    assert pipeline.REPLAY.name == "point_only_v32_endpoint_replay"
    assert pipeline.CONFIRMATION.name == "point_only_v32_confirmation"
    assert len(pipeline.training.confirmation_cases()) == 284
    assert pipeline.PARENT_CACHE == pipeline.template.PARENT_CACHE
    assert pipeline.STAGE_WRITE_BUDGETS == pipeline.template.STAGE_WRITE_BUDGETS
    assert pipeline.MINIMUM_FREE == 18 * 1024**3 and pipeline.deepcopy is deepcopy
    for module in (pipeline, pipeline.training, pipeline.training_audit):
        for name, function in vars(module).items():
            if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(module):
                assert not code_globals(inspect.unwrap(function).__code__) - vars(module).keys() - vars(builtins).keys(), name
    for text in ('"prediction_grids": count, "cases": 284', 'run_single_matrixformer_cached_benchmarks.py',
                 'review_single_matrixformer_campaign.py', 'len(report["cases"]) != 152', '"research_selection_tolerance": 0',
                 'training.paired.paired_displacement(', 'require_compact_audit(audit)'):
        assert text in pipeline.ADAPTED_SOURCE
    assert 'training.paired.strata.paired_displacement(' not in pipeline.ADAPTED_SOURCE
    assert pipeline.no_confirmation_reason.__code__.co_code == pipeline.template.no_confirmation_reason.__code__.co_code


def test_source_and_confirmation_identities_survive_fresh_interpreter():
    expected = {"pipeline": pipeline.ADAPTED_SOURCE_SHA256, "training": pipeline.training.ADAPTED_SOURCE_SHA256,
                "audit": pipeline.training_audit.ADAPTED_SOURCE_SHA256, "cases": pipeline.training.confirmation_cases()}
    command = ("import json, runpy; module=runpy.run_path('scripts/run_single_model_point_only_pipeline.py', run_name='identity_test'); "
               "print(json.dumps({'pipeline':module['ADAPTED_SOURCE_SHA256'], 'training':module['training'].ADAPTED_SOURCE_SHA256, "
               "'audit':module['training_audit'].ADAPTED_SOURCE_SHA256, 'cases':module['training'].confirmation_cases()}))")
    actual = subprocess.check_output([sys.executable, "-c", command], cwd=pipeline.REPO,
        env={**os.environ, "PYTHONHASHSEED": "125", "CUDA_VISIBLE_DEVICES": ""}, text=True)
    assert json.loads(actual) == expected


def test_replay_resolves_every_training_proposal_attribute(monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_replay_resolves_every_training_proposal_attribute()


def test_actual_endpoint_materialization_executes_all28_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_actual_endpoint_materialization_executes_all28_tasks(tmp_path, monkeypatch)


def test_actual_endpoint_can_precede_terminal_attempt(monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_actual_endpoint_can_precede_terminal_attempt(monkeypatch)


def test_empty_endpoint_does_not_infer_or_select(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_empty_endpoint_does_not_infer_or_select(tmp_path, monkeypatch)


@pytest.mark.parametrize("mutation", [None, "fixed_task_count", "omitted_admission", "rating_count"])
def test_exact_independent_admission_and_task_counts_required(monkeypatch, mutation):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_exact_independent_admission_and_task_counts_required(monkeypatch, mutation)


def test_only_exact_registered_shell_exec_handoff_is_accepted(monkeypatch, tmp_path):
    expected = {"pid": os.getpid(), "start_ticks": 42, "command":
                "bash -c python scripts/train_single_matrixformer_point_only.py --stage train; " + pipeline.AUDITOR_COMMAND}
    observed = {**expected, "command": pipeline.AUDITOR_COMMAND}
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: observed)
    monkeypatch.setattr(pipeline, "REPO", Path.cwd())
    assert pipeline.chain_is_live(expected)
    observed["start_ticks"] += 1
    assert not pipeline.chain_is_live(expected)
    observed["start_ticks"] -= 1
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    assert not pipeline.chain_is_live(expected)
