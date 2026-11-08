import ast
from copy import deepcopy
import inspect
import os
from pathlib import Path
import types

import pytest

import run_single_model_four_draw_pipeline as pipeline
import test_single_model_legacy_strata_pipeline as admission_tests
import test_single_model_full_network_pipeline as full_tests
import test_single_model_legacy_blend_teacher_pipeline as gate_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


for name in ("test_actual_prerequisite_requires_all_original_and_added_evidence",):
    source = inspect.getsource(getattr(gate_tests, name))
    source = source.replace('"through_step": 64', '"through_step": 128').replace('"attempted_batches": 64', '"attempted_batches": 128')
    source = source.replace("1792", "7168").replace("1368", "1824")
    exec(compile(source, __file__, "exec"), globals())
source = inspect.getsource(full_tests.test_independent_admission_counts_and_full_optimizer_scope_are_mandatory)
for before, after in (("1736", "7056"), ("1792", "7168"), ("1536", "6144"), ("24 * 62", "48 * 126"),
        ('"scheduled_attempts": 64', '"scheduled_attempts": 128'), ('"rating_contexts": 124', '"rating_contexts": 504'),
        ('"independently_regenerated_rating_contexts": 124', '"independently_regenerated_rating_contexts": 504'),
        ('report["independently_regenerated_rating_contexts"] = 128', 'report["independently_regenerated_rating_contexts"] = 512'),
        ('"top_block_training": True,', '"top_block_training": True, "top_two_block_training": True,')):
    assert before in source
    source = source.replace(before, after)
exec(compile(source, __file__, "exec"), globals())


def test_actual_wrappers_keep_no_endpoint_routing_and_scientific_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(admission_tests, "pipeline", pipeline)
    admission_tests.test_empty_endpoint_is_audited_without_fake_neural_replay_or_selection(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(pipeline, "prerequisite", lambda output: {"admission_attempts": {"last_optimizer_attempt": 127}})
    monkeypatch.setattr(pipeline, "replay_actual", lambda output: calls.append("replay"))
    monkeypatch.setattr(pipeline, "audit_replay_actual", lambda output: calls.append("audit"))
    pipeline.replay(tmp_path)
    pipeline.audit_replay(tmp_path)
    assert calls == ["replay", "audit"]


def test_core_decorated_functions_and_all_callbacks_resolve_to_new_modules():
    assert pipeline.training is pipeline.training_audit.training
    assert pipeline.training.diagnostics is pipeline.diagnostics
    assert pipeline.paired is pipeline.training.paired is pipeline.fixed_trial.protection
    assert pipeline.retention_training is pipeline.training.retention_training
    assert pipeline.blend_replay is pipeline.training.blend_replay
    for name in ("replay_actual", "confirm"):
        assert hasattr(getattr(pipeline, name), "__wrapped__")
        assert inspect.unwrap(getattr(pipeline, name)).__globals__ is vars(pipeline)
    for name, function in vars(pipeline).items():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(pipeline):
            assert not unresolved_globals(inspect.unwrap(function)), name
    assert pipeline.run.__globals__["replay"] is pipeline.replay
    assert pipeline.replay is not pipeline.replay_actual
    assert pipeline.audit_replay is not pipeline.audit_replay_actual
    for name, before, after in (("require_compact_audit", 24, 48), ("materialize_endpoint_tasks", 28, 56),
            ("check_materialized_task", 28, 56)):
        current, previous = getattr(pipeline, name).__code__, getattr(pipeline.blueprint, name).__code__
        assert current.co_code == previous.co_code
        assert current.co_consts == tuple(after if value == before else value for value in previous.co_consts)
    assert pipeline.blueprint.training.STEPS == 256


@pytest.mark.parametrize("name", ["replay_actual", "confirm"])
def test_neural_replay_and_confirmation_disable_autograd_before_inference(monkeypatch, name):
    class StopBeforeInference(Exception):
        pass

    def prerequisite(output):
        assert not pipeline.torch.is_grad_enabled()
        raise StopBeforeInference()

    monkeypatch.setattr(pipeline, "prerequisite", prerequisite)
    with pipeline.torch.enable_grad(), pytest.raises(StopBeforeInference):
        getattr(pipeline, name)(pipeline.OUTPUT)


def test_confirmation_and_full_benchmark_functions_are_ast_identical():
    for name in ("confirmation_setup", "confirm", "audit_confirmation", "run", "no_confirmation_reason"):
        def node(source):
            return next(item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == name)
        assert ast.dump(node(pipeline.ADAPTED_SOURCE)) == ast.dump(node(pipeline.blueprint.ADAPTED_SOURCE))
    assert len(pipeline.training.confirmation_cases()) == 284
    assert 'expected_count = 56 if payload["record"]["point_after"] is None else 112' in pipeline.ADAPTED_SOURCE
    for suffix in ("pipeline", "audit", "endpoint_replay", "confirmation", "benchmarks", "review"):
        assert f'ROOT / "four_draw_v57_{suffix}"' in pipeline.ADAPTED_SOURCE


def test_exact_chain_identity_and_same_process_exec_handoff(monkeypatch, tmp_path):
    expected = {"pid": os.getpid(), "start_ticks": 42, "command":
        "bash -c python scripts/train_single_matrixformer_four_draw.py --stage train; " + pipeline.AUDITOR_COMMAND}
    observed = {**expected, "command": pipeline.AUDITOR_COMMAND}
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: observed)
    monkeypatch.setattr(pipeline, "REPO", Path.cwd())
    assert pipeline.chain_is_live(expected)
    observed["start_ticks"] += 1
    assert not pipeline.chain_is_live(expected)
    observed["start_ticks"] -= 1
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    assert not pipeline.chain_is_live(expected)


def test_materializer_executes_all56_original_tasks(tmp_path, monkeypatch):
    from contextlib import nullcontext

    replay = tmp_path / "replay"
    replay.mkdir()
    monkeypatch.setattr(pipeline, "REPLAY", replay)
    monkeypatch.setattr(pipeline, "actual_endpoint_step", lambda: 127)
    tasks = [{"name": f"draw_{index}"} for index in range(56)]
    config = {"base": {}, "teacher_identities": {}, "batches": [{"tasks": []} for step in range(128)]}
    config["batches"][126]["tasks"] = tasks
    manifest = {"step": 127, "tasks": [{"name": task["name"], "index": index, "artifact": str(tmp_path / f"capsule_{index}.pt"),
        "sha256": "digest"} for index, task in enumerate(tasks)]}
    original = deepcopy(manifest)
    calls = []
    monkeypatch.setattr(pipeline.training.teacher_policy, "runtime", lambda *args: nullcontext())
    monkeypatch.setattr(pipeline.training.original.alignment.mixture.backend, "setup", lambda base: (None, None, "criterion", "teachers", "sampler"))
    monkeypatch.setattr(pipeline, "sha256", lambda path: "digest")

    def load(path, **kwargs):
        index = int(Path(path).stem.split("_")[-1])
        return {"task": tasks[index], "content_sha256": "full_digest"}

    monkeypatch.setattr(pipeline.torch, "load", load)
    monkeypatch.setattr(pipeline.compact_cache, "regenerate", lambda capsule, *args: calls.append(capsule["task"]) or {"full": capsule["task"]})
    monkeypatch.setattr(pipeline.training.original, "save_tensor", lambda payload, path: "saved_digest")
    result = pipeline.materialize_endpoint_tasks(manifest, config)
    assert calls == tasks and len(result["tasks"]) == 56 and manifest == original
    assert all(row["sha256"] == "saved_digest" for row in result["tasks"])
