import ast
from contextlib import contextmanager
from copy import deepcopy
import inspect

import pytest

import train_single_matrixformer_anchor_retention as training
import test_single_model_full_network_trial as original_tests
import test_single_model_legacy_strata_training as loop_tests
from single_model_registered_adaptation import replace_once


def test_original_network_tasks_losses_gates_and_fresh_seed_scope_are_retained():
    assert training.PARENT == training.predecessor.PARENT == training.preflight.PARENT
    assert training.PARENT_SHA256 == training.retention.REFERENCE_SHA256
    assert training.PARENT_UPDATES == 98 and training.RATE == 5e-6 and training.STEPS == 64
    assert training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert training.paired is training.predecessor.paired and training.teacher_policy is training.predecessor.teacher_policy
    assert training.TRIAL_QUALIFICATION == training.predecessor.TRIAL_QUALIFICATION
    assert training.DATA_SEED == 881100901 and training.GRADIENT_SEED == 882100901
    assert training.CONFIRMATION_SEEDS == (891100901, 892100901)
    assert training.POLICY == training.ANCHOR_POLICY
    before = training.original.alignment.fixed.seeds_in([training.predecessor.batches(), training.predecessor.confirmation_cases(), training.preflight.batch()])
    after = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    assert len(after) == 3868 and not before.intersection(after)
    for batch, original in zip(training.batches(), training.predecessor.batches()):
        for task, prior in zip(batch["tasks"], original["tasks"]):
            copied = deepcopy(task)
            copied["gradient_seed"] = prior["gradient_seed"]
            copied["specification"].get("structured", copied["specification"])["seed"] = prior["specification"].get("structured", prior["specification"])["seed"]
            assert copied == prior
    for volume, amounts in training.STORAGE_ACCOUNTING.items():
        assert sum(amounts.values()) <= training.diagnostics.WRITE_BUDGETS["training"][volume]
    assert training.diagnostics.RESERVES == training.predecessor.diagnostics.RESERVES


def test_actual_compact_capture_and_training_loop_reach_retention_without_changing_validation():
    assert training.compact_cache._capture.__globals__["alignment"].mixture.backend is training.original.alignment.mixture.backend
    functions = {node.name: ast.get_source_segment(training.ADAPTED_SOURCE, node)
                 for node in ast.parse(training.ADAPTED_SOURCE).body if isinstance(node, ast.FunctionDef)}
    run = functions["train"]
    assert run.index("retention_reference =") < run.index('for batch in config["batches"]')
    assert "with retention.runtime(original.alignment.mixture.backend" in run
    assert "retention_manifest_sha256=" in run
    assert "retention_reference" not in functions["validate"]
    assert "cap90_recipe.validate_base" in functions["verify_registration"]
    assert 'config["retention_storage"] != retention_training.layout(SCRATCH)' in functions["verify_registration"]
    assert "retention_training.prepare(SCRATCH)" in training.REGISTER_SOURCE


SOURCE = inspect.getsource(loop_tests.test_actual_training_loop_keeps_last_real_endpoint_across_noops)
SOURCE = replace_once(SOURCE, '"base": {},', '"base": {"query_seed_offset": 5},')
namespace = {**vars(loop_tests), "training": training}
exec(compile(SOURCE, __file__, "exec"), namespace)
run_original_loop_test = namespace["test_actual_training_loop_keeps_last_real_endpoint_across_noops"]


@pytest.mark.parametrize("eligible_steps,accepted_steps", [({1, 3}, {1}), ({1}, {1}), (set(), set()), ({4}, {4})])
def test_full_loop_keeps_frozen_reference_and_exact_noop_rollback_semantics(tmp_path, monkeypatch, eligible_steps, accepted_steps):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    training.retention_training.prepare(scratch)
    monkeypatch.setattr(training, "SCRATCH", scratch)
    monkeypatch.setattr(training, "top_two", training.full_network, raising=False)
    monkeypatch.setattr(training.retention_training, "identity", lambda config: {})
    references, used = [], []

    def reference(model, identity):
        result = deepcopy(model).eval().requires_grad_(False)
        references.append(result)
        return result

    @contextmanager
    def runtime(backend, directory, current, identity, *, query_seed_offset):
        assert current is references[0] and current.decoder.weight.item() == 0. and query_seed_offset == 5
        directory.mkdir()
        used.append(directory.name)
        yield
        training.write_json(directory / "manifest.json", {"complete": True})

    monkeypatch.setattr(training.retention_training, "frozen_reference", reference)
    monkeypatch.setattr(training.retention, "runtime", runtime)
    run_original_loop_test(tmp_path, monkeypatch, eligible_steps, accepted_steps)
    assert len(references) == 1 and used == [f"step_{step:03d}" for step in sorted(eligible_steps)]
    for line in (tmp_path / "training.jsonl").read_text().splitlines():
        row = training.json.loads(line)
        if row["step"] in eligible_steps:
            assert row["retention_manifest_sha256"] == training.sha256(training.Path(row["retention_manifest"]))
        else:
            assert "retention_manifest" not in row


@pytest.mark.parametrize("entry,path,value", [
    (0, ("record", "accepted"), False), (0, ("trainable_parameters",), 36036497),
    (0, ("frozen", "frozen_tensors"), 164), (1, ("training_tasks",), 27),
    (1, ("prediction_grids",), 55), (1, ("teacher_replay", "bitwise_equal"), False),
    (1, ("raw_endpoint", "gradient_vectors"), 28), (2, ("passed",), False), (3, ("state",), "running")])
def test_original_preflight_qualification_checks_remain_mandatory(entry, path, value):
    records = original_tests.valid_preflight()
    records[-1].pop("accepted")
    records[-1]["retention_audited"] = True
    training.validate_preflight(*records, "digest")
    target = records[entry]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        training.validate_preflight(*records, "digest")


@pytest.mark.parametrize("mutation", [None, "missing_extra_audit", "old_status", "result_rejected", "audit_rejected", "finite_rejected"])
def test_real_probe_status_schema_does_not_weaken_any_scientific_acceptance(mutation):
    records = original_tests.valid_preflight()
    records[-1].pop("accepted")
    records[-1]["retention_audited"] = True
    if mutation == "missing_extra_audit":
        records[-1]["retention_audited"] = False
    elif mutation == "old_status":
        records[-1].pop("retention_audited")
        records[-1]["accepted"] = True
    elif mutation == "result_rejected":
        records[0]["record"]["accepted"] = False
    elif mutation == "audit_rejected":
        records[1]["raw_endpoint"]["accepted"] = False
    elif mutation == "finite_rejected":
        records[0]["record"]["finite"]["accepted"] = False
    if mutation:
        with pytest.raises(ValueError):
            training.validate_preflight(*records, "digest")
    else:
        training.validate_preflight(*records, "digest")
    assert 'current.get("retention_audited") is not True' in training.BASE_PREFLIGHT_SOURCE
    assert 'result["record"]["accepted"] is not True' in training.BASE_PREFLIGHT_SOURCE
    assert 'audit["raw_endpoint"]["accepted"] is not True' in training.BASE_PREFLIGHT_SOURCE
