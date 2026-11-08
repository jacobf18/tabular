from collections import Counter
from copy import deepcopy
import inspect

import pytest

import preflight_single_model_four_draw as probe
import preflight_single_model_balanced_native_quarter as original
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_probe_preserves_complete_fresh_pair_and_only_stresses_native_requested_shapes():
    expected = probe.four_draw.pair_batches(probe.long_budget.batches(probe.schedule_template,
        probe.DATA_SEED, probe.GRADIENT_SEED))[0]
    for task in expected["tasks"]:
        if task["cohort"] == "legacy":
            task["specification"]["shape"] = [2000, 1000]
    assert probe.batch() == expected and expected["source_steps"] == [1, 2]
    assert Counter(task["cohort"] for task in expected["tasks"]) == dict.fromkeys(probe.four_draw.COHORTS, 4)
    for family in probe.four_draw.FAMILIES:
        assert Counter(task["specification"]["structured"]["component"] for task in expected["tasks"]
            if task["cohort"] == family) == dict.fromkeys(probe.four_draw.COMPONENTS, 1)
    seeds = probe.training.alignment.fixed.seeds_in(expected)
    assert len(seeds) == 112
    assert seeds.isdisjoint(probe.training.alignment.fixed.seeds_in(original.batch()))
    assert seeds.isdisjoint(probe.training.alignment.fixed.seeds_in(probe.long_budget.batches(probe.schedule_template)))
    native = [task["specification"] for task in expected["tasks"] if task["cohort"] == "legacy"]
    assert [(spec["component"], spec["transpose"]) for spec in native] == [(0, False), (4, True), (1, False), (5, True)]


def test_all_callbacks_bind_new_namespaces_and_originals_remain_unchanged():
    for function in vars(probe).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(probe):
            assert not unresolved_globals(function), function.__name__
    for name in ("run", "audit_admitted", "retention_audit", "audit_original", "original_verify", "register_trial",
            "predictions", "replay_admitted", "replay_original_teachers", "replay_cohort_targets", "retention_records"):
        assert inspect.unwrap(getattr(probe, name)).__globals__ is vars(probe), name
    assert probe.paired is not original.paired and probe.retention is not original.retention
    assert original.paired.FRACTION == probe.paired.FRACTION == .25
    assert original.retention.REPLICAS == 2 and probe.retention.REPLICAS == 4
    assert len(original.batch()["tasks"]) == 28
    assert original.OUTPUT.name == "balanced_native_quarter_preflight_v1"
    assert probe.reference_identities() == original.reference_identities()
    for name in ("PARENT", "PARENT_SHA256", "RATE", "TRAINABLE_PARAMETERS", "TRAINABLE_TENSORS", "FROZEN_TENSORS", "ARITHMETIC"):
        assert getattr(probe, name) == getattr(original, name), name


def test_capture_staging_precedes_original_cuda_optimizer_and_frees_cpu_bank():
    source = probe.RUN_SOURCE
    assert 'device="cpu", dtype=torch.float32)' in source
    assert source.index("replicated.capture_batch") < source.index("staging.stage_for_update")
    assert source.index("staging.stage_for_update") < source.index("del captured") < source.index("paired.protected_update")
    assert source.index("del references") < source.index("staging.stage_for_update")
    assert 'before, "cuda")' in source
    assert "gradient_transfer_sha256" in source
    assert "torch.cuda.max_memory_allocated()" in source and "torch.cuda.max_memory_reserved()" in source
    assert "Never restart a started preflight" in source
    assert "admitted_sampler.eligible" in source


def test_full_storage_and_replay_counts_are_not_inherited_two_draw_counts():
    result = probe.blend_replay.storage_accounting(probe.batch())
    assert result["blend_tasks"] == 52 and result["component_grids"] == 104
    assert result["reserved_checkpoint_bytes"] == probe.BLEND_COMPONENT_ALLOWANCE
    assert result["array_bytes"] > 8 * 4 * 2000 * 1000
    assert probe.STORAGE_ACCOUNTING["raw_gradient_bytes"] == 168 * 4 * 77998496
    assert probe.STORAGE_ACCOUNTING["atomic_raw_allowance_bytes"] > probe.STORAGE_ACCOUNTING["raw_gradient_bytes"]
    assert sum(value for key, value in probe.STORAGE_ACCOUNTING.items() if key != "checkpoint_artifact_allowance_bytes") < probe.WRITE_BUDGETS["scratch"]
    assert probe.RESERVES == original.RESERVES
    assert '"training_tasks": 56' in probe.AUDIT_ADMITTED_SOURCE
    assert '"full_grids": 56' in probe.TEACHER_REPLAY_AUDIT_SOURCE
    assert '{"v29": 4, "structured_mean": 48}' in probe.RETENTION_RECORD_SOURCE
    assert '{"v8": 48, "v29": 52}' in probe.RETENTION_REPLAY_SOURCE
    assert '"component_grids": 100' in probe.RETENTION_AUDIT_SOURCE
    assert "len(fresh) != 112" in probe.REGISTER_SOURCE
    assert "gradient_staging_policy" in probe.REGISTER_SOURCE
    assert "test_single_model_four_draw*.py" in probe.REGISTER_SOURCE


@pytest.mark.parametrize("field", ["four_draw_batching", "gradient_staging_policy", "native_coverage", "native_probe_shape",
    "structured_component_balance", "working_allowance_bytes", "reserve_bytes", "available_bytes"])
def test_prospective_configuration_cannot_change(monkeypatch, field):
    config = {"four_draw_batching": deepcopy(probe.four_draw.SPECIFICATION), "gradient_staging_policy": probe.staging.POLICY,
        "native_coverage": deepcopy(probe.native.SPECIFICATION), "native_probe_shape": list(probe.NATIVE_SHAPE),
        "structured_component_balance": deepcopy(probe.components.SPECIFICATION), "host_memory_admission": {
            "working_allowance_bytes": probe.HOST_WORKING_BYTES, "reserve_bytes": probe.HOST_RESERVE_BYTES,
            "available_bytes": probe.HOST_WORKING_BYTES + probe.HOST_RESERVE_BYTES}}
    monkeypatch.setattr(probe, "original_verify", lambda output: config)
    assert probe.verify(probe.OUTPUT) == config
    if field in config["host_memory_admission"]:
        config["host_memory_admission"][field] = 0
    else:
        config[field] = None
    with pytest.raises(ValueError, match="complete four-draw"):
        probe.verify(probe.OUTPUT)
