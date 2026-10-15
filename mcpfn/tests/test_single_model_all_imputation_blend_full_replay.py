from copy import deepcopy
import inspect
from math import prod

import numpy as np
import pytest

import single_model_all_imputation_blend_full_replay as integration
import test_single_model_legacy_blend_full_replay as full_test_blueprint
from single_model_registered_adaptation import replace_once


TEST_SOURCE = inspect.getsource(full_test_blueprint)
TEST_SOURCE = replace_once(TEST_SOURCE,
    "import single_model_legacy_blend_full_replay as integration",
    "import single_model_all_imputation_blend_full_replay as integration")
TEST_SOURCE = TEST_SOURCE.replace("legacy_blend_replay", "all_imputation_blend_replay")
TEST_SOURCE = TEST_SOURCE.replace("legacy_tasks", "blend_tasks")
TEST_SOURCE = replace_once(TEST_SOURCE, 'calls == ["specialist", "general"] * 2',
    'calls == ["specialist", "general"] * 26')
TEST_SOURCE = replace_once(TEST_SOURCE, 'report["blend_tasks"] == 2 and report["raw_component_grids"] == 4',
    'report["blend_tasks"] == 26 and report["raw_component_grids"] == 52')
TEST_SOURCE = replace_once(TEST_SOURCE, 'len(report["artifact_sha256"]) == 34',
    'len(report["artifact_sha256"]) == 58')
TEST_SOURCE = replace_once(TEST_SOURCE,
    'batches[0]["tasks"][0]["specification"]["structured"] = {}',
    'batches[0]["tasks"][0]["specification"]["domain"] = "recsys"')
start = TEST_SOURCE.index('@pytest.mark.parametrize("mutation", [None, "batch_count"')
end = TEST_SOURCE.index('@pytest.mark.parametrize("field,value",', start)
TEST_SOURCE = TEST_SOURCE[:start] + TEST_SOURCE[end:]
TEST_SOURCE = TEST_SOURCE.replace("test_reopened_components_reject_every_omission_or_changed_target",
    "test_reopened_legacy_components_reject_every_omission_or_changed_target")
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


STRUCTURED_TEST_SOURCE = inspect.getsource(full_test_blueprint.test_reopened_components_reject_every_omission_or_changed_target)
STRUCTURED_TEST_SOURCE = replace_once(STRUCTURED_TEST_SOURCE,
    "def test_reopened_components_reject_every_omission_or_changed_target",
    "def test_reopened_structured_components_reject_every_omission_or_changed_target")
STRUCTURED_TEST_SOURCE = replace_once(STRUCTURED_TEST_SOURCE, "record = rows[0]", "record = rows[1]")
exec(compile(STRUCTURED_TEST_SOURCE, __file__, "exec"), globals())


@pytest.mark.parametrize("mutation", [None, "batch_count", "task_count", "imputation_count", "shape", "structured_shape"])
def test_storage_preserves_all_imputation_sources_targets_and_components(mutation):
    batches = [{**deepcopy(probe.batch()), "step": step} for step in range(1, 65)]
    if mutation == "batch_count":
        batches.pop()
    elif mutation == "task_count":
        batches[0]["tasks"].pop()
    elif mutation == "imputation_count":
        batches[0]["tasks"][0]["specification"]["domain"] = "recsys"
    elif mutation == "shape":
        batches[0]["tasks"][0]["specification"]["shape"] = [0, 128]
    elif mutation == "structured_shape":
        batches[0]["tasks"][1]["specification"]["structured"]["shape"] = [0, 128]
    if mutation:
        with pytest.raises(ValueError):
            integration.storage_accounting(batches)
    else:
        areas = [prod(task["specification"].get("structured", task["specification"])["shape"])
            for batch in batches for task in batch["tasks"] if task["specification"]["domain"] == "imputation"]
        expected = {"blend_tasks": 1664, "raw_component_grids": 3328, "cells": sum(areas),
            "array_bytes": 16 * sum(areas), "atomic_archive_bytes": 16 * max(areas),
            "serialization_and_manifest_allowance": 8 * 1024**2}
        assert integration.storage_accounting(batches) == expected
        assert integration.storage_bytes(batches) == 16 * (sum(areas) + max(areas)) + 8 * 1024**2
        accounting = integration.replay.storage_accounting(batches[0])
        assert accounting["blend_tasks"] == 26 and accounting["component_grids"] == 52
        assert accounting["array_bytes"] == 8 * sum(areas[:26])
        assert accounting["reserved_checkpoint_bytes"] == accounting["array_bytes"] + 1024**2


def test_existing_full_dimensions_fit_without_changing_any_budget_or_configuration():
    import train_single_matrixformer_uniform_v8_retention as previous_training

    batches = previous_training.batches()
    accounting = integration.storage_accounting(batches)
    assert accounting == {"blend_tasks": 1664, "raw_component_grids": 3328, "cells": 310341632,
        "array_bytes": 4965466112, "atomic_archive_bytes": 32000000,
        "serialization_and_manifest_allowance": 8388608}
    assert integration.storage_bytes(batches) == 5005854720
    original_budget = previous_training.STORAGE_ACCOUNTING["scratch"]
    total = sum(original_budget.values()) - original_budget["legacy_blend_component_allowance"] + integration.storage_bytes(batches)
    assert total == 41224310912 < previous_training.diagnostics.WRITE_BUDGETS["training"]["scratch"] == 40 * 1024**3


def test_replay_adaptation_does_not_change_predecessor_policy_or_single_teacher_callbacks():
    import single_model_legacy_blend_replay as legacy

    task = {"specification": {"domain": "imputation", "structured": {}}}
    assert integration.replay.expected_role(task) == "observed_blend"
    assert legacy.expected_role(task) == "general"
    assert integration.teacher_policy.POLICY["structured"] == "observed_blend"
    assert legacy.teacher_policy.POLICY["structured"] == "general"
    assert integration.replay.expected_role({"specification": {"domain": "recsys"}}) == "specialist"
    with pytest.raises(ValueError, match="Structured teachers"):
        integration.replay.expected_role({"specification": {"domain": "recsys", "structured": {}}})
