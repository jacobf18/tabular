import ast
from copy import deepcopy
import math

import pytest

import audit_single_model_anchor_rehearsal as audit
import train_single_matrixformer_anchor_rehearsal as training
from test_single_model_top_two_blocks_trial import valid_preflight


def test_every_training_draw_is_preserved_except_registered_anchor_stratum_and_fresh_seeds():
    batches, previous = training.batches(), training.template.batches()
    assert len(batches) == len(previous) == 64
    assert training.BOUNDARIES == (0, 10, 25, 50, 64)
    for batch, reference in zip(batches, previous):
        assert batch["step"] == reference["step"] and len(batch["tasks"]) == 28
        assert math.fsum(task["weight"] for task in batch["tasks"]) == 1.
        for task, original in zip(batch["tasks"], reference["tasks"]):
            restored = deepcopy(task)
            restored["gradient_seed"] = original["gradient_seed"]
            specification = restored["specification"].get("structured", restored["specification"])
            original_specification = original["specification"].get("structured", original["specification"])
            specification["seed"] = original_specification["seed"]
            if task["replicate"] == 1 and task["specification"]["domain"] == "imputation":
                assert specification["component"] == ("anchor_latent_mcar" if "structured" in task["specification"] else 6)
                specification["component"] = original_specification["component"]
            assert restored == original
    seeds = training.original.alignment.fixed.seeds_in(batches)
    confirmation = training.original.alignment.fixed.seeds_in(training.confirmation_cases())
    used = training.original.alignment.fixed.seeds_in([previous, training.template.confirmation_cases(),
        training.preflight.batch(), training.preflight.template.batch()])
    assert len(seeds) == 3584 and len(confirmation) == 284
    assert not seeds.intersection(confirmation) and not (seeds | confirmation).intersection(used)
    assert (training.DATA_SEED, training.GRADIENT_SEED) == (631100901, 632100901)
    assert training.CONFIRMATION_SEEDS == (641100901, 642100901)


def test_actual_selected_v27_initializer_not_terminal_or_probe_and_original_baselines_are_bound():
    assert training.PARENT == training.template.OUTPUT / "selected_checkpoint.pth" == training.preflight.PARENT
    assert training.PARENT_SHA256 == training.preflight.PARENT_SHA256
    assert training.PARENT_UPDATES == 64 + 9 == 73
    assert training.teacher_policy.POLICY == {"legacy": "general", "structured": "general", "recsys": "specialist"}
    assert 'read(INITIAL_RUN / "validation_10.json")["student"]' in training.ADAPTED_SOURCE
    assert '"parent_training_updates": PARENT_UPDATES' in training.REGISTER_SOURCE
    assert '"retention_baseline": str(original.alignment.fixed.PARENT)' in training.REGISTER_SOURCE
    assert 'sha256(PARENT) != PARENT_SHA256' in training.REGISTER_SOURCE
    assert training.PARENT != training.original.alignment.fixed.PARENT


def test_optimizer_math_selection_and_confirmation_do_not_change_with_training_prior():
    trees = [ast.parse(module.ADAPTED_SOURCE) for module in (training, training.template)]
    for name in ("setup", "train", "selected_boundary", "confirmation_cases", "batches"):
        functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name) for tree in trees]
        assert ast.dump(functions[0]) == ast.dump(functions[1])
    assert training.PREFIXES == ('transformer_blocks.6.', 'transformer_blocks.7.', 'decoder.')
    assert all(training.constant_rate(step) == 5e-6 for step in range(1, 65))
    for name in ("replicated_batches", "require_evidence", "validate_preflight", "save_endpoint", "audit_endpoint"):
        assert getattr(training, name).__globals__ is training.__dict__


@pytest.mark.parametrize("step", [0, 65, 1.5, True])
def test_no_rate_or_attempt_budget_drift(step):
    with pytest.raises(ValueError):
        training.constant_rate(step)


@pytest.mark.parametrize("mutation", [None, "rejected", "old_rate", "teacher", "gradients", "cuda", "state"])
def test_only_complete_accepted_half_rate_probe_can_authorize_training(mutation):
    result, report, gate, current = valid_preflight()
    result["record"]["rate"] = 5e-6
    if mutation == "rejected":
        result["record"]["accepted"] = False
    elif mutation == "old_rate":
        result["record"]["rate"] = 1e-5
    elif mutation == "teacher":
        report["teacher_replay"]["bitwise_equal"] = False
    elif mutation == "gradients":
        report["raw_endpoint"]["gradient_vectors"] -= 1
    elif mutation == "cuda":
        report["cuda_initialized"] = True
    elif mutation == "state":
        current["state"] = "running"
    if mutation is None:
        training.validate_preflight(result, report, gate, current, "digest")
    else:
        with pytest.raises(ValueError):
            training.validate_preflight(result, report, gate, current, "digest")


def test_independent_audit_preserves_every_original_and_actual_parent_reference_grid():
    source = audit.ADAPTED_SOURCE
    assert audit.training is training
    assert 'saved["parent_training_updates"] != training.PARENT_UPDATES' in source
    assert 'training.INITIAL_RUN / "validation_10.json"' in source
    assert '(parent_report, ("student",))' in source
    assert '152 * (4 + len(boundaries))' in source
    assert 'len(selected) != 62' in source and '27766670' in source
    assert 'frozen["frozen_tensors"] != 192' in source
    assert 'len(decision["comparisons"]) != 22' in source
    assert 'training.preflight.check_routing(payload, task, config)' in source
