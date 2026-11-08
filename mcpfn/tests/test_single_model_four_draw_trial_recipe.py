import ast
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import time
import weakref

import pytest
import torch

import single_model_four_draw_trial_recipe as recipe
import train_single_matrixformer_balanced_native_long as original


def training_tree():
    return ast.parse(recipe.adapt_training(original.ADAPTED_SOURCE))


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def constants(tree):
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and type(node.value) is int}


def test_all_validation_selection_and_checkpoint_functions_are_unchanged():
    before, after = ast.parse(original.ADAPTED_SOURCE), training_tree()
    for name in ("confirmation_cases", "selected_boundary", "checkpoint", "validate", "setup"):
        assert ast.dump(function(before, name)) == ast.dump(function(after, name))
    assert "all22 original V8 guards" in recipe.adapt_training(original.ADAPTED_SOURCE)
    assert "284 fresh reserved synthetic cases" in recipe.adapt_training(original.ADAPTED_SOURCE)
    assert "ineligible step0 and earlier-boundary ties" in recipe.adapt_training(original.ADAPTED_SOURCE)


def test_no_gpu_capture_bank_or_inherited84_row_capacity_remains():
    tree = training_tree()
    train = function(tree, "train")
    assert 84 not in constants(train)
    calls = [node for node in ast.walk(train) if isinstance(node, ast.Call)]
    allocations = [node for node in calls if ast.unparse(node.func) == "torch.empty"]
    assert len(allocations) == 1
    assert ast.literal_eval(allocations[0].args[0].elts[0]) == 168
    assert next(ast.literal_eval(keyword.value) for keyword in allocations[0].keywords if keyword.arg == "device") == "cpu"
    captures = [node for node in calls if ast.unparse(node.func) == "compact_cache.capture_batch"]
    assert len(captures) == 1 and ast.unparse(captures[0].args[-1]) == "captured"
    transfers = [node for node in calls if ast.unparse(node.func) == "staging.stage_for_update"]
    assert len(transfers) == 1 and ast.literal_eval(transfers[0].args[-1]) == "cuda"


def test_rejected_admission_preserves_the_previous_actual_gradient_bank():
    train = function(training_tree(), "train")
    admission = next(node for node in ast.walk(train) if isinstance(node, ast.If) and ast.unparse(node.test) == "checker.eligible")
    assert ast.unparse(admission.body[0]) == "gradients = None"
    assert not any(isinstance(node, ast.Name) and node.id in {"gradients", "captured", "transfer_path"}
        for statement in admission.orelse for node in ast.walk(statement))
    source = ast.unparse(admission)
    assert source.index("staging.stage_for_update") < source.index("del captured") < source.index("paired.protected_update")
    assert source.index("write_json(transfer_path, transfer)") < source.index("paired.protected_update")
    assert "save_last_endpoint(output, last_endpoint, gradients, step, accepted)" in ast.unparse(train)


def test_actual_optimizer_call_and_original_failure_evidence_are_preserved():
    before, after = ast.parse(original.ADAPTED_SOURCE), training_tree()
    for call_name in ("paired.protected_update", "save_endpoint", "compact_cache.evaluate_training"):
        calls = []
        for tree in (before, after):
            calls.append([ast.dump(node) for node in ast.walk(function(tree, "train"))
                if isinstance(node, ast.Call) and ast.unparse(node.func) == call_name])
        assert calls[0] == calls[1]


def test_registration_keeps_original_seed_admission_and_records_new_scientific_fields():
    source = recipe.adapt_registration(original.REGISTER_SOURCE)
    assert "len(fresh) != 14620" in source
    assert '"four_draw_batching": batching.SPECIFICATION' in source
    assert '"gradient_staging_policy": staging.POLICY' in source
    verifier = ast.unparse(function(training_tree(), "verify_registration"))
    assert "config['four_draw_batching'] != batching.SPECIFICATION" in verifier
    assert "config['gradient_staging_policy'] != staging.POLICY" in verifier
    assert 'test_single_model_four_draw*.py' in source


def test_complete_fixed_schedule_matches_prospective_metadata_not_probe_stress_override():
    assignments = {}
    for node in training_tree().body:
        if isinstance(node, ast.Assign):
            try:
                assignments[ast.unparse(node.targets[0])] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    assert assignments['(STEPS, BOUNDARIES)'] == (recipe.budget.STEPS, recipe.budget.BOUNDARIES)
    assert assignments['(DATA_SEED, GRADIENT_SEED)'] == (recipe.budget.DATA_SEED, recipe.budget.GRADIENT_SEED)
    assert assignments['CONFIRMATION_SEEDS'] == recipe.budget.CONFIRMATION_SEEDS
    assert "NATIVE_SHAPE" not in recipe.adapt_training(original.ADAPTED_SOURCE)


@pytest.mark.parametrize("eligibility", [(True, False), (True, False, True), (False, False)])
def test_actual_adapted_loop_releases_cpu_capture_and_retains_last_defined_bank(tmp_path, eligibility):
    train = function(training_tree(), "train")
    loop = next(node for node in ast.walk(train) if isinstance(node, ast.For) and ast.unparse(node.target) == "batch")
    final = next(node for node in ast.walk(train) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "save_last_endpoint")
    prefix = ast.parse("gradients = None\nlast_endpoint = None\naccepted = 0\nstop = False\nbest = {}\n").body
    runner = ast.FunctionDef(name="run_loop", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=prefix + [deepcopy(loop), deepcopy(final)], decorator_list=[])
    program = ast.fix_missing_locations(ast.Module(body=[runner], type_ignores=[]))
    captured_reference, gradient_reference, observed = [], [], []
    writes, records, endpoint = {}, [], []
    config = {"base": {"query_seed_offset": 1, "memory_fraction": .9}, "batches": [{"step": index + 1,
        "tasks": [{"weight": 1., "index": 0}], "eligible": eligible} for index, eligible in enumerate(eligibility)]}

    def capture(output, directory, batch, model, optimizer, criterion, teachers, checker, bank):
        assert bank.device.type == "cpu" and tuple(bank.shape) == (168, 3)
        assert not gradient_reference or gradient_reference[-1]() is None
        bank.fill_(batch["step"])
        captured_reference.append(weakref.ref(bank))
        observed.append(batch["step"])
        return [{"index": 0}], batch["tasks"], {}

    def transfer(bank, entries, tasks, before, device):
        assert device == "cuda" and bank is captured_reference[-1]()
        gradients = bank[:len(entries)].clone()
        gradient_reference.append(weakref.ref(gradients))
        return gradients, {"defined_rows": len(entries)}

    def protected(view, optimizer, gradients, weights, rate, before, evaluate):
        assert captured_reference[-1]() is None
        assert torch.equal(gradients, gradient_reference[-1]())
        return {"accepted": True, "reason": "test"}, {}

    def save(output, last, gradients, step, accepted):
        if any(eligibility):
            assert gradients is gradient_reference[-1]()
            assert last["batch"]["step"] == max(index + 1 for index, eligible in enumerate(eligibility) if eligible)
        else:
            assert last is gradients is None
        endpoint.append((step, accepted))

    cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None, empty_cache=lambda: None,
        max_memory_allocated=lambda: 1, max_memory_reserved=lambda: 2,
        get_device_properties=lambda device: SimpleNamespace(total_memory=10))
    namespace = {"config": config, "output": tmp_path, "SCRATCH": tmp_path, "selected": [["tiny", [3], 3]],
        "torch": SimpleNamespace(cuda=cuda, empty=torch.empty, float32=torch.float32), "time": time,
        "resource": SimpleNamespace(getrusage=lambda who: SimpleNamespace(ru_maxrss=1), RUSAGE_SELF=0),
        "admission": SimpleNamespace(TrainingBatchAdmission=lambda sampler, batch: SimpleNamespace(
            eligible=batch["eligible"], record={"eligible": batch["eligible"]}, assert_consumed=lambda: None)),
        "write_json": lambda path, payload: writes.update({str(path): payload}), "sampler": None,
        "retention_training": SimpleNamespace(directory=lambda scratch, step: scratch / f"retention_{step}"),
        "retention": SimpleNamespace(runtime=lambda *args, **kwargs: nullcontext()), "retention_reference": None,
        "retention_identity": None, "full_network": SimpleNamespace(capture_coordinates=lambda *args: nullcontext()),
        "original": SimpleNamespace(alignment=SimpleNamespace(mixture=SimpleNamespace(backend=None)),
            guarded=SimpleNamespace(StepEvaluationError=RuntimeError)),
        "compact_cache": SimpleNamespace(capture_batch=capture, evaluate_training=lambda *args: None),
        "paired": SimpleNamespace(validate_layout=lambda *args: None, protected_update=protected),
        "staging": SimpleNamespace(stage_for_update=transfer), "sha256": lambda path: "digest",
        "constant_rate": lambda step: 5e-6, "status": lambda *args, **kwargs: None,
        "append_json": lambda path, payload: records.append(payload), "print": lambda *args, **kwargs: None,
        "json": SimpleNamespace(dumps=lambda payload: ""), "BOUNDARIES": (), "save_last_endpoint": save,
        **dict.fromkeys(("model", "optimizer", "criterion", "teachers", "view")),
    }
    exec(compile(program, "<four-draw-training-loop-test>", "exec"), namespace)
    namespace["run_loop"]()
    assert endpoint == [(len(eligibility), sum(eligibility))]
    assert observed == [index + 1 for index, eligible in enumerate(eligibility) if eligible]
    assert ["gradient_transfer" in row for row in records] == list(eligibility)
    assert len([path for path in writes if "/gradient_transfers/" in path]) == sum(eligibility)
