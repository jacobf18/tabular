"""Train one ordinary V57 checkpoint over the complete registered four-draw schedule."""

import argparse
import fcntl
import hashlib
import inspect
from pathlib import Path
import types

import train_single_matrixformer_balanced_native_long as blueprint
import single_model_four_draw_full_trial as all_trial
import single_model_four_draw_full_storage as diagnostics
import single_model_four_draw_trial_budget as budget
import single_model_four_draw_trial_storage as accounting
import single_model_four_draw_trial_recipe as recipe
import single_model_four_draw_trial_retention as retention_training
import single_model_four_draw_trial_blend_replay as blend_replay
import single_model_four_draw_retention as retention
import single_model_four_draw_protection as paired
import single_model_four_draw_batching as batching
import single_model_four_draw_gradient_staging as staging
import single_model_four_draw_fixed_trial as fixed_trial
import relocate_single_model_four_draw_capacity as capacity
import preflight_single_model_four_draw as preflight
from single_model_registered_adaptation import replace_assignment, replace_once


for name, value in vars(blueprint).items():
    if not name.startswith("__") and name not in globals():
        globals()[name] = value
for name, function in vars(blueprint).items():
    if inspect.isfunction(function) and function.__globals__ is vars(blueprint):
        if function.__closure__ is not None:
            raise ValueError("Never retain a closed trainer's callback namespace")
        cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
        cloned.__kwdefaults__ = function.__kwdefaults__
        globals()[name] = cloned

CAPACITY_RECEIPT_SHA256 = all_trial.CAPACITY_RECEIPT_SHA256
BASE_PREFLIGHT_SOURCE = blueprint.BASE_PREFLIGHT_SOURCE
for before, after in (('audit["training_tasks"] != 28', 'audit["training_tasks"] != 56'),
        ('audit["prediction_grids"] != 56', 'audit["prediction_grids"] != 112'),
        ('audit["teacher_replay"]["full_grids"] != 28', 'audit["teacher_replay"]["full_grids"] != 56'),
        ('audit["raw_endpoint"]["gradient_vectors"] != 84', 'audit["raw_endpoint"]["gradient_vectors"] != 167')):
    BASE_PREFLIGHT_SOURCE = replace_once(BASE_PREFLIGHT_SOURCE, before, after)
exec(compile(BASE_PREFLIGHT_SOURCE, __file__, "exec"), globals())
POINT_PREFLIGHT_SOURCE = inspect.getsource(blueprint.validate_point_preflight)
for before, after in (('def validate_preflight(', 'def validate_point_preflight('),
        ('["structured_draws"] != 24', '["structured_draws"] != 48'),
        ('len(record["projection"]["groups"]) != 30', 'len(record["projection"]["groups"]) != 34'),
        ('["gradient_vectors"] != 84', '["gradient_vectors"] != 167'),
        ('all84 proposal rows and30 point-only constraints', 'all167 defined proposal rows and34 point-only constraints')):
    POINT_PREFLIGHT_SOURCE = replace_once(POINT_PREFLIGHT_SOURCE, before, after)
exec(compile(POINT_PREFLIGHT_SOURCE, __file__, "exec"), globals())
ADAPTED_SOURCE = replace_assignment(recipe.adapt_training(blueprint.ADAPTED_SOURCE), "POLICY", all_trial.POLICY)
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, 'def verify_registration(output):\n    config = read(output / "config.json")\n',
    'def verify_registration(output):\n    config = read(output / "config.json")\n'
    '    all_trial.validate_host_admission(config["host_memory_admission"])\n')
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())
REGISTER_SOURCE = recipe.adapt_registration(blueprint.REGISTER_SOURCE)
for before, after in (("audit_single_model_balanced_native_long.py", "audit_single_model_four_draw.py"),
        ("run_single_model_balanced_native_long_pipeline.py", "run_single_model_four_draw_pipeline.py")):
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, before, after)
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, 'config = {"policy": POLICY,',
    'config = {"policy": POLICY, "host_memory_admission": preflight.require_host_space(),')
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())


def batches():
    return budget.batches(blueprint.previous_training, DATA_SEED, GRADIENT_SEED)


STORAGE_ACCOUNTING = accounting.accounting(batches())


def require_capacity():
    for volume, fields in STORAGE_ACCOUNTING.items():
        if sum(fields.values()) > diagnostics.WRITE_BUDGETS["training"][volume]:
            raise ValueError("Retain the complete requested-area and atomic168-row storage evidence")
    diagnostics.storage.require_mount()
    diagnostics.require_capacity("training")
    preflight.require_host_space()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    arguments = parser.parse_args()
    with arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == "register":
            register(OUTPUT)
        else:
            with (OUTPUT / "run.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if read(OUTPUT / "status.json")["state"] != "registered":
                    raise ValueError("Never restart a begun four-draw trial")
                try:
                    train(OUTPUT)
                except BaseException as error:
                    status(OUTPUT, "failed", error=repr(error))
                    raise


if __name__ == "__main__":
    main()
