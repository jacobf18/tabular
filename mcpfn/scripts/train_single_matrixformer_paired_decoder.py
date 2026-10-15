"""Train a single V8-start decoder with paired-example training protection and unchanged qualification."""

import inspect

import train_single_matrixformer_targeted_decoder as template
import train_single_matrixformer_v8_decoder as previous
import single_model_paired_protection as paired
import single_model_replicated_decoder_batch as replicated
from single_model_registered_adaptation import install, replace_once


TRIAL_POLICY = ("Initialize one ordinary network directly from original V8, not selected V18 or any preflight. "
                "Freeze all248 nondecoder state tensors. Use V18's64 scheduled28-task batches with two equally "
                "weighted independent fresh draws per cohort, .25 legacy/.25 structured/.5 ratings mass, "
                "unchanged shapes/components/regimes/missingness and MAR/skip-logic emphasis. Preserve V8 "
                "imputation/original-specialist rating teachers, normalization and Huber/NLL/CDF/query losses. "
                "Keep the original per-draw clipped composite Adam proposal. Change training protection to "
                "equal-draw cohort/objective means with0.1 inward margin and2x realized norm cap. Accept only "
                "if every actual paired point-metric mean retains and one improves. Record every individual "
                "draw score and regression; preserve truth-defined nulls. Decoder Adam initially has1e-5 for "
                "rollback, and every actual proposal uses2e-5 without warmup/decay. Rejected attempts restore "
                "parameters, moments and preceding rate exactly, then advance to new tasks. This changes "
                "training acceptance only, not development, independent-confirmation or real-benchmark gates. "
                "No architecture change, inference ensemble, equal-seed causal claim or inherited unqualified tail.")
TRIAL_QUALIFICATION = replace_once(previous.QUALIFICATION, "896 cached tasks", "1792 cached tasks")
constant_rate = previous.constant_rate


def replicated_batches():
    return [replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED) for batch in previous.batches()]


install(template, globals(), {
    'import preflight_single_model_targeted_decoder as preflight': 'import preflight_single_model_paired_decoder as preflight',
    'OUTPUT = ROOT / "targeted_decoder_v15"': 'OUTPUT = ROOT / "paired_decoder_v19"',
    'INITIAL_RUN = ROOT / "decoder_repair_v14"': 'INITIAL_RUN = original.alignment.fixed.PRODUCER',
    'STEPS, BOUNDARIES = 100, (0, 10, 25, 50, 100)': 'STEPS, BOUNDARIES = 64, (0, 10, 25, 50, 64)',
    'DATA_SEED, GRADIENT_SEED = 241100901, 242100901': 'DATA_SEED, GRADIENT_SEED = 351100901, 352100901',
    'CONFIRMATION_SEEDS = (251100901, 252100901)': 'CONFIRMATION_SEEDS = (361100901, 362100901)',
    inspect.getsource(template.batches): 'def batches():\n    return replicated_batches()\n',
    'config["learning_rate_horizon"] != 100':
        'config["learning_rate_schedule"] != {"constant": 2e-5} or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256\n'
        '            or config["capture_source_sha256"] != replicated.CAPTURE_SOURCE_SHA256\n'
        '            or config["protection_source_sha256"] != paired.PROTECTED_SOURCE_SHA256',
    'config["guarded_policy"] != original.guarded.POLICY': 'config["guarded_policy"] != paired.POLICY',
    'shutil.disk_usage(ROOT).free < 29 * 1024**3': 'shutil.disk_usage(ROOT).free < 31 * 1024**3',
    'Reserve all100 task batches, decoder endpoint, validation and18 GiB free space':
        'Reserve all64 doubled task batches,84-gradient decoder endpoint, validation and18 GiB free space',
    'audit["training_tasks"] != 14': 'audit["training_tasks"] != 28',
    'audit["prediction_grids"] != 28': 'audit["prediction_grids"] != 56',
    'audit["raw_endpoint"]["gradient_vectors"] != 42': 'audit["raw_endpoint"]["gradient_vectors"] != 84',
    'len(fresh) != 3084': 'len(fresh) != 3868',
    'Reserve all2800 new data/backward seeds': 'Reserve all3584 new data/backward seeds',
    'paths = [PARENT, INITIAL_RUN / "validation_50.json", ROOT / "decoder_repair_v14_audit/audit.json",':
        'paths = [PARENT, INITIAL_RUN / "validation_0.json",',
    'REPO / "tests/test_single_model_targeted_decoder_training.py"': 'REPO / "tests/test_single_model_paired_decoder_training.py"',
    '"parent_selection_qualified": False': '"parent_selection_qualified": True',
    '"learning_rate_horizon": 100,':
        '"learning_rate_schedule": {"constant": 2e-5}, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,\n'
        '              "capture_source_sha256": replicated.CAPTURE_SOURCE_SHA256,\n'
        '              "protection_source_sha256": paired.PROTECTED_SOURCE_SHA256,',
    '"guarded_policy": original.guarded.POLICY': '"guarded_policy": paired.POLICY',
    'never promotion of the V14 training input': 'never promotion of the unchanged V8 training input',
    '"parent_training_updates": 95': '"parent_training_updates": 0',
    'read(INITIAL_RUN / "validation_50.json")["student"]': 'read(INITIAL_RUN / "validation_0.json")["student"]',
    'config = verify_registration(output)\n    wait_for_gpu': 'config = verify_registration(output)\n    status(output, "initializing_training")\n    wait_for_gpu',
    'gradients = torch.empty((42, sum(row[2] for row in selected))': 'gradients = torch.empty((84, sum(row[2] for row in selected))',
    'entries, tasks, before = original.capture_batch(': 'entries, tasks, before = replicated.capture_batch(',
    'rate = learning_rate(step, 100, 1e-5, 5)':
        'paired.validate_layout(entries, batch["tasks"], before)\n            rate = constant_rate(step)',
    'original.guarded.protected_update(': 'paired.protected_update(',
}, {"POLICY": TRIAL_POLICY, "QUALIFICATION": TRIAL_QUALIFICATION})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(OUTPUT / "status.json")["state"] != "registered":
                raise ValueError("Never restart the registered paired-protection trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
