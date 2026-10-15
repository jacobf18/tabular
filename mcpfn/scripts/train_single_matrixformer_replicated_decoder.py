"""Train one V8-start decoder with two independent fresh draws per original cohort."""

import inspect

import train_single_matrixformer_targeted_decoder as template
import train_single_matrixformer_v8_decoder as previous
import single_model_replicated_decoder_batch as replicated
from single_model_registered_adaptation import install, replace_once


TRIAL_POLICY = ("Initialize one unchanged network directly from frozen V8, never from the replicated preflight "
                "or an unqualified tail. Freeze all nondecoder state. Retain V17's64 scheduled component/shape/"
                "regime/missingness combinations, but use two independent fresh data/backward draws per cohort "
                "at each attempt:28 tasks with half the original per-task composite weight and all individual "
                "point guards. This doubles training draws, not optimizer attempts, without claiming a paired "
                "or equal-compute causal ablation or a proven variance reduction. Preserve .25 legacy/.25 "
                "structured/.5 ratings mass, MAR/skip-logic emphasis, V8 imputation/specialist rating teachers, "
                "normalization and Huber/NLL/CDF/query objectives. Fresh decoder Adam initially has1e-5 for "
                "rollback; every actual proposal uses constant2e-5 without warmup/decay. Preserve0.1 joint "
                "protection,2x displacement bound and strict finite-point parameter/moment/preceding-rate "
                "rollback for every draw. Rejections advance to new tasks. Count parent updates since V8, "
                "not lifetime pretraining. No architecture change or inference ensemble.")
TRIAL_QUALIFICATION = replace_once(previous.QUALIFICATION, "896 cached tasks", "1792 cached tasks")
constant_rate = previous.constant_rate


def replicated_batches():
    return [replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED) for batch in previous.batches()]


install(template, globals(), {
    'import preflight_single_model_targeted_decoder as preflight': 'import preflight_single_model_replicated_decoder as preflight',
    'OUTPUT = ROOT / "targeted_decoder_v15"': 'OUTPUT = ROOT / "replicated_decoder_v18"',
    'INITIAL_RUN = ROOT / "decoder_repair_v14"': 'INITIAL_RUN = original.alignment.fixed.PRODUCER',
    'STEPS, BOUNDARIES = 100, (0, 10, 25, 50, 100)': 'STEPS, BOUNDARIES = 64, (0, 10, 25, 50, 64)',
    'DATA_SEED, GRADIENT_SEED = 241100901, 242100901': 'DATA_SEED, GRADIENT_SEED = 321100901, 322100901',
    'CONFIRMATION_SEEDS = (251100901, 252100901)': 'CONFIRMATION_SEEDS = (331100901, 332100901)',
    inspect.getsource(template.batches): 'def batches():\n    return replicated_batches()\n',
    'config["learning_rate_horizon"] != 100':
        'config["learning_rate_schedule"] != {"constant": 2e-5} or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256\n'
        '            or config["capture_source_sha256"] != replicated.CAPTURE_SOURCE_SHA256',
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
    'REPO / "tests/test_single_model_targeted_decoder_training.py"': 'REPO / "tests/test_single_model_replicated_decoder_training.py"',
    '"parent_selection_qualified": False': '"parent_selection_qualified": True',
    '"learning_rate_horizon": 100,':
        '"learning_rate_schedule": {"constant": 2e-5}, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,\n'
        '              "capture_source_sha256": replicated.CAPTURE_SOURCE_SHA256,',
    'never promotion of the V14 training input': 'never promotion of the unchanged V8 training input',
    '"parent_training_updates": 95': '"parent_training_updates": 0',
    'read(INITIAL_RUN / "validation_50.json")["student"]': 'read(INITIAL_RUN / "validation_0.json")["student"]',
    'config = verify_registration(output)\n    wait_for_gpu': 'config = verify_registration(output)\n    status(output, "initializing_training")\n    wait_for_gpu',
    'gradients = torch.empty((42, sum(row[2] for row in selected))': 'gradients = torch.empty((84, sum(row[2] for row in selected))',
    'entries, tasks, before = original.capture_batch(': 'entries, tasks, before = replicated.capture_batch(',
    'rate = learning_rate(step, 100, 1e-5, 5)': 'rate = constant_rate(step)',
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
                raise ValueError("Never restart a started registered training run")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
