"""Train one decoder from the original V8 baseline after a successful actual optimizer preflight."""

from single_model_registered_adaptation import install
import train_single_matrixformer_targeted_decoder as template


TRIAL_POLICY = ("Initialize one ordinary network directly from the original frozen V8 baseline, never an "
                "unqualified V13/V14/V15/V16 tail or the V8 preflight endpoint. Freeze all nondecoder state. "
                "Use64 fresh fourteen-task batches with the unchanged V16 targeted prior weights, component "
                "cycles, shapes, missingness, domain mass, normalization, V8 imputation/specialist rating "
                "teachers and Huber/NLL/CDF/query objectives. This tests baseline initialization without "
                "claiming a paired-data causal ablation. Fresh decoder Adam initially has1e-5 for rollback; "
                "every actual proposal uses constant2e-5 without warmup or decay. Preserve0.1 joint protection, "
                "2x displacement bound and exact finite-point parameter/moment/preceding-rate rollback. "
                "Rejections advance to fresh tasks. Parent lineage counts zero accepted updates since V8, "
                "not zero lifetime pretraining updates. No architecture change or inference ensemble.")
TRIAL_QUALIFICATION = ("Independently audit all64 attempts,896 cached tasks, every scheduled development grid, "
                       "all22 zero-tolerance V8 guards and bitwise-frozen V8 backbone. The training input and "
                       "retention baseline are the same V8 checkpoint: audit its saved reference once rather "
                       "than counting it again as a separate V14 parent. Five152-case boundaries plus the "
                       "three original reference roles require1216 case/role grids. Select only a trained "
                       "strict minimax improvement at0/10/25/50/64, with earlier ties and no step0 promotion. "
                       "Independently reconstruct/replay the final restricted optimizer endpoint and replay "
                       "the sole selected152-case boundary without reselection. Confirm once on284 reserved "
                       "fresh cases against V8, then audit all1136 confirmation grids before any alias and "
                       "every unchanged frozen real suite/full review. No fallback, relaxed guard or hidden regression.")


def constant_rate(step):
    if type(step) is not int or not 1 <= step <= 64:
        raise ValueError("Require a registered attempted batch from1 through64")
    return 2e-5


install(template, globals(), {
    'import preflight_single_model_targeted_decoder as preflight': 'import preflight_single_model_v8_decoder as preflight',
    'OUTPUT = ROOT / "targeted_decoder_v15"': 'OUTPUT = ROOT / "v8_decoder_v17"',
    'INITIAL_RUN = ROOT / "decoder_repair_v14"': 'INITIAL_RUN = original.alignment.fixed.PRODUCER',
    'STEPS, BOUNDARIES = 100, (0, 10, 25, 50, 100)': 'STEPS, BOUNDARIES = 64, (0, 10, 25, 50, 64)',
    'DATA_SEED, GRADIENT_SEED = 241100901, 242100901': 'DATA_SEED, GRADIENT_SEED = 291100901, 292100901',
    'CONFIRMATION_SEEDS = (251100901, 252100901)': 'CONFIRMATION_SEEDS = (301100901, 302100901)',
    'config["learning_rate_horizon"] != 100': 'config["learning_rate_schedule"] != {"constant": 2e-5} or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256',
    'shutil.disk_usage(ROOT).free < 29 * 1024**3': 'shutil.disk_usage(ROOT).free < 27 * 1024**3',
    'Reserve all100 task batches, decoder endpoint, validation and18 GiB free space': 'Reserve all64 task batches, decoder endpoint, validation and18 GiB free space',
    'len(fresh) != 3084': 'len(fresh) != 2076',
    'Reserve all2800 new data/backward seeds': 'Reserve all1792 new data/backward seeds',
    'paths = [PARENT, INITIAL_RUN / "validation_50.json", ROOT / "decoder_repair_v14_audit/audit.json",':
        'paths = [PARENT, INITIAL_RUN / "validation_0.json",',
    'REPO / "tests/test_single_model_targeted_decoder_training.py"': 'REPO / "tests/test_single_model_v8_decoder_training.py"',
    '"parent_selection_qualified": False': '"parent_selection_qualified": True',
    '"learning_rate_horizon": 100,': '"learning_rate_schedule": {"constant": 2e-5}, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,',
    'never promotion of the V14 training input': 'never promotion of the unchanged V8 training input',
    '"parent_training_updates": 95': '"parent_training_updates": 0',
    'read(INITIAL_RUN / "validation_50.json")["student"]': 'read(INITIAL_RUN / "validation_0.json")["student"]',
    'config = verify_registration(output)\n    wait_for_gpu': 'config = verify_registration(output)\n    status(output, "initializing_training")\n    wait_for_gpu',
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
                raise ValueError("Never restart the declared V8-start decoder trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
