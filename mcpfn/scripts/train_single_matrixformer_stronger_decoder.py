"""Test a fixed stronger decoder rate on fresh synthetic tasks with unchanged qualification."""

import probe_single_model_decoder_rates as rate_probe
import recover_single_model_decoder_rate_audit as rate_recovery
from single_model_registered_adaptation import install
import train_single_matrixformer_targeted_decoder as template


TRIAL_POLICY = ("One ordinary network initialized from the audited unqualified V14 endpoint, never a preflight "
                "or rate-probe candidate. Freeze every nondecoder tensor. Use64 fresh fourteen-task batches "
                "with the exact V15 prior weights, component cycles, shapes, missingness, V8 imputation and "
                "original specialist rating teachers, normalization and loss objectives. Retain .25 legacy, "
                ".25 structured, .5 rating mass and all families; the64-step cycle covers every emphasized "
                "component/context/regime/rate combination. Fresh decoder Adam has its original1e-5 initial "
                "rollback rate, but every actual proposal uses constant2e-5, without warmup or decay. Preserve "
                "0.1 joint protection,2x displacement bound and exact finite-point rollback. Rejections advance "
                "to fresh tasks. This separately registered hypothesis follows the fully audited fixed-batch "
                "rate probe; feasibility is not evidence of generalization. No inference ensemble.")
TRIAL_QUALIFICATION = ("Independently audit all64 attempts,896 cached tasks, every scheduled development grid, "
                       "all22 zero-tolerance V8 retention guards and bitwise-frozen V14 backbone. Audit and "
                       "independently reconstruct/replay the final restricted optimizer endpoint. Select only "
                       "a trained strict minimax improvement at0/10/25/50/64, with earlier ties and no step0 "
                       "promotion. Independently replay the sole selected152-case boundary without reselection. "
                       "Confirm once on284 separately reserved fresh cases against V8, never V14; independently "
                       "audit all1136 grids before alias publication and every frozen real suite/full review. "
                       "No fallback, guard relaxation, benchmark-based selection or hidden regression.")


def constant_rate(step):
    if type(step) is not int or not 1 <= step <= 64:
        raise ValueError("Require a registered attempted batch from1 through64")
    return 2e-5


def verify_rate_evidence():
    rate_probe.verify(rate_probe.OUTPUT)
    recovery = rate_probe.read(rate_recovery.OUTPUT / "recovery.json")
    audit = rate_probe.read(rate_recovery.OUTPUT / "audit.json")
    result = rate_probe.read(rate_probe.OUTPUT / "result.json")
    if (rate_probe.read(rate_recovery.OUTPUT / "status.json")["state"] != "complete"
            or not recovery["complete"] or not recovery["original_artifacts_unchanged"]
            or recovery["neural_evaluation_repeated"] or not recovery["per_metric_decision_signs_exact"]
            or recovery["audit_sha256"] != rate_probe.sha256(rate_recovery.OUTPUT / "audit.json")
            or not audit["complete"] or audit["prediction_grids"] != 112 or audit["training_tasks"] != 14
            or audit["rates"] != [2e-6, 2e-5, 1e-4] or audit["research_selection_tolerance"] != 0
            or [row["finite_feasible"] for row in result["rates"]] != [True, True, False]):
        raise ValueError("Require the full independently recovered rate evidence, including the rejected largest rate")
    rate_probe.verify_hashes(audit["artifact_sha256"])
    config = rate_probe.read(rate_recovery.OUTPUT / "recovery_config.json")
    rate_probe.verify_hashes(config["input_sha256"])
    rate_probe.verify_hashes(config["source_sha256"])
    return [rate_recovery.OUTPUT / name for name in ("recovery.json", "audit.json", "recovery_config.json")] + [rate_probe.OUTPUT / "config.json"]


install(template, globals(), {
    'OUTPUT = ROOT / "targeted_decoder_v15"': 'OUTPUT = ROOT / "stronger_decoder_v16"',
    'STEPS, BOUNDARIES = 100, (0, 10, 25, 50, 100)': 'STEPS, BOUNDARIES = 64, (0, 10, 25, 50, 64)',
    'DATA_SEED, GRADIENT_SEED = 241100901, 242100901': 'DATA_SEED, GRADIENT_SEED = 261100901, 262100901',
    'CONFIRMATION_SEEDS = (251100901, 252100901)': 'CONFIRMATION_SEEDS = (271100901, 272100901)',
    'config["learning_rate_horizon"] != 100': 'config["learning_rate_schedule"] != {"constant": 2e-5} or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256',
    'shutil.disk_usage(ROOT).free < 29 * 1024**3': 'shutil.disk_usage(ROOT).free < 27 * 1024**3',
    'Reserve all100 task batches, decoder endpoint, validation and18 GiB free space': 'Reserve all64 task batches, decoder endpoint, validation and18 GiB free space',
    '    preflight.verify(preflight.OUTPUT)\n': '    rate_evidence = verify_rate_evidence()\n    preflight.verify(preflight.OUTPUT)\n',
    'len(fresh) != 3084': 'len(fresh) != 2076',
    'Reserve all2800 new data/backward seeds': 'Reserve all1792 new data/backward seeds',
    '    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_targeted_decoder_training.py"}':
        '    paths.extend(rate_evidence)\n    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_stronger_decoder.py"}',
    '"learning_rate_horizon": 100,': '"learning_rate_schedule": {"constant": 2e-5}, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,',
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
                raise ValueError("Never restart the declared stronger-rate trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
