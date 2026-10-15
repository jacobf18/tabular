"""Test paired-mean protection on fresh draws from V8 after V18's failed qualification."""

import inspect

import preflight_single_model_targeted_decoder as template
import run_single_model_replicated_decoder_pipeline as preceding
import single_model_paired_protection as paired
import single_model_replicated_decoder_batch as replicated
from single_model_registered_adaptation import install


POLICY_TEXT = ("One fresh28-task decoder-only optimizer feasibility test directly from original V8 after V18 "
               "finishes independent qualification unsuccessfully. Keep two fresh draws per cohort, .25 legacy/"
               ".25 structured/.5 ratings mass, identical V18 first-batch shapes/families/weights, V8 imputation "
               "and original-specialist rating teachers, preprocessing and Huber/NLL/CDF/query objectives. "
               "Keep the original individually clipped28-composite Adam proposal at2e-5, but project against "
               "within-cohort/objective equal-draw means at0.1 margin and2x norm bound. Accept only if all actual "
               "paired point-metric means retain and one improves; preserve all individual losses/regressions. "
               "Restore parameters, moments and preceding rate exactly on rejection. Keep all248 nondecoder "
               "state tensors bitwise frozen. Save every raw gradient, optimizer state and before/candidate "
               "prediction grid for independent CPU auditing. No development or confirmation case, no model "
               "selection, no promotion, no real benchmark and no inference ensemble. This preflight endpoint "
               "cannot initialize a future model. No completed protocol or held-out qualification gate changes.")


def require_space():
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise OSError("Reserve all raw28-task evidence and the unchanged18 GiB free-space floor")


def require_predecessor():
    result, current = read(preceding.OUTPUT / "result.json"), read(preceding.OUTPUT / "status.json")
    if (not result["complete"] or result["gate"]["passed"] or not result["gate"]["confirmation_evaluated"]
            or current["state"] != "complete" or Path(f'/proc/{current["pid"]}').exists()
            or result["campaign"] is not None or result["review"] is not None):
        raise ValueError("Require V18's completed unsuccessful independent qualification")
    preceding.prerequisite(preceding.OUTPUT)
    for directory, key, expected in ((preceding.REPLAY, "endpoint_grids", 56),
                                      (preceding.CONFIRMATION, "prediction_grids", 1136)):
        audit = read(directory / "audit.json")
        if not audit["complete"] or audit[key] != expected:
            raise ValueError("Preserve the complete independent predecessor audits")
        verify_hashes(audit["artifact_sha256"])
    if (preceding.training.OUTPUT / "best_validation_checkpoint.pth").exists():
        raise ValueError("Do not compete with a published preceding checkpoint")


def register_trial(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Use only the new declared paired-protection preflight")
    require_space()
    require_predecessor()
    base = read(training.alignment.fixed.PRODUCER / "config.json")
    if PARENT != training.alignment.fixed.PARENT or sha256(PARENT) != base["input_sha256"][str(PARENT)]:
        raise ValueError("Initialize only from original V8, never the V18 candidate or a preflight endpoint")
    previous = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(training.alignment.fixed.seeds_in(read(path)) for path in previous))
    fresh = training.alignment.fixed.seeds_in(batch())
    if len(fresh) != 56 or fresh.intersection(used):
        raise ValueError("Reserve all56 fresh data/backward seeds without reusing any earlier held-out case")
    evidence = ROOT / "replicated_preflight_projection_v1"
    geometry = read(evidence / "result.json")
    if (not geometry["complete"] or geometry["training_performed"] or geometry["prediction_performed"]
            or geometry["selection_performed"] or len(geometry["pairs"]) != 42):
        raise ValueError("Require the complete saved-gradient diagnostic, not a performance claim")
    for key in ("input_sha256", "source_sha256"):
        verify_hashes(read(evidence / "config.json")[key])
    paths = [PARENT, training.alignment.fixed.PRODUCER / "config.json",
             ROOT / "domain_gradient_probe_v1/parameter_inventory.json", evidence / "config.json", evidence / "result.json",
             preceding.training.OUTPUT / "config.json", preceding.OUTPUT / "config.json",
             preceding.OUTPUT / "registration.json", preceding.OUTPUT / "result.json",
             preceding.AUDIT / "audit.json", preceding.REPLAY / "audit.json",
             preceding.CONFIRMATION / "audit.json", preceding.CONFIRMATION / "gate.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_paired_protection.py",
               REPO / "tests/test_single_model_paired_decoder_preflight.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "base": base, "parent": str(PARENT), "parent_selection_qualified": True,
              "retention_baseline_for_future_training": str(PARENT), "mutable_prefix": "decoder.",
              "batch": batch(), "rate": RATE, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,
              "capture_source_sha256": replicated.CAPTURE_SOURCE_SHA256,
              "protection_source_sha256": paired.PROTECTED_SOURCE_SHA256, "audit_source_sha256": paired.AUDIT_SOURCE_SHA256,
              "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in previous}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
               "registered_unix": time.time(), "optimizer_update_performed": False,
               "confirmation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


install(template, globals(), {
    'OUTPUT = ROOT / "targeted_decoder_preflight_v1"': 'OUTPUT = ROOT / "paired_decoder_preflight_v1"',
    'PARENT = ROOT / "decoder_repair_v14/latest_checkpoint.pth"': 'PARENT = training.alignment.fixed.PARENT',
    'RATE = 2e-6': 'RATE = 2e-5',
    'DATA_SEED, GRADIENT_SEED = 231100901, 232100901': 'DATA_SEED, GRADIENT_SEED = 341100901, 342100901',
    inspect.getsource(template.batch): 'def batch():\n    return replicated.replicate_batch(preceding.training.previous.batches()[0], DATA_SEED, GRADIENT_SEED)\n',
    inspect.getsource(template.register): 'def register(output):\n    return register_trial(output)\n',
    'or config["parent"] != str(PARENT) or config["mutable_prefix"] != "decoder."':
        'or config["parent"] != str(PARENT) or config["mutable_prefix"] != "decoder."\n'
        '            or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256\n'
        '            or config["capture_source_sha256"] != replicated.CAPTURE_SOURCE_SHA256\n'
        '            or config["protection_source_sha256"] != paired.PROTECTED_SOURCE_SHA256\n'
        '            or config["audit_source_sha256"] != paired.AUDIT_SOURCE_SHA256',
    'wait_for_gpu(output, True, False)': 'wait_for_gpu(output, True, False)\n    require_space()',
    'gradients = torch.empty((42, sum(row[2] for row in selected))':
        'gradients = torch.empty((3 * len(config["batch"]["tasks"]), sum(row[2] for row in selected))',
    'entries, tasks, before = training.capture_batch(': 'entries, tasks, before = replicated.capture_batch(',
    'measured = predictions(output, model, view, criterion, tasks, "before")':
        'paired.validate_layout(entries, config["batch"]["tasks"], before)\n'
        '        measured = predictions(output, model, view, criterion, tasks, "before")',
    'training.guarded.protected_update(': 'paired.protected_update(',
    'checks.check_record(record,': 'paired.check_record(record,',
    'endpoint = checks.check_endpoint(': 'endpoint = paired.check_endpoint(',
    'decision = training.guarded.metric_decision(measured["before"], measured["candidate"])':
        'decision = paired.metric_decision(measured["before"], measured["candidate"])',
    'record = result["record"]\n    paired.check_record':
        'paired.validate_layout(entries, config["batch"]["tasks"], before)\n'
        '    record = result["record"]\n    paired.check_record',
    'len(manifest["tasks"]) != 14': 'len(manifest["tasks"]) != 28',
    'all fourteen fresh tasks': 'all twenty-eight fresh tasks',
    'len(report["rows"]) != 14': 'len(report["rows"]) != 28',
    '"training_tasks": 14': '"training_tasks": 28',
    '# Targeted-prior decoder optimizer preflight': '# Paired-mean decoder optimizer preflight',
}, {"POLICY": POLICY_TEXT})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "audit": audit}[arguments.stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
