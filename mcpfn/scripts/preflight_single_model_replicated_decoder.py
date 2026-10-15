"""Test one larger fresh-data decoder update from V8 without changing any qualification gate."""

import inspect

import preflight_single_model_targeted_decoder as template
import run_single_model_v8_decoder_pipeline as preceding
import single_model_replicated_decoder_batch as replicated
from single_model_registered_adaptation import install


POLICY_TEXT = ("After V17 completes its full independent qualification and fails its unchanged gate, test one "
               "fresh decoder-only update directly from the original frozen V8 baseline. Use two independent "
               "data/backward draws per cohort of V17's first fourteen-cohort scheduled batch:28 tasks, each "
               "with half its previous composite weight. Preserve every other specification, .25 legacy/.25 "
               "structured/.5 ratings mass, V8 imputation/original-specialist rating teachers, preprocessing "
               "and Huber/NLL/CDF/query/point objectives. Keep all per-draw point guards, not just their means. "
               "Fresh fused-CUDA decoder AdamW uses2e-5,0.1 joint protection,2x displacement bound and exact "
               "finite-point parameter/moment/rate rollback. All248 nondecoder state tensors remain bitwise "
               "unchanged. Save and independently audit all raw gradients, actual optimizer states and full "
               "before/candidate prediction grids. This tests feasibility of a larger fresh-data batch, not "
               "a proven variance reduction or generalization gain. No development, confirmation, selection, "
               "alias or real benchmark is consumed; the endpoint cannot initialize a future model. No "
               "inference ensemble, prior-run restart or guard relaxation.")


def require_space():
    if shutil.disk_usage(ROOT).free < 26 * 1024**3:
        raise OSError("Reserve the doubled raw-gradient evidence plus18 GiB of free space")


def require_predecessor():
    result = read(preceding.OUTPUT / "result.json")
    current = read(preceding.OUTPUT / "status.json")
    if (not result["complete"] or result["gate"]["passed"] is not False
            or not result["gate"]["confirmation_evaluated"] or current["state"] != "complete"
            or result["campaign"] is not None or result["review"] is not None):
        raise ValueError("Require V17's completed unsuccessful independent qualification, never a live or qualified trial")
    preceding.prerequisite(preceding.OUTPUT)
    for directory, key, expected in ((preceding.REPLAY, "endpoint_grids", 28),
                                      (preceding.CONFIRMATION, "prediction_grids", 1136)):
        audit = read(directory / "audit.json")
        if not audit["complete"] or audit[key] != expected:
            raise ValueError("Do not bypass any original endpoint or confirmation audit")
        verify_hashes(audit["artifact_sha256"])
    if (preceding.training.OUTPUT / "best_validation_checkpoint.pth").exists():
        raise ValueError("Do not compete with a published V17 checkpoint")


def register_trial(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Require the new declared replicated-batch preflight")
    require_space()
    require_predecessor()
    base = read(training.alignment.fixed.PRODUCER / "config.json")
    if PARENT != training.alignment.fixed.PARENT or sha256(PARENT) != base["input_sha256"][str(PARENT)]:
        raise ValueError("Start only from the exact original V8 checkpoint, not the V17 candidate")
    previous = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(training.alignment.fixed.seeds_in(read(path)) for path in previous))
    fresh = training.alignment.fixed.seeds_in(batch())
    if len(fresh) != 56 or fresh.intersection(used):
        raise ValueError("Reserve all56 independent fresh data/backward seeds without earlier training or confirmation reuse")
    paths = [PARENT, training.alignment.fixed.PRODUCER / "config.json",
             ROOT / "domain_gradient_probe_v1/parameter_inventory.json",
             preceding.training.OUTPUT / "config.json", preceding.OUTPUT / "config.json",
             preceding.OUTPUT / "registration.json", preceding.OUTPUT / "result.json",
             preceding.AUDIT / "audit.json", preceding.REPLAY / "audit.json",
             preceding.CONFIRMATION / "audit.json", preceding.CONFIRMATION / "gate.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_replicated_decoder_batch.py",
               REPO / "tests/test_single_model_replicated_decoder_preflight.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "base": base, "parent": str(PARENT), "parent_selection_qualified": True,
              "retention_baseline_for_future_training": str(PARENT), "mutable_prefix": "decoder.",
              "batch": batch(), "rate": RATE, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,
              "capture_source_sha256": replicated.CAPTURE_SOURCE_SHA256,
              "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in previous}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "optimizer_update_performed": False, "confirmation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


install(template, globals(), {
    'OUTPUT = ROOT / "targeted_decoder_preflight_v1"': 'OUTPUT = ROOT / "replicated_decoder_preflight_v1"',
    'PARENT = ROOT / "decoder_repair_v14/latest_checkpoint.pth"': 'PARENT = training.alignment.fixed.PARENT',
    'RATE = 2e-6': 'RATE = 2e-5',
    'DATA_SEED, GRADIENT_SEED = 231100901, 232100901': 'DATA_SEED, GRADIENT_SEED = 311100901, 312100901',
    inspect.getsource(template.batch): 'def batch():\n    return replicated.replicate_batch(preceding.training.batches()[0], DATA_SEED, GRADIENT_SEED)\n',
    inspect.getsource(template.register): 'def register(output):\n    return register_trial(output)\n',
    'or config["parent"] != str(PARENT) or config["mutable_prefix"] != "decoder."':
        'or config["parent"] != str(PARENT) or config["mutable_prefix"] != "decoder."\n'
        '            or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256\n'
        '            or config["capture_source_sha256"] != replicated.CAPTURE_SOURCE_SHA256',
    'wait_for_gpu(output, True, False)': 'wait_for_gpu(output, True, False)\n    require_space()',
    'gradients = torch.empty((42, sum(row[2] for row in selected))':
        'gradients = torch.empty((3 * len(config["batch"]["tasks"]), sum(row[2] for row in selected))',
    'entries, tasks, before = training.capture_batch(': 'entries, tasks, before = replicated.capture_batch(',
    'len(manifest["tasks"]) != 14': 'len(manifest["tasks"]) != 28',
    'all fourteen fresh tasks': 'all twenty-eight fresh tasks',
    'len(report["rows"]) != 14': 'len(report["rows"]) != 28',
    '"training_tasks": 14': '"training_tasks": 28',
    '# Targeted-prior decoder optimizer preflight': '# Replicated-batch decoder optimizer preflight',
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
