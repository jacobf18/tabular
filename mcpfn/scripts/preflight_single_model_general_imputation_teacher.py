"""Probe a general-imputation teacher update from the audited, unqualified V26 state."""

import hashlib
import inspect

import torch

import preflight_single_model_top_two_blocks as template
import preflight_single_model_top_block as storage_template
import preflight_single_model_structured_teacher as teacher_template
import run_single_model_top_two_blocks_pipeline as preceding
import recover_single_model_two_block_pipeline_identity as recovery
import single_model_general_imputation_teacher as teacher_policy
import single_model_top_two_block_training as top_two
from preflight_single_model_top_two_blocks import composite, paired, replicated, storage, schedule
from single_model_registered_adaptation import replace_once, replace_assignment


PREFIXES = template.PREFIXES
PARENT_SHA256 = "5096341d7ed637848b4eb8807a81c96777849fac22e67d3213a1b9ae3cce242a"
POLICY_TEXT = ("One fresh28-task two-block/decoder optimizer feasibility test after fully audited, unsuccessful "
    "V26 confirmation. Warm-start only the student from audited unqualified V26 step64; it is never a teacher or "
    "retention baseline. Original V8 remains the baseline identity, original general now teaches both legacy and "
    "structured synthetic imputation, and original specialist still teaches ratings. Keep .25 legacy/.25 structured/"
    ".5 ratings mass, shapes/components/regimes, MAR/skip emphasis, observed-only normalization, Huber/NLL/CDF/query "
    "objectives, fresh AdamW moments, constant1e-5, paired0.1 protection,2x norm cap and exact rollback. Train only "
    "62 original final-two-block/decoder tensors,27,766,670 parameters; preserve192 other state tensors bitwise. "
    "Keep the ordinary77,998,496-parameter network and no inference ensemble. Save every individual-draw regression, "
    "all84 raw gradients and identical28-row proposal alias. Independently replay all28 actual teacher grids and "
    "CPU-audit all56 before/candidate grids and the optimizer endpoint. No validation, fresh confirmation, model "
    "nomination, real benchmark or seed reuse; never omit the failed tiny-support case from future complete gates. "
    "The diagnostic checkpoint cannot initialize a future trained model. Keep ordinary diagnostic checkpoint and "
    "canonical evidence on checkpoint storage; only bulky raw endpoint evidence uses declared machine-local "
    "scratch, not provider-persistent storage or external backup. Preserve18/6/50 GiB reserves and1/0/16 GiB "
    "checkpoint/root/scratch write budgets. Preserve the original failed V26 coordinator and successful additive "
    "orchestration recovery. No scientific retry, source change or claim of accuracy from this feasibility test.")


def validate_predecessor(result, audit, replay, confirmation, digest):
    if (not result["complete"] or result["gate"]["passed"] is not False
            or result["gate"]["confirmation_evaluated"] is not True or result["scientific_stages_repeated"] is not False
            or result["campaign"] is not None or result["review"] is not None
            or not audit["complete"] or not audit["complete_training_audit"] or audit["through_step"] != 64
            or audit["accepted_updates"] != 64 or audit["best"]["step"] != 64 or audit["training_episodes"] != 1792
            or audit["prediction_grids"] != 1216 or audit.get("top_two_block_training") is not True
            or audit["independently_regenerated_training_tasks"] != 1792
            or audit["independently_regenerated_rating_contexts"] != 128
            or digest != PARENT_SHA256 or audit["artifact_sha256"].get(str(PARENT)) != digest
            or not replay["complete"] or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 152
            or not confirmation["complete"] or confirmation["cases"] != 284 or confirmation["prediction_grids"] != 1136):
        raise ValueError("Require complete audited but unqualified V26, never a replacement baseline or teacher")


def require_predecessor():
    recovery.verify()
    directories = (preceding.training.OUTPUT, recovery.OUTPUT, preceding.REPLAY, preceding.CONFIRMATION,
                   preceding.AUDIT.with_name(preceding.AUDIT.name + "_task_replay"))
    for directory in directories:
        current = read(directory / "status.json")
        if current["state"] not in {"complete", "training_complete"} or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Require every original V26 producer and recovery owner to finish and stop")
    audit = preceding.prerequisite(preceding.OUTPUT)
    replay, confirmation = [read(directory / "audit.json") for directory in (preceding.REPLAY, preceding.CONFIRMATION)]
    validate_predecessor(read(recovery.OUTPUT / "result.json"), audit, replay, confirmation, sha256(PARENT))
    if ((preceding.training.OUTPUT / "best_validation_checkpoint.pth").exists()
            or preceding.CAMPAIGN.exists() or preceding.REVIEW.exists()):
        raise ValueError("Preserve rejected V26 without a published model or real campaign")
    for report in (replay, confirmation):
        verify_hashes(report["artifact_sha256"])
    paths = [recovery.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "status.json")]
    paths.extend(preceding.OUTPUT / name for name in ("config.json", "registration.json", "status.json"))
    paths.extend(directory / "status.json" for directory in directories)
    paths.extend(directory / name for directory in (preceding.REPLAY, preceding.CONFIRMATION) for name in ("audit.json", "gate.json"))
    paths.extend([preceding.AUDIT / "audit.json", preceding.training.OUTPUT / "config.json",
                  preceding.training.OUTPUT / "registration.json", PARENT])
    return {str(path): sha256(path) for path in paths}


def register_trial(output):
    if output != OUTPUT or output.exists() or output.is_symlink() or SCRATCH.exists() or SCRATCH.is_symlink():
        raise FileExistsError("Use only the new declared general-imputation-teacher preflight directories")
    require_space()
    evidence = require_predecessor()
    base = read(training.alignment.fixed.PRODUCER / "config.json")
    identities = teacher_policy.identities(base, training.alignment.fixed.PARENT)
    if PARENT != preceding.training.OUTPUT / "selected_checkpoint.pth" or sha256(PARENT) != PARENT_SHA256:
        raise ValueError("Warm-start only from the exact audited unqualified V26 selection")
    earlier = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(training.alignment.fixed.seeds_in(read(path)) for path in earlier))
    fresh = training.alignment.fixed.seeds_in(batch())
    if len(fresh) != 56 or fresh.intersection(used):
        raise ValueError("Reserve all56 new data/backward seeds without reusing confirmation")
    paths = [PARENT, training.alignment.fixed.PRODUCER / "config.json", ROOT / "domain_gradient_probe_v1/parameter_inventory.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_general_imputation_teacher.py",
               REPO / "tests/test_single_model_general_imputation_preflight.py", REPO / "tests/test_single_model_top_two_block_training.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "base": base, "parent": str(PARENT), "parent_selection_qualified": False,
              "retention_baseline_for_future_training": str(training.alignment.fixed.PARENT), "mutable_prefixes": list(PREFIXES),
              "teacher_policy": teacher_policy.POLICY, "teacher_identities": identities,
              "batch": batch(), "rate": RATE, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,
              "capture_source_sha256": replicated.CAPTURE_SOURCE_SHA256,
              "protection_source_sha256": paired.PROTECTED_SOURCE_SHA256, "audit_source_sha256": paired.AUDIT_SOURCE_SHA256,
              "scratch": str(SCRATCH), "scratch_is_provider_persistent": False,
              "input_sha256": {**base["input_sha256"], **evidence, **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in earlier}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    SCRATCH.mkdir()
    output.mkdir()
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
               "registered_unix": time.time(), "optimizer_update_performed": False,
               "confirmation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


def expected_role(task):
    specification = task["specification"]
    if specification["domain"] == "recsys":
        if "structured" in specification:
            raise ValueError("Structured teachers apply only to imputation")
        return "specialist"
    if specification["domain"] != "imputation":
        raise ValueError("Unsupported registered training domain")
    return "general"


for name in ("require_space", "frozen_inventory", "save_raw", "audit_raw"):
    source = inspect.getsource(getattr(storage_template, name))
    if name == "require_space":
        source = replace_once(source, "(SCRATCH.parent, 50, 12)", "(SCRATCH.parent, 50, 16)")
    elif name == "frozen_inventory":
        source = replace_once(source, "len(expected) != 34", "len(expected) != 62")
        source = replace_once(source, "and34 last-block/decoder tensors", "and62 final-two-block/decoder tensors")
    exec(compile(source, __file__, "exec"), globals())

for name in ("check_routing", "replay_targets", "audit_teacher_replay"):
    exec(compile(inspect.getsource(getattr(teacher_template, name)), __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "top_two_blocks_preflight_v1"': 'OUTPUT = ROOT / "general_imputation_teacher_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/top_two_blocks_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/general_imputation_teacher_preflight_v1")',
    'PARENT = training.alignment.fixed.PARENT': 'PARENT = preceding.training.OUTPUT / "selected_checkpoint.pth"',
    'DATA_SEED, GRADIENT_SEED = 551100901, 552100901': 'DATA_SEED, GRADIENT_SEED = 581100901, 582100901',
    'teacher_policy.identities(config["base"], PARENT)':
        'teacher_policy.identities(config["base"], training.alignment.fixed.PARENT)\n'
        '            or config["parent_selection_qualified"] is not False\n'
        '            or config["retention_baseline_for_future_training"] != str(training.alignment.fixed.PARENT)\n'
        '            or config["input_sha256"][str(PARENT)] != PARENT_SHA256',
    '# Final-two-block plus decoder optimizer preflight': '# General-imputation teacher optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[arguments.stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
