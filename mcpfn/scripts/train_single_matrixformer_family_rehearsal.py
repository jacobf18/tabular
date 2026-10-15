"""V11: prospectively increase rehearsal of V10's two failed synthetic families."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import random
import shutil
import time

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import train_single_matrixformer_domain_clipping as previous
from train_streamed_large_matrices import wait_for_gpu


ROOT = previous.ROOT
CONTROL = ROOT / "domain_clipping_v10"
EVIDENCE = ROOT / "domain_clipping_v10_family_diagnosis"
mixture = previous.previous.mixture
CONTROL_POLICY = deepcopy(mixture.POLICY)
FOCUS = ("mnar_two_phase_subset", "mnar_skip_logic")
SPECIFICATION = {
    "focus_families": list(FOCUS), "structured_episodes": 200, "all_family_episodes": 100,
    "focused_episodes": 100, "all_pair_cycle": 48, "focused_pair_cycle": 8,
    "coverage_seed_offset": 1000000, "focused_seed_offset": 2000000,
    "coverage": "Alternate100 full-coverage and100 focused structured episodes; full coverage cycles all48 family/component pairs, focus cycles both failed families with all four generators; independently shuffled cycles; unchanged per-step shape/regime/missingness draws",
    "motivation": "Exploratory synthetic development diagnosis, not blinded evidence or a performance guarantee",
    "unchanged": "V8 initialization,400 updates, every legacy and rating episode, scalar losses, teachers, domain clipping, schedule, architecture,152 selection and284 reserved confirmation cases, and all22 strict retention guards",
}
POLICY = {**CONTROL_POLICY, "coverage": SPECIFICATION["coverage"]}
SOURCES = ("train_single_matrixformer_family_rehearsal.py", "confirm_single_matrixformer_family_rehearsal.py",
           "audit_single_matrixformer_family_rehearsal.py")
EXTRA_FIELDS = {"family_rehearsal", "rehearsal_evidence", "rehearsal_control"}
ALLOWED_DIFFERENCES = {"version", "experiment", "source_sha256", "input_sha256", "structured_policy", "training_structured_cases"}


@contextmanager
def policy_runtime(policy=POLICY):
    saved = mixture.POLICY
    mixture.POLICY = policy
    try:
        yield
    finally:
        mixture.POLICY = saved


def training_cases():
    all_pairs = [(family, component) for family in mixture.FAMILIES for component in mixture.COMPONENTS]
    focus_pairs = [(family, component) for family in FOCUS for component in mixture.COMPONENTS]
    orders, result = {}, []
    for index in range(200):
        focused = bool(index % 2)
        pairs = focus_pairs if focused else all_pairs
        cycle, within = divmod(index // 2, len(pairs))
        key = (focused, cycle)
        if key not in orders:
            orders[key] = list(pairs)
            offset = SPECIFICATION["focused_seed_offset" if focused else "coverage_seed_offset"]
            random.Random(mixture.SEEDS["coverage_order_seed"] + offset + cycle).shuffle(orders[key])
        family, component = orders[key][within]
        step = 2 * (index + 1)
        seed = mixture.SEEDS["data_seed"] + 16 * step
        chooser = random.Random(seed)
        shapes = mixture.COMPACT_SHAPES if family in mixture.NEURAL_FAMILIES else mixture.SPECTRAL_SHAPES
        result.append({"step": step, "case": {"family": family, "component": component, "seed": seed,
            "shape": list(chooser.choice(shapes)), "regime": chooser.choice(("legacy", "area_scaled")),
            "requested_missingness": chooser.choice((.2, .6))}})
    return result


def make_config(base):
    config = deepcopy(base)
    config.update(version=11, family_rehearsal=deepcopy(SPECIFICATION), rehearsal_evidence=str(EVIDENCE),
        rehearsal_control=str(CONTROL), structured_policy=deepcopy(POLICY), training_structured_cases=training_cases(),
        experiment="V10 sibling from selected V8: increase full-generator rehearsal of two development-failing families, retain all other families and every rating/legacy episode; no gate relaxation or unselected-tail initialization")
    return config


def validate_pair(config, base):
    if set(config) != set(base) | EXTRA_FIELDS or config.get("version") != 11:
        raise ValueError("Require exactly the prospectively specified V11 fields")
    for key in set(base) - ALLOWED_DIFFERENCES:
        if config[key] != base[key]:
            raise ValueError(f"Original training/selection recipe changed: {key}")
    for key in ("source_sha256", "input_sha256"):
        if any(config[key].get(path) != digest for path, digest in base[key].items()):
            raise ValueError("Keep every original source/input fingerprint")
    expected = make_config(base)
    for key in EXTRA_FIELDS | {"structured_policy", "training_structured_cases"}:
        if config[key] != expected[key]:
            raise ValueError(f"Frozen family-rehearsal specification changed: {key}")


def validate_recipe(config):
    base = previous.previous.read(CONTROL / "config.json")
    validate_pair(config, base)
    original = deepcopy(config)
    for key in EXTRA_FIELDS:
        del original[key]
    for key in ("version", "structured_policy", "training_structured_cases"):
        original[key] = deepcopy(base[key])
    with policy_runtime(CONTROL_POLICY):
        previous.validate_recipe(base)
        previous.validate_recipe(original)


def register(output):
    if output.exists():
        raise FileExistsError("Never overwrite a registered training experiment")
    read = previous.previous.read
    base = read(CONTROL / "config.json")
    previous.validate_recipe(base)
    previous.previous.backend.verify_run(base, CONTROL)
    status, evidence = read(CONTROL / "status.json"), read(EVIDENCE / "summary.json")
    if status["state"] != "training_complete" or status["step"] != 400 or status["best_step"] != 0:
        raise ValueError("Require the completed rejected V10 control")
    if (not evidence["complete"] or evidence["cases"] != 96 or not evidence["original_artifacts_unchanged"]
            or evidence["new_forward_passes"] != 0
            or {family for family, record in evidence["families"].items() if record["delta"] > 0} != set(FOCUS)):
        raise ValueError("Require the complete saved-prediction diagnosis of both failed families")
    receipt = read(EVIDENCE / "registration.json")
    verify_hashes(receipt["input_sha256"])
    verify_hashes({str(REPO / "scripts/diagnose_single_model_family_retention.py"): receipt["source_sha256"]})
    confirmation = ROOT / "domain_clipping_v10_confirmation"
    gate = read(confirmation / "gate.json")
    if (gate["passed"] or gate["selected_step"] != 0 or any((confirmation / name).exists() for name in
            ("synthetic_bank.pt", "reference.json", "initialization.json", "student.json"))):
        raise ValueError("The independent284-case reservation must remain unconsumed")
    config = make_config(base)
    artifacts = [CONTROL / "config.json", CONTROL / "status.json", confirmation / "gate.json"]
    artifacts += [EVIDENCE / name for name in ("registration.json", "summary.json", "cases.csv", "columns.csv", "trajectory.csv", "RESULTS.md")]
    config["input_sha256"].update({str(path): sha256(path) for path in artifacts})
    config["source_sha256"].update({str(REPO / "scripts" / name): sha256(REPO / "scripts" / name) for name in SOURCES})
    config["source_sha256"].update({str(REPO / "scripts/diagnose_single_model_family_retention.py"): receipt["source_sha256"]})
    validate_recipe(config)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    previous.previous.storage_guard(output)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
        "training_started": False, "synthetic_development_decision_only": True, "same_development_selection_bank": True,
        "independent_confirmation_not_previously_evaluated": True, "real_benchmark_protocols_unchanged": True,
        "unselected_tail_used_as_initialization": False})
    for name in SOURCES:
        shutil.copy2(REPO / "scripts" / name, output / name)
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if previous.previous.read(output / "status.json")["state"] != "registered":
            raise ValueError("Never automatically restart a started experiment")
        try:
            write_json(output / "status.json", {"state": "starting_training", "pid": os.getpid(), "updated_unix": time.time()})
            config = previous.previous.read(output / "config.json")
            validate_recipe(config)
            previous.previous.backend.verify_run(config, output)
            wait_for_gpu(output, True, False)
            with policy_runtime(), mixture.runtime(config, output), previous.clipping.runtime(previous.previous.backend):
                write_json(output / "status.json", {"state": "preflighting", "pid": os.getpid(), "updated_unix": time.time()})
                previous.previous.preflight(config, output)
                previous.previous.backend.train(config, output, resume=False)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
