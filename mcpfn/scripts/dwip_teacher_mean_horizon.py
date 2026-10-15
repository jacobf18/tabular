"""One128k cosine-horizon proposal retaining the completed teacher-mean recipe."""
from copy import deepcopy
import json

from dwip_distribution_compact_artifacts import distribution_budget, measured_budget
from dwip_teacher_mean_training import control_recipe, read_json
from run_rowcol_controlled_experiments import BENCH, REPO, WORK


CONTROL_STEPS = 64000
HORIZON_STEPS = 128000
QUEUE = "dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918"
PREVIOUS_QUEUE = "dwip_spectral_anchor_mix25_teacher_mean_huber1_64k_20260917"
PREFLIGHT = REPO / "runs/objective_queues/dwip_teacher_mean_horizon_preflight_20260918"
PROPOSAL = REPO / "runs/objective_queues/dwip_teacher_distribution_adam_pilot_20260918/next_teacher_mean_horizon_proposal.json"


def verify_horizon_only_change(control, candidate):
    before, after = vars(control), vars(candidate)
    output_fields = {"run_name", "work_dir", "fixed_validation_cache"}
    if before.keys() != after.keys() or any(before[name] != after[name]
            for name in before.keys() - output_fields - {"phases_json", "decay_steps"}):
        raise ValueError("Only the128k phase endpoint, cosine decay and output paths may change")
    original = json.loads(control.phases_json)
    expected = deepcopy(original)
    if len(expected) != 1 or expected[0]["end_step"] != CONTROL_STEPS:
        raise ValueError("Require the completed single stationary64k phase")
    expected[0]["end_step"] = HORIZON_STEPS
    if json.loads(candidate.phases_json) != expected:
        raise ValueError("The horizon change cannot alter shapes, batch size or phase identity")
    if (control.warmup_steps, control.stable_steps, control.decay_steps) != (600, 0, 63400):
        raise ValueError("Require the unchanged original cosine schedule")
    if candidate.decay_steps != HORIZON_STEPS - 600:
        raise ValueError("Require127400-step decay with the original600-step warmup")


def horizon_recipe():
    control = control_recipe()
    candidate = deepcopy(control)
    phases = json.loads(candidate.phases_json)
    if len(phases) != 1 or phases[0]["end_step"] != CONTROL_STEPS:
        raise ValueError("Require one completed64k phase, not a new curriculum")
    phases[0]["end_step"] = HORIZON_STEPS
    candidate.phases_json = json.dumps(phases)
    candidate.decay_steps = HORIZON_STEPS - candidate.warmup_steps
    verify_horizon_only_change(control, candidate)
    return candidate


def horizon_budget():
    preflight = read_json(PREFLIGHT / "results.json")
    if preflight.get("core_preflight_passed") is not True:
        raise ValueError("Budget requires the completed actual horizon preflight")
    previous = REPO / "runs/objective_queues" / PREVIOUS_QUEUE
    work = WORK / f"tabimpute-{PREVIOUS_QUEUE}-dwip1e5"
    external = list((BENCH / "logs").glob(f"tabimpute_{PREVIOUS_QUEUE}_*.npz"))
    measured = measured_budget(previous, work, preflight["serialization_bytes"], external)
    budget = distribution_budget(measured["reference_model_bytes"], measured["reference_optimizer_bytes"],
                                 2 * measured["observed_other_artifacts_bytes"])
    budget["horizon_steps"] = HORIZON_STEPS
    budget["noncheckpoint_reference_multiplier"] = 2
    return budget
