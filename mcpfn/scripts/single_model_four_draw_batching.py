"""Prospective four-draw batches; metadata only, without sampling or training."""

from collections import Counter
from copy import deepcopy
import math

from single_model_complete_missingness_prior import COMPONENTS
from single_model_family_balanced_retention import validate_task_mixture
from single_model_structured_mixture import FAMILIES


SOURCE_ATTEMPTS = 256
ATTEMPTS = 128
COHORTS = ("legacy", *FAMILIES, "recsys")
TASK_FIELDS = {"name", "weight", "specification", "index", "gradient_seed", "cohort", "replicate"}
SPECIFICATION = {
    "source_attempts": SOURCE_ATTEMPTS,
    "attempts": ATTEMPTS,
    "source_tasks_per_attempt": 28,
    "tasks_per_attempt": 56,
    "replicas_per_cohort": 4,
    "scheduled_tasks": 7168,
    "source_pair": "original ordered attempts2*step-1 and2*step",
    "task_weight": "exactly one half of each original task weight",
    "domain_masses": {"legacy": .5, "structured": .25, "recsys": .25},
    "potential_gradient_slots_per_attempt": 168,
    "grouped_metric_slots_per_attempt": 34,
    "seed_policy": "caller must separately admit fresh data/backward streams; this helper reserves no seeds",
}
POLICY = (
    "Prospective batch-granularity hypothesis after V56's complete negative closure. Pair adjacent "
    "uncaptured specifications from the same256-attempt balanced schedule, retaining every task, "
    "seed, requested shape, orientation, component and missingness specification exactly once. "
    "Halve each task weight so every new56-task batch keeps native/structured/rating masses.5/.25/.25. "
    "Four draws per structured family contain all four original components in every update, rather "
    "than distributing them across two successive updates. Keep four individual native draws and "
    "all rating contexts. This changes optimization granularity, not the scheduled7168-task budget. "
    "More within-update coverage may reduce mixture noise but does not eliminate gradient conflicts, "
    "guarantee more accepted updates or prove generalization. Fewer optimizer attempts may underfit; "
    "more protected directions may worsen feasibility and require more memory. The original admission "
    "rule must apply to the whole paired batch: inspect all48 structured draws, defer all eight "
    "native/rating draws and consume the attempt if any structured draw is unusable. Never salvage "
    "one half, repair a mask or replace a draw. This can reduce actual admitted coverage. Preserve original "
    "losses, teachers, V29 initialization, retention references, quarter displacement and per-draw "
    "gradient protections; retain every native finite metric and all structured/rating grouped "
    "metrics, including explicit undefined slots. All22 development guards,32 fresh-confirmation "
    "comparisons and the full frozen real benchmark scope remain required. No validation-selected "
    "pairing, favorable-draw selection, hidden-truth routing or inference ensemble. This schedule "
    "helper implements no trainer, gradient capture, admission, audit, seed registration or GPU "
    "probe. Require independent fresh-seed admission, complete integration, capacity planning and "
    "a separately registered feasibility test before any scientific execution. Never reuse consumed "
    "V56 draws for a new experiment merely because their specifications pass metadata checks."
)


def _validate_source_batch(batch, expected_step):
    if (set(batch) != {"step", "tasks"} or type(batch["step"]) is not int
            or batch["step"] != expected_step or len(batch["tasks"]) != 28):
        raise ValueError("Require every original ordered uncaptured28-task batch")
    for index, task in enumerate(batch["tasks"]):
        replicate, cohort = index // 14, COHORTS[index % 14]
        if (set(task) != TASK_FIELDS or type(task["index"]) is not int or task["index"] != index
                or type(task["replicate"]) is not int or task["replicate"] != replicate
                or task["cohort"] != cohort or task["name"] != f"{cohort}_draw_{replicate}"
                or isinstance(task["weight"], bool) or not math.isfinite(task["weight"])
                or task["weight"] <= 0):
            raise ValueError("Never omit, reorder or relabel captured task evidence")
        specification = task["specification"]
        expected_domain = "recsys" if cohort == "recsys" else "imputation"
        if specification.get("domain") != expected_domain:
            raise ValueError("Preserve every original task domain")
        if cohort in FAMILIES:
            structured = specification.get("structured")
            family_index = FAMILIES.index(cohort)
            component_index = (expected_step - 1 + family_index + family_index // 4 + 2 * replicate) % 4
            if (not isinstance(structured, dict) or structured.get("family") != cohort
                    or structured.get("component") != COMPONENTS[component_index]):
                raise ValueError("Require the unchanged staggered four-component family schedule")
            geometry = structured
        else:
            if "structured" in specification:
                raise ValueError("Never convert native or rating tasks to structured imputation")
            geometry = specification
        if cohort == "legacy":
            if (type(specification.get("component")) is not int
                    or specification["component"] != (expected_step - 1 + 4 * replicate) % 8
                    or type(specification.get("transpose")) is not bool
                    or specification["transpose"] != bool(((expected_step - 1) // 8 + replicate) % 2)):
                raise ValueError("Preserve the original balanced native components and orientations")
        shape = geometry.get("shape")
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(size) is not int or size <= 0 for size in shape)):
            raise ValueError("Preserve complete positive requested matrix shapes")
        for seed in [geometry.get("seed"), task["gradient_seed"]]:
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("Require explicit data/backward seed identities, without reserving them")
    validate_task_mixture(batch["tasks"])


def _validate_sources(batches):
    if len(batches) != SOURCE_ATTEMPTS or len(COHORTS) != 14 or len(COMPONENTS) != 4:
        raise ValueError("Pair the complete original256-attempt schedule, not a selected subset")
    seeds, strata = set(), Counter()
    for step, batch in enumerate(batches, start=1):
        _validate_source_batch(batch, step)
        for task in batch["tasks"]:
            specification = task["specification"]
            geometry = specification.get("structured", specification)
            for seed in [geometry["seed"], task["gradient_seed"]]:
                if seed in seeds:
                    raise ValueError("Keep all data/backward seeds distinct within the supplied schedule")
                seeds.add(seed)
            if task["cohort"] == "legacy":
                strata[specification["component"], tuple(specification["shape"]), specification["transpose"]] += 1
    if len(strata) != 32 or set(strata.values()) != {16}:
        raise ValueError("Retain all32 native component/requested-shape/orientation strata equally")


def validate_schedule(batches):
    if len(batches) != ATTEMPTS:
        raise ValueError("Require all128 prospective four-draw batches")
    sources = []
    for step, batch in enumerate(batches, start=1):
        if (set(batch) != {"step", "source_steps", "tasks"} or type(batch["step"]) is not int
                or batch["step"] != step or batch["source_steps"] != [2 * step - 1, 2 * step]
                or any(type(source_step) is not int for source_step in batch["source_steps"])
                or len(batch["tasks"]) != 56):
            raise ValueError("Keep every declared adjacent pair and all56 task slots")
        for member in range(2):
            tasks = deepcopy(batch["tasks"][28 * member:28 * (member + 1)])
            for index, task in enumerate(tasks):
                replicate, cohort = 2 * member + index // 14, COHORTS[index % 14]
                if (type(task["index"]) is not int or task["index"] != 28 * member + index
                        or type(task["replicate"]) is not int or task["replicate"] != replicate
                        or task["cohort"] != cohort or task["name"] != f"{cohort}_draw_{replicate}"):
                    raise ValueError("Require unique ordered four-draw identities")
                task.update(index=index, replicate=index // 14, name=f"{cohort}_draw_{index // 14}",
                    weight=2 * task["weight"])
            sources.append({"step": 2 * step - 1 + member, "tasks": tasks})
        for family in FAMILIES:
            counts = Counter(task["specification"]["structured"]["component"]
                for task in batch["tasks"] if task["cohort"] == family)
            if counts != dict.fromkeys(COMPONENTS, 1):
                raise ValueError("Every structured family must contain all four components in every batch")
    _validate_sources(sources)
    return {"attempts": ATTEMPTS, "tasks": 7168, "native_tasks": 512,
        "structured_tasks": 6144, "rating_tasks": 512, "seed_identities": 14336}


def pair_batches(batches):
    _validate_sources(batches)
    result = []
    for offset in range(0, SOURCE_ATTEMPTS, 2):
        tasks = []
        for member, source in enumerate(batches[offset:offset + 2]):
            for original in source["tasks"]:
                task = deepcopy(original)
                replicate = 2 * member + task["replicate"]
                task.update(index=len(tasks), replicate=replicate,
                    name=f"{task['cohort']}_draw_{replicate}", weight=task["weight"] / 2)
                tasks.append(task)
        result.append({"step": offset // 2 + 1, "source_steps": [offset + 1, offset + 2], "tasks": tasks})
    validate_schedule(result)
    return result
