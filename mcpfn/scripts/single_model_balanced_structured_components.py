"""Prospective component-balanced structured priors without extra task slots."""

from copy import deepcopy

from single_model_complete_missingness_prior import COMPONENTS
from single_model_family_balanced_retention import validate_task_mixture
from single_model_structured_mixture import FAMILIES


SPECIFICATION = {
    "components": list(COMPONENTS),
    "families": list(FAMILIES),
    "replicas": 2,
    "component_index": "(step-1 + family_index + family_index//4 + 2*replicate) % 4",
    "scope": "structured component identifiers in fresh, uncaptured training specifications only",
    "native_and_ratings": "unchanged, including native anchor rehearsal",
    "task_count": 28,
    "supervised_domain_masses": {"legacy": .5, "structured": .25, "recsys": .25},
    "balance_scope": "scheduled draws, not necessarily admitted or optimizer-accepted draws",
}
POLICY = (
    "Prospective prior-mixture hypothesis motivated by V52's complete development trajectory and "
    "consumed-task coverage, not a causal conclusion or a qualified model. Replace only structured "
    "component assignments with a fixed staggered cyclic schedule over all four original components "
    "and all twelve original families. Opposite components in the two replicas and advancing the "
    "cycle each step give every family all four components over two consecutive scheduled attempts. "
    "Stagger four-family blocks by one phase; preserve the existing family order and weights. This "
    "gives equal component task counts and equal structured objective mass within each scheduled "
    "batch under the original weights. Native rehearsal, ratings, requested shapes, seed identities, "
    "missingness mechanisms/requested rates, regimes, normalization recipes, losses, teachers, "
    "retention references, optimizer protections, frozen development "
    "and confirmation gates and every real benchmark remain unchanged. Generated intrinsic latent "
    "shapes must still be reported honestly. Rejected/admission-only attempts must remain visible; "
    "schedule balance is not gradient balance, balanced accepted updates, or efficacy. Reduced "
    "structured-anchor exposure may harm retention. No family exception, inference ensemble, "
    "gate relaxation or real-data selection. This helper authorizes no new scientific seed, GPU "
    "probe, training, checkpoint selection or qualification; wait for original V52 audited closure "
    "and require separate registration, fresh streams and capacity admission."
)


def balance(batch):
    tasks = batch["tasks"]
    cohorts = ("legacy", *FAMILIES, "recsys")
    if (type(batch["step"]) is not int or batch["step"] < 1 or len(tasks) != 28
            or len(FAMILIES) != 12 or len(COMPONENTS) != 4):
        raise ValueError("Require the original complete paired batch and component inventory")
    validate_task_mixture(tasks)
    for index, task in enumerate(tasks):
        replicate, cohort = index // 14, cohorts[index % 14]
        if (type(task["index"]) is not int or task["index"] != index
                or type(task["replicate"]) is not int or task["replicate"] != replicate
                or task["cohort"] != cohort or task["name"] != f"{cohort}_draw_{replicate}"
                or any(field in task for field in ("artifact", "sha256", "metadata", "metrics_before"))):
            raise ValueError("Never reorder tasks or relabel an already captured episode")
        specification = task["specification"]
        structured = specification.get("structured")
        if cohort in FAMILIES:
            if (specification["domain"] != "imputation" or not isinstance(structured, dict)
                    or structured["family"] != cohort or structured["component"] not in COMPONENTS):
                raise ValueError("Preserve every original structured family and domain")
        elif structured is not None or "structured" in specification:
            raise ValueError("Never turn a native or rating task into structured imputation")
        elif cohort == "legacy":
            if (specification["domain"] != "imputation" or type(specification["component"]) is not int
                    or specification["component"] not in range(8)
                    or (replicate == 1 and specification["component"] != 6)):
                raise ValueError("Preserve original native coverage and its existing anchor replica")
        elif specification["domain"] != "recsys":
            raise ValueError("Never replace or relabel a rating draw")
    result = deepcopy(batch)
    for task in result["tasks"]:
        if task["cohort"] not in FAMILIES:
            continue
        family_index = FAMILIES.index(task["cohort"])
        component_index = (batch["step"] - 1 + family_index + family_index // 4 + 2 * task["replicate"]) % 4
        task["specification"]["structured"]["component"] = COMPONENTS[component_index]
    return result
