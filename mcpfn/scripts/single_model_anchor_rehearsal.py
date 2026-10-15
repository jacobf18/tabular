"""Stratify paired synthetic training draws without altering any held-out bank."""

from collections import defaultdict
from copy import deepcopy
import math

from single_model_complete_missingness_prior import COMPONENTS
from single_model_paired_protection import identity


POLICY = ("Keep draw0 of every original cohort and both ratings draws unchanged. Replace only the generator "
    "component of each imputation draw1 with the existing native anchor-latent prior: legacy component6 or "
    "structured anchor_latent_mcar. Preserve all28 task slots, fresh seeds, equal within-cohort weights, "
    ".25 legacy/.25 structured/.5 ratings mass, mask families, requested shapes, regimes, normalization and "
    "objectives. Native latent shapes still override requested shapes and must be reported honestly. The "
    "paired training guard now averages original-coverage and anchor strata, not two exchangeable component "
    "draws; preserve each individual regression. No added task, input bank reuse, held-out gate change, "
    "inference ensemble or claim of a teacher-only or equal-data causal comparison.")


def stratify(batch):
    tasks = batch["tasks"]
    if (type(batch["step"]) is not int or batch["step"] < 1 or len(tasks) != 28
            or [task["index"] for task in tasks] != list(range(28))):
        raise ValueError("Require the complete ordered28-task paired training batch")
    cohorts, masses = defaultdict(list), defaultdict(float)
    for task in tasks:
        cohort, draw = identity(task["name"])
        specification = task["specification"]
        if (task["cohort"] != cohort or task["replicate"] != draw or task["index"] // 14 != draw
                or not math.isfinite(task["weight"]) or task["weight"] <= 0):
            raise ValueError("Preserve positive weights and explicit original paired identities")
        cohorts[cohort].append(task)
        domain = specification["domain"]
        structured = specification.get("structured")
        if domain == "recsys":
            if cohort != "recsys" or structured is not None:
                raise ValueError("Never replace or relabel a ratings draw")
            group = "recsys"
        elif domain == "imputation":
            group = "legacy" if structured is None else "structured"
            if structured is None:
                if cohort != "legacy" or type(specification["component"]) is not int or specification["component"] not in range(8):
                    raise ValueError("Require the original legacy component inventory")
            elif structured["component"] not in COMPONENTS or structured["family"] != cohort:
                raise ValueError("Preserve original structured components and cohort mechanisms")
        else:
            raise ValueError("Require an original synthetic training domain")
        masses[group] += task["weight"]
    if (len(cohorts) != 14 or set(masses) != {"legacy", "structured", "recsys"}
            or any(not math.isclose(masses[group], mass, rel_tol=0, abs_tol=1e-15)
                   for group, mass in (("legacy", .25), ("structured", .25), ("recsys", .5)))
            or any(len(members) != 2 or members[0]["weight"] != members[1]["weight"]
                   or members[0]["index"] + 14 != members[1]["index"] for members in cohorts.values())):
        raise ValueError("Preserve all14 equally weighted pairs and the original domain masses")
    result = deepcopy(batch)
    for task in result["tasks"]:
        specification = task["specification"]
        if task["replicate"] == 1 and specification["domain"] == "imputation":
            if "structured" in specification:
                specification["structured"]["component"] = "anchor_latent_mcar"
            else:
                specification["component"] = 6
    return result
