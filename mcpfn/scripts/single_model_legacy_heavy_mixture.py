"""Reweight fresh paired training tasks without replacing their priors or protections."""

from copy import deepcopy
import math

import single_model_anchor_rehearsal as rehearsal


MASSES = {"legacy": .5, "structured": .25, "recsys": .25}
FACTORS = {"legacy": 2., "structured": 1., "recsys": .5}
POLICY = ("After the existing native-anchor stratification, change only positive task weights from .25 legacy/"
          ".25 structured/.5 ratings to .5 legacy/.25 structured/.25 ratings. Retain all28 tasks, both draws, "
          "all original structured-family relative weights, seeds, shapes, masks, teachers, observed-only "
          "normalization and Huber/NLL/CDF/query objectives. Legacy draws remain independently protected; "
          "structured and ratings pairs retain their equal-weight protections. Raw composite and point "
          "gradients remain unweighted and complete; the28-row Adam proposal alone receives these task "
          "weights. No extra episode, data reuse, held-out reweighting, relaxed guard or inference ensemble.")


def group(task):
    specification = task["specification"]
    if specification["domain"] == "recsys":
        return "recsys"
    if specification["domain"] != "imputation":
        raise ValueError("Only the original two training domains may be reweighted")
    return "structured" if "structured" in specification else "legacy"


def reweight(batch):
    if rehearsal.stratify(batch) != batch:
        raise ValueError("Reweight only the already-stratified original28-task batch")
    result = deepcopy(batch)
    for task in result["tasks"]:
        task["weight"] *= FACTORS[group(task)]
    validate(result)
    return result


def validate(batch):
    restored = deepcopy(batch)
    for task in restored["tasks"]:
        task["weight"] /= FACTORS[group(task)]
    if rehearsal.stratify(restored) != restored:
        raise ValueError("Preserve every original paired identity, anchor stratum and relative structured weight")
    for name, mass in MASSES.items():
        total = math.fsum(task["weight"] for task in batch["tasks"] if group(task) == name)
        if not math.isclose(total, mass, rel_tol=0, abs_tol=1e-15):
            raise ValueError("Require the exact prospectively declared legacy-heavy domain masses")
    return restored
