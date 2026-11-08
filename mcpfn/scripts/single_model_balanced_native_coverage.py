"""Prospective balanced native priors and orientations under the unchanged task budget."""

import single_model_balanced_structured_components as structured


SPECIFICATION = {
    "native_components": list(range(8)),
    "replicas": 2,
    "component_index": "(step-1 + 4*replicate) % 8",
    "transpose": "bool(((step-1)//8 + replicate) % 2)",
    "joint_balance_period_attempts": 16,
    "joint_strata_per_period": 32,
    "scope": "native component and transpose fields in fresh uncaptured specifications only",
    "structured": "unchanged V53/V54 staggered component schedule",
    "task_count": 28,
    "supervised_domain_masses": {"legacy": .5, "structured": .25, "recsys": .25},
    "balance_scope": "scheduled draws and requested shapes, not admitted or accepted updates or intrinsic latent shapes",
}
POLICY = (
    "Prospective native-coverage hypothesis after V54's audited negative confirmation and unchanged-parent "
    "attribution. The original64-attempt schedule uses128 native draws with explicit transpose=False:72 anchor-MCAR, eight "
    "of each other native component, and requested shape tied to replica. Actual intrinsic anchor matrices "
    "are smaller than requested. Compose the unchanged structured component schedule with a fixed uniform "
    "cycle over all eight native components and both original transpose options. Opposite components and "
    "opposite orientations in the two native replicas preserve both draws and their separate point guards. "
    "Every16 scheduled attempts cover all32 component/requested-shape/orientation strata once; over64 "
    "attempts each stratum appears four times under the original two eight-attempt shape blocks per period. Never infer "
    "balanced actual dimensions, accepted updates, gradients or efficacy from this schedule. Preserve every "
    "structured/rating specification, native requested shapes, regimes, seeds, weights, normalization "
    "implementation, all losses, teachers, V29 initialization, retention targets, optimizer protections, "
    "all22 development guards,32 fresh confirmation comparisons and every frozen real benchmark. "
    "Transposition intentionally changes native input/mask/normalization geometry through the existing "
    "sampler. Reduced anchor exposure may harm retention; larger spectral contexts and their transposes "
    "need new memory and feasibility checks. No family/case-specific exception, query-error selection, "
    "inference ensemble or gate change. This helper authorizes no scientific seed, GPU probe, training, "
    "selection or qualification. Require a separate prospective registration, fresh streams and audits."
)


def balance(batch):
    result = structured.balance(batch)
    for task in result["tasks"]:
        if task["cohort"] != "legacy":
            continue
        specification = task["specification"]
        shape = specification.get("shape")
        if (type(specification.get("transpose")) is not bool or specification["transpose"] is not False
                or not isinstance(shape, list) or len(shape) != 2
                or any(type(dimension) is not int or dimension <= 0 for dimension in shape)):
            raise ValueError("Change only original untransposed native specifications with complete positive shapes")
        specification["component"] = (batch["step"] - 1 + 4 * task["replicate"]) % 8
        specification["transpose"] = bool(((batch["step"] - 1) // 8 + task["replicate"]) % 2)
    return result
