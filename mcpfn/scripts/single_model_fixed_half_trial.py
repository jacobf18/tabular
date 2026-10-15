"""Bind fixed-half training, independent optimizer audit and endpoint reconstruction."""

import torch

import single_model_fixed_half_protection as protection
from single_model_registered_adaptation import replace_once


POLICY = ("Use the independently tested fixed-half projected-displacement rule throughout all64 fresh "
    "mixed-prior attempts, with the original count-shrunk retention and unchanged qualification. "
    + protection.POLICY + " Preserve both full proposal vectors in terminal optimizer evidence and "
    "independently reconstruct the actual contracted candidate on GPU before replaying its predictions. "
    "CPU replay audit must bind that candidate exactly to the raw contracted vector, even if rejected. "
    "Initialize only from V29, never the feasibility endpoint. Keep all22 development guards,32 fresh "
    "confirmation comparisons and every frozen real benchmark requirement unchanged. A synthetic pass "
    "does not establish benchmark parity; retain all individual regressions and no fallback selection.")


def adapt_training_recipe(source):
    source = replace_once(source, 'RATE, PREFIXES = template.RATE, full_network.PREFIXES',
        'import single_model_fixed_half_protection as paired\n\n\nRATE, PREFIXES = template.RATE, full_network.PREFIXES')
    source = replace_once(source, '"snapshot_vector_allowance": 8 * 4 * TRAINABLE_PARAMETERS',
        '"snapshot_vector_allowance": 10 * 4 * TRAINABLE_PARAMETERS')
    source = replace_once(source, 'exec(compile(BASE_PREFLIGHT_SOURCE, __file__, "exec"), globals())',
        'exec(compile(BASE_PREFLIGHT_SOURCE, __file__, "exec"), globals())\n'
        'validate_point_preflight = validate_preflight\n\n\n'
        'def validate_preflight(result, audit, gate, current, digest):\n'
        '    validate_point_preflight(result, audit, gate, current, digest)\n'
        '    fixed_trial.check_preflight_evidence(result, audit)\n')
    source = replace_once(source, 'ADAPTED_SOURCE = retention_training.adapt_training(ADAPTED_SOURCE)',
        'ADAPTED_SOURCE = retention_training.adapt_training(ADAPTED_SOURCE)\n'
        'ADAPTED_SOURCE = fixed_trial.adapt_training(ADAPTED_SOURCE)')
    return replace_once(source, 'REGISTER_SOURCE = retention_training.adapt_registration(REGISTER_SOURCE)',
        'REGISTER_SOURCE = retention_training.adapt_registration(REGISTER_SOURCE)\n'
        'REGISTER_SOURCE = fixed_trial.adapt_registration(REGISTER_SOURCE)')


def adapt_training(source):
    return replace_once(source, 'or config["guarded_policy"] != paired.POLICY',
        'or config["displacement_fraction"] != paired.FRACTION or config["guarded_policy"] != paired.POLICY')


def check_preflight_evidence(result, audit):
    contraction = audit["raw_endpoint"].get("fixed_half_step", {})
    if (result["record"].get("displacement_fraction") != protection.FRACTION
            or contraction.get("fraction") != protection.FRACTION or contraction.get("evaluated") is not True
            or contraction.get("parameters") != 77998496 or contraction.get("actual_saved_displacements_checked") is not True):
        raise ValueError("Require the accepted fresh full-network fixed-half optimizer audit")


def adapt_registration(source):
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "displacement_fraction": paired.FRACTION,')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.add(REPO / "tests/test_single_model_fixed_half_trial.py")\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_audit_recipe(source):
    return replace_once(source, 'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()',
        'ADAPTED_SOURCE = fixed_trial.adapt_audit(ADAPTED_SOURCE)\n'
        'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()')


def adapt_audit(source):
    source = replace_once(source, 'import single_model_point_only_protection as checks',
        'import single_model_fixed_half_protection as checks')
    return replace_once(source, '"complete_training_audit": complete_trial,',
        '"complete_training_audit": complete_trial, "fixed_half_training": True, "displacement_fraction": training.paired.FRACTION,')


def adapt_pipeline_recipe(source):
    source = replace_once(source, 'ADAPTED_SOURCE = template.ADAPTED_SOURCE',
        'POLICY_TEXT += " " + fixed_trial.POLICY\nADAPTED_SOURCE = template.ADAPTED_SOURCE')
    return replace_once(source, 'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()',
        'ADAPTED_SOURCE = fixed_trial.adapt_pipeline(ADAPTED_SOURCE)\n'
        'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()')


def adapt_pipeline(source):
    source = replace_once(source, '    retention_training.require_report(training, audit)',
        '    retention_training.require_report(training, audit)\n    fixed_trial.require_report(training, audit)')
    source = replace_once(source,
        '        gradients = payload["gradients"].to("cuda")\n'
        '        candidate = training.original.alignment.projection.projected_parameters(snapshot["tentative"].to("cuda"), gradients,\n'
        '                                        record["projection"]["raw_gradient_coefficients"])',
        '        candidate, gradients = fixed_trial.reconstruct_candidate(training, payload, "cuda")')
    return replace_once(source,
        '        products, norm = direction_products(payload["gradients"], payload["snapshot"]["before"].double() - candidate["parameters"].double())',
        '        fixed_trial.check_replayed_candidate(payload, candidate["parameters"])\n'
        '        products, norm = direction_products(payload["gradients"], payload["snapshot"]["before"].double() - candidate["parameters"].double())')


def require_report(training, audit):
    if (training.paired is not protection or audit.get("fixed_half_training") is not True
            or audit.get("displacement_fraction") != protection.FRACTION):
        raise ValueError("Require independent fixed-half optimizer evidence, not a legacy projection audit")
    endpoint = audit["raw_endpoint"]
    if endpoint.get("optimizer_attempt_step") is None:
        if (endpoint.get("unchanged_initializer_checked") is not True or endpoint.get("accepted_updates") != 0
                or endpoint.get("gradient_vectors") != 0 or audit["admission_attempts"]["optimizer_attempts"] != 0):
            raise ValueError("A missing optimizer endpoint requires the original complete no-op proof")
        return
    contraction = endpoint.get("fixed_half_step", {})
    if contraction.get("fraction") != protection.FRACTION or type(contraction.get("evaluated")) is not bool:
        raise ValueError("The actual terminal optimizer attempt must bind the fixed contraction")
    if contraction["evaluated"] and (contraction.get("parameters") != training.TRAINABLE_PARAMETERS
            or contraction.get("actual_saved_displacements_checked") is not True):
        raise ValueError("Every actual rounded full-network proposal vector must be independently checked")


@torch.no_grad()
def reconstruct_candidate(training, payload, device):
    if training.paired is not protection or payload["record"].get("displacement_fraction") != protection.FRACTION:
        raise ValueError("Reconstruct only the registered fixed-half proposal")
    snapshot, record = payload["snapshot"], payload["record"]
    gradients = payload["gradients"].to(device)
    uncontracted = training.original.alignment.projection.projected_parameters(
        snapshot["tentative"].to(device), gradients, record["projection"]["raw_gradient_coefficients"])
    if not torch.equal(uncontracted.cpu(), snapshot["uncontracted"]):
        raise ValueError("Identical projection inputs must reconstruct the saved uncontracted FP32 vector")
    candidate = protection.half_step(snapshot["before"].to(device), uncontracted)
    check_replayed_candidate(payload, candidate.cpu())
    return candidate, gradients


def check_replayed_candidate(payload, candidate):
    if (payload["record"].get("displacement_fraction") != protection.FRACTION
            or candidate.dtype != torch.float32 or not torch.equal(candidate, payload["snapshot"]["contracted"])):
        raise ValueError("Replayed candidate must equal the actual saved contracted vector, accepted or rejected")
