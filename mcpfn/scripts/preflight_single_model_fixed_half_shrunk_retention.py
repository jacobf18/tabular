"""Test a fixed half-displacement update on fresh synthetic data after the saved panel."""

from pathlib import Path

import preflight_single_model_shrunk_column_anchor_retention as predecessor
import probe_single_model_shrunk_retention_displacements as panel
import single_model_fixed_half_protection as half_protection
from single_model_registered_adaptation import replace_assignment, replace_once


PANEL_CONFIG_SHA256 = "ece2ecdb8586f12764c81932223de7447004fe0dd05eaa89820adcbc714533f9"
PANEL_AUDIT_SHA256 = "03805f360415360d23659c91e83eb80bc98bae17cedd7b3d8c27ced42934000a"
POLICY_TEXT = ("Exactly one fresh28-task full-network synthetic feasibility probe after the completely audited "
    "negative V38 trial, rejected count-shrunk probe and complete336-grid displacement panel. The largest "
    "locally feasible tested fraction0.5 motivates this separately registered training recipe, not a "
    "diagnostic checkpoint warm start or claimed benchmark benefit. Initialize only from unchanged V29. "
    + half_protection.POLICY + " Preserve the identical count-shrunk weight10 native-anchor retention, "
    "unit-variance pseudocount, original general/specialist teachers, observed-only preprocessing, "
    "Huber/NLL/CDF/query objectives,.5/.25/.25 prior masses,238 trainable tensors/77,998,496 parameters, "
    "all16 frozen buffers, fresh5e-6 AdamW and the.9 allocator cap. Preserve all84 raw gradients and30 "
    "point guards and independently audit both proposal vectors, all teacher/reference grids and56 full "
    "before/candidate grids. The32 GiB scratch allowance now explicitly accounts for ten snapshot vectors, "
    "not eight;384 MiB/16 MiB checkpoint/root allowances and18/6/50 GiB reserves stay unchanged. Scratch is "
    "machine-local, not external backup. Independently regenerate original admission inputs and audit all "
    "count-shrunk penalties with CUDA hidden. No new validation, confirmation, nomination, real benchmark, "
    "inference ensemble, held-out guard change or consumed-seed reuse. No alternate seed or fraction if "
    "this probe fails. The diagnostic endpoint must never initialize a scientific training trial.")


def validate_panel(result, audit, status):
    if (result.get("complete") is not True or result.get("prediction_grids") != 336
            or result.get("checkpoint_selected") is not False or result.get("optimizer_updates") != 0
            or result.get("new_data") is not False or result.get("real_benchmark_success") is not False
            or result.get("config_sha256") != PANEL_CONFIG_SHA256
            or audit.get("complete") is not True or audit.get("prediction_grids") != 336
            or audit.get("training_tasks") != 28 or audit.get("parameter_vectors") != 6
            or audit.get("gradient_vectors_per_fraction") != 84 or audit.get("protected_directions") != 30
            or audit.get("frozen_buffers_per_fraction") != 16 or audit.get("cuda_initialized") is not False
            or audit.get("research_selection_tolerance") != 0 or audit.get("checkpoint_selected") is not False
            or audit.get("real_benchmark_success") is not False or status.get("state") != "complete"
            or status.get("checkpoint_selected") is not False or status.get("locally_feasible_fractions") != [.5, .25, .125]):
        raise ValueError("Require the complete fixed negative-control/positive-local panel without model selection")
    for report in (result, audit):
        if ([row["fraction"] for row in report["fractions"]] != list(panel.FRACTIONS)
                or [row["fraction"] for row in report["fractions"] if row["locally_feasible"]] != [.5, .25, .125]):
            raise ValueError("Retain every successful and failed fraction; do not select a diagnostic endpoint")
        for row in report["fractions"]:
            if set(row["finite"]["repetitions"]) != {"0", "1"}:
                raise ValueError("Both independent same-seed repetitions are mandatory")
            if any(current["defined_slots"] != 30 for current in row["finite"]["repetitions"].values()):
                raise ValueError("Keep all30 original finite point guards")


def require_displacement_panel():
    panel.require_cpu()
    if (panel.sha256(panel.OUTPUT / "config.json") != PANEL_CONFIG_SHA256
            or panel.sha256(panel.OUTPUT / "audit.json") != PANEL_AUDIT_SHA256):
        raise ValueError("Bind the exact completed saved-proposal diagnostic")
    panel.verify()
    result, audit, current = (panel.read(panel.OUTPUT / name) for name in ("result.json", "audit.json", "status.json"))
    validate_panel(result, audit, current)
    if Path(f'/proc/{current["pid"]}').exists():
        raise ValueError("The predecessor audit must be stopped")
    panel.verify_hashes(audit["artifact_sha256"])
    paths = [panel.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "audit.json", "status.json", "RESULTS.md")]
    return {str(path): panel.sha256(path) for path in paths}


SOURCE = predecessor.SOURCE
SOURCE = replace_once(SOURCE, "import single_model_shrunk_column_anchor_retention as retention",
    "import single_model_shrunk_column_anchor_retention as retention\nimport single_model_fixed_half_protection as paired")
SOURCE = replace_once(SOURCE, "STORAGE_ACCOUNTING = dict(template.STORAGE_ACCOUNTING)",
    "STORAGE_ACCOUNTING = dict(template.STORAGE_ACCOUNTING)\n"
    "STORAGE_ACCOUNTING['snapshot_vector_allowance_bytes'] = 10 * 4 * TRAINABLE_PARAMETERS")
SOURCE = replace_once(SOURCE, "def require_predecessor():", "def require_v38_predecessor():")
SOURCE = replace_once(SOURCE, 'for name in ("require_space",',
    'def require_predecessor():\n    return {**require_v38_predecessor(), **require_displacement_panel()}\n\n\n'
    'for name in ("require_space",')
SOURCE = SOURCE.replace("shrunk_column_anchor_retention_preflight_v1", "fixed_half_shrunk_retention_preflight_v1")
SOURCE = replace_once(SOURCE, "931100901, 932100901", "961100901, 962100901")
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", POLICY_TEXT)
REGISTER_ADAPTATIONS = [
    ('config = {"policy": POLICY,', 'config = {"policy": POLICY, "displacement_fraction": paired.FRACTION,'),
    ('REPO / "tests/test_single_model_shrunk_column_anchor_retention_preflight.py",',
        'REPO / "tests/test_single_model_shrunk_column_anchor_retention_preflight.py", '
        'REPO / "tests/test_single_model_fixed_half_protection.py", '
        'REPO / "tests/test_single_model_fixed_half_shrunk_retention_preflight.py",'),
]
registration_patch = "\n".join(f"REGISTER_SOURCE = replace_once(REGISTER_SOURCE, {old!r}, {new!r})" for old, new in REGISTER_ADAPTATIONS)
SOURCE = replace_once(SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    registration_patch + '\nexec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
AUDIT_ADAPTATIONS = [
    ('or config["proposal_policy"] != paired.POLICY',
        'or config["displacement_fraction"] != paired.FRACTION or config["proposal_policy"] != paired.POLICY'),
    ('        expected = training.alignment.projection.projected_parameters(payload["snapshot"]["tentative"], payload["gradients"],\n'
     '                                                                      record["projection"]["raw_gradient_coefficients"])',
        '        expected = payload["snapshot"]["contracted"]\n'
        '        if not torch.equal(candidate, expected):\n'
        '            raise ValueError("Scored candidate must equal the exact saved contracted proposal")'),
]
audit_patch = "\n".join(f"ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, {old!r}, {new!r})" for old, new in AUDIT_ADAPTATIONS)
SOURCE = replace_once(SOURCE, 'ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)',
    audit_patch + '\nADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)')
exec(compile(SOURCE, __file__, "exec"), globals())
