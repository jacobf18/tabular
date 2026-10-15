"""Preserve completed diagnostic tensors and synthetic banks in small lossless archives."""

import inspect

import archive_single_model_idle_evidence as template
from single_model_registered_adaptation import install, replace_once


BANK_RUNS = ("structured_mixture_v9", "domain_clipping_v10", "family_rehearsal_v11")
CONFIRMATION_RUNS = ("v8_decoder_v17", "replicated_decoder_v18", "paired_decoder_v19")
ARTIFACT_COUNT = 39


def terminal_sources():
    sources = [(name, "training_complete") for name in BANK_RUNS]
    sources.extend((f"{name}_pipeline", "complete") for name in CONFIRMATION_RUNS)
    sources.extend((("family_projection_preflight_v1", "failed"),
                    ("family_projection_preflight_v1_audit_recovery1", "complete"),
                    ("point_proposal_grid_v1", "complete")))
    for name, expected in sources:
        current = read(ROOT / name / "status.json")
        if current["state"] != expected or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Require terminal idle sources without relabeling failed original executions")
    return [ROOT / name / "status.json" for name, expected in sources]


def capacity_inventory():
    inputs = {str(path): sha256(path) for path in terminal_sources()}
    records = []
    path = ROOT / "family_projection_preflight_v1_audit_recovery1/audit.json"
    report = read(path)
    if (not report["complete"] or not report["recovery_complete"] or report["tasks"] != 14
            or not report["original_artifacts_unchanged"] or report["training_restarted"]
            or report["selection_tolerance_changed"]):
        raise ValueError("Require the completed independent recovery, preserving the original audit failure")
    inputs[str(path)] = sha256(path)
    source = ROOT / "family_projection_preflight_v1"
    files = [source / "tasks" / f"task_{index:02d}.pt" for index in range(14)] + [source / "optimizer_step.pt"]
    records.extend(original_record(path, report["artifact_sha256"][str(path)]) for path in files)
    path = ROOT / "point_proposal_grid_v1/audit.json"
    report = read(path)
    if (not report["complete"] or not report["every_available_candidate_arithmetic_and_direction_recomputed"]
            or not report["every_saved_prediction_cpu_rescored"] or report["checkpoint_promoted"] or report["goal_achieved"]):
        raise ValueError("Require the completed non-promoting proposal diagnostic")
    inputs[str(path)] = sha256(path)
    source = ROOT / "point_proposal_grid_v1"
    files = sorted(path for path in source.glob("*/*.pt"))
    if len(files) != 18:
        raise ValueError("Preserve all18 proposal vectors")
    records.extend(original_record(path, report["input_sha256"][str(path)]) for path in files)
    for name in BANK_RUNS:
        path = ROOT / name / "reference.json"
        reference = read(path)
        inputs[str(path)] = sha256(path)
        records.append(original_record(ROOT / name / "synthetic_bank.pt", reference["bank_sha256"]))
    for name in CONFIRMATION_RUNS:
        path = ROOT / f"{name}_confirmation" / "audit.json"
        report = read(path)
        if not report["complete"] or report["prediction_grids"] != 1136:
            raise ValueError("Require complete independent confirmation before lossless bank archival")
        inputs[str(path)] = sha256(path)
        bank = path.parent / "synthetic_bank.pt"
        records.append(original_record(bank, report["artifact_sha256"][str(bank)]))
    if len(records) != ARTIFACT_COUNT:
        raise ValueError("Preserve the exact39-artifact inventory")
    return [[record] for record in records], [], inputs


def capacity_compressor():
    source = replace_once(inspect.getsource(archive.compress_group),
                          "if not isinstance(payload, dict):", "if not isinstance(payload, (dict, list)):")
    namespace = {**archive.__dict__, "MINIMUM_FREE": MINIMUM_FREE}
    exec(compile(source, __file__, "exec"), namespace)
    return namespace["compress_group"]


install(template, globals(), {
    'OUTPUT = ROOT / "idle_evidence_archive_v1"': 'OUTPUT = ROOT / "structured_teacher_capacity_archive_v1"',
    inspect.getsource(template.require_idle): 'def require_idle():\n    terminal_sources()\n',
    inspect.getsource(template.inventory): 'def inventory():\n    return capacity_inventory()\n',
    inspect.getsource(template.idle_compressor): 'def idle_compressor():\n    return capacity_compressor()\n',
    'or len(actual) != 21:': 'or len(actual) != ARTIFACT_COUNT:',
    '"canonical_artifacts_checked": 21': '"canonical_artifacts_checked": ARTIFACT_COUNT',
    '"preserved_root_links_checked": 11': '"preserved_root_links_checked": 0',
    '"canonical_artifacts": 21': '"canonical_artifacts": ARTIFACT_COUNT',
    'REPO / "tests/test_single_model_idle_evidence_archive.py"]':
        'Path(template.__file__).resolve(), Path(__file__).with_name("single_model_registered_adaptation.py"),\n'
        '               REPO / "tests/test_single_model_teacher_trial_capacity.py"]',
}, {})


if __name__ == "__main__":
    main()
