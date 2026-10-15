"""Losslessly archive terminal optimizer evidence before a larger-batch training trial."""

import inspect

import archive_single_model_idle_evidence as template
from single_model_registered_adaptation import install


PREFLIGHTS = ("decoder_training_preflight_v1", "targeted_decoder_preflight_v1", "v8_decoder_preflight_v1")


def require_completed():
    states = [(name, "complete") for name in (*PREFLIGHTS, "replicated_decoder_preflight_v1", "v8_decoder_v17_pipeline")]
    states.extend((name, "failed") for name in ("family_projection_v12_pair", "family_projection_v12_control",
                                               "family_projection_v12_projected", "family_projection_v12_pipeline_recovery1"))
    states.append(("v8_decoder_v17", "training_complete"))
    for name, expected in states:
        current = read(ROOT / name / "status.json")
        if current["state"] != expected or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Require terminal idle sources and no larger-batch preflight writes")


def completed_inventory():
    records, inputs = [], {}
    for name in PREFLIGHTS:
        path = ROOT / name / "audit.json"
        report = read(path)
        if not report["complete"] or report["training_tasks"] != 14 or report["prediction_grids"] != 28:
            raise ValueError("Every archived preflight endpoint requires its completed independent audit")
        inputs[str(path)] = sha256(path)
        raw = ROOT / name / "raw_update.pt"
        records.append(original_record(raw, report["artifact_sha256"][str(raw)]))
    path = ROOT / "v8_decoder_v17_audit/audit.json"
    report = read(path)
    if not report["complete_training_audit"] or report["through_step"] != 64 or report["prediction_grids"] != 1216:
        raise ValueError("Require the complete V17 audit before archiving its final raw endpoint")
    inputs[str(path)] = sha256(path)
    raw = ROOT / "v8_decoder_v17/raw_final_update.pt"
    records.append(original_record(raw, report["artifact_sha256"][str(raw)]))
    path = ROOT / "family_projection_v12_first_update_recovery1/audit.json"
    report = read(path)
    if not report["complete"] or report["cached_training_episodes_checked"] != 14 or report["raw_gradient_vectors_remeasured"] != 28:
        raise ValueError("Preserve the audited first update of both terminal V12 arms without relabeling the trial complete")
    inputs[str(path)] = sha256(path)
    for arm in ("control", "projected"):
        raw = ROOT / f"family_projection_v12_{arm}/raw_update_1.pt"
        records.append(original_record(raw, report["input_sha256"][str(raw)]))
    prior = ROOT / "idle_evidence_archive_v1/config.json"
    inputs[str(prior)] = sha256(prior)
    preserved = read(prior)["preserved_root_links"]
    if len(records) != 6 or len(preserved) != 11:
        raise ValueError("Keep all six declared raw artifacts and every prior root link")
    return [[record] for record in records], preserved, inputs


install(template, globals(), {
    'OUTPUT = ROOT / "idle_evidence_archive_v1"': 'OUTPUT = ROOT / "completed_optimizer_archive_v1"',
    inspect.getsource(template.require_idle): 'def require_idle():\n    return require_completed()\n',
    inspect.getsource(template.inventory): 'def inventory():\n    return completed_inventory()\n',
    'or len(actual) != 21:': 'or len(actual) != 6:',
    '"canonical_artifacts_checked": 21': '"canonical_artifacts_checked": 6',
    '"canonical_artifacts": 21': '"canonical_artifacts": 6',
    'REPO / "tests/test_single_model_idle_evidence_archive.py"]':
        'Path(template.__file__).resolve(), Path(__file__).with_name("single_model_registered_adaptation.py"),\n'
        '               REPO / "tests/test_single_model_completed_optimizer_archive.py"]',
}, {})


if __name__ == "__main__":
    main()
