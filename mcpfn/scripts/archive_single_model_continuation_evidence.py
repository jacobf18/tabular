"""Losslessly archive the completed rejected point preflight and V20 confirmation bank."""

import inspect

import archive_single_model_idle_evidence as template
import archive_single_model_teacher_trial_capacity as capacity
from single_model_registered_adaptation import install


def completed_sources():
    paths = [ROOT / name for name in ("point_inclusive_preflight_v1/status.json",
        "point_inclusive_preflight_v1/audit.json", "point_inclusive_preflight_v1/gate.json",
        "structured_teacher_v20_pipeline/status.json", "structured_teacher_v20_pipeline/result.json",
        "structured_teacher_v20_confirmation/audit.json", "structured_teacher_v20_confirmation/gate.json")]
    terminal, audit, gate, pipeline, result, confirmation, confirmed = map(read, paths)
    if (terminal["state"] != "complete" or pipeline["state"] != "complete"
            or any(Path(f'/proc/{current["pid"]}').exists() for current in (terminal, pipeline))
            or not audit["complete"] or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
            or audit["raw_endpoint"]["gradient_vectors"] != 84 or audit["raw_endpoint"]["accepted"]
            or gate["passed"] or not gate["preflight_only"] or gate["audit_sha256"] != sha256(paths[1])
            or not audit["teacher_replay"]["complete"] or audit["teacher_replay"]["full_grids"] != 28
            or not result["complete"] or result["gate"]["passed"] or not result["gate"]["confirmation_evaluated"]
            or result["campaign"] is not None or result["review"] is not None
            or not confirmation["complete"] or confirmation["prediction_grids"] != 1136 or confirmed["passed"]):
        raise ValueError("Require the complete rejected preflight and independent V20 confirmation")
    return paths, audit, confirmation


def evidence_inventory():
    paths, audit, confirmation = completed_sources()
    endpoint = ROOT / "point_inclusive_preflight_v1/raw_update.pt"
    bank = ROOT / "structured_teacher_v20_confirmation/synthetic_bank.pt"
    records = [original_record(endpoint, audit["artifact_sha256"][str(endpoint)]),
               original_record(bank, confirmation["artifact_sha256"][str(bank)])]
    return [[record] for record in records], [], {str(path): sha256(path) for path in paths}


install(template, globals(), {
    'OUTPUT = ROOT / "idle_evidence_archive_v1"': 'OUTPUT = ROOT / "continuation_evidence_archive_v1"',
    inspect.getsource(template.require_idle): 'def require_idle():\n    completed_sources()\n',
    inspect.getsource(template.inventory): 'def inventory():\n    return evidence_inventory()\n',
    inspect.getsource(template.idle_compressor): 'def idle_compressor():\n    return capacity.capacity_compressor()\n',
    'or len(actual) != 21:': 'or len(actual) != 2:',
    '"canonical_artifacts_checked": 21': '"canonical_artifacts_checked": 2',
    '"preserved_root_links_checked": 11': '"preserved_root_links_checked": 0',
    '"canonical_artifacts": 21': '"canonical_artifacts": 2',
    'REPO / "tests/test_single_model_idle_evidence_archive.py"]':
        'Path(template.__file__).resolve(), Path(capacity.__file__).resolve(),\n'
        '               REPO / "tests/test_single_model_continuation_evidence.py"]',
}, {})


if __name__ == "__main__":
    main()
