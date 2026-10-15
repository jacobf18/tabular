"""Preserve four terminal confirmation and three development banks in lossless archives."""

import inspect

import archive_single_model_idle_evidence as template
import archive_single_model_teacher_trial_capacity as capacity
from single_model_registered_adaptation import install


RUNS = ("area_mixture_v7_confirmation", "query_budget_v8_confirmation_defined1",
        "query_budget_v8_tail_average_confirmation_retry1", "column_aligned_v6_confirmation")
DEVELOPMENT_RUNS = ("area_mixture_v7", "column_aligned_v6", "query_budget_v8")
ALL_RUNS = RUNS + DEVELOPMENT_RUNS


def require_terminal():
    for name in ALL_RUNS:
        current = read(ROOT / name / "status.json")
        expected = "complete" if name in RUNS else "training_complete"
        if current["state"] != expected or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Only completed idle confirmation banks may be archived")


def bank_inventory():
    require_terminal()
    records, inputs = [], {}
    for name in ALL_RUNS:
        directory = ROOT / name
        filenames = ("status.json", "reference.json", "manifest.json", "confirmation_audit.json") if name in RUNS else (
            "config.json", "registration.json", "status.json", "reference.json")
        for filename in filenames:
            path = directory / filename
            inputs[str(path)] = sha256(path)
        if name in RUNS:
            report = read(directory / "confirmation_audit.json")
            if not report["complete"] or report["cases"] != 92:
                raise ValueError("Retain independently audited confirmation only")
        records.append(original_record(directory / "synthetic_bank.pt", read(directory / "reference.json")["bank_sha256"]))
    return [[record] for record in records], [], inputs


install(template, globals(), {
    'OUTPUT = ROOT / "idle_evidence_archive_v1"': 'OUTPUT = ROOT / "legacy_bank_capacity_archive_v1"',
    inspect.getsource(template.require_idle): 'def require_idle():\n    require_terminal()\n',
    inspect.getsource(template.inventory): 'def inventory():\n    return bank_inventory()\n',
    inspect.getsource(template.idle_compressor): 'def idle_compressor():\n    return capacity.capacity_compressor()\n',
    'or len(actual) != 21:': 'or len(actual) != len(ALL_RUNS):',
    '"canonical_artifacts_checked": 21': '"canonical_artifacts_checked": len(ALL_RUNS)',
    '"preserved_root_links_checked": 11': '"preserved_root_links_checked": 0',
    '"canonical_artifacts": 21': '"canonical_artifacts": len(ALL_RUNS)',
    'REPO / "tests/test_single_model_idle_evidence_archive.py"]':
        'Path(template.__file__).resolve(), Path(capacity.__file__).resolve(),\n'
        '               REPO / "tests/test_single_model_legacy_confirmation_archive.py"]',
}, {})


if __name__ == "__main__":
    main()
