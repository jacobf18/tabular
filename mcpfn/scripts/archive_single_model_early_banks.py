"""Losslessly retain five older terminal synthetic banks without upgrading their audit claims."""

import inspect

import archive_single_model_idle_evidence as template
import archive_single_model_teacher_trial_capacity as capacity
from single_model_registered_adaptation import install


TRAINING_RUNS = ("probability_balance_v3", "domain_teachers_v4", "parent_teacher_v5")
CONFIRMATION_RUNS = ("probability_balance_v3_confirmation", "parent_teacher_v5_confirmation")
RUNS = TRAINING_RUNS + CONFIRMATION_RUNS


def require_terminal():
    for name in RUNS:
        current = read(ROOT / name / "status.json")
        expected = "training_complete" if name in TRAINING_RUNS else "complete"
        if current["state"] != expected or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Require every early bank's terminal idle source")


def bank_inventory():
    require_terminal()
    records, inputs = [], {}
    for name in RUNS:
        directory = ROOT / name
        names = ("config.json", "registration.json") if name in TRAINING_RUNS else ("manifest.json", "gate.json")
        for filename in (*names, "status.json", "reference.json"):
            path = directory / filename
            inputs[str(path)] = sha256(path)
        records.append(original_record(directory / "synthetic_bank.pt", read(directory / "reference.json")["bank_sha256"]))
    return [[record] for record in records], [], inputs


install(template, globals(), {
    'OUTPUT = ROOT / "idle_evidence_archive_v1"': 'OUTPUT = ROOT / "early_bank_archive_v1"',
    inspect.getsource(template.require_idle): 'def require_idle():\n    require_terminal()\n',
    inspect.getsource(template.inventory): 'def inventory():\n    return bank_inventory()\n',
    inspect.getsource(template.idle_compressor): 'def idle_compressor():\n    return capacity.capacity_compressor()\n',
    'or len(actual) != 21:': 'or len(actual) != len(RUNS):',
    '"canonical_artifacts_checked": 21': '"canonical_artifacts_checked": len(RUNS)',
    '"preserved_root_links_checked": 11': '"preserved_root_links_checked": 0',
    '"canonical_artifacts": 21': '"canonical_artifacts": len(RUNS)',
    'REPO / "tests/test_single_model_idle_evidence_archive.py"]':
        'Path(template.__file__).resolve(), Path(capacity.__file__).resolve(),\n'
        '               REPO / "tests/test_single_model_early_bank_archive.py"]',
    'Lossless idle-evidence archival; no checkpoint, baseline, scientific source, registered path/hash or existing archive changes':
        'Lossless archival of five terminal early synthetic banks; byte preservation only, not a new independent performance audit or an upgrade to their original qualification. No checkpoint, baseline bytes, source, registered path/hash or existing archive changes.',
}, {})


if __name__ == "__main__":
    main()
