"""Losslessly archive idle optimizer-bearing checkpoints, never selected or published aliases."""

import inspect

import archive_single_model_idle_evidence as template
from single_model_registered_adaptation import install


RUNS = ("probability_balance_v3", "domain_teachers_v4", "parent_teacher_v5", "column_aligned_v6",
        "area_mixture_v7", "query_budget_v8", "structured_mixture_v9", "domain_clipping_v10",
        "family_rehearsal_v11", "finite_guarded_v13")


def require_terminal():
    for name in RUNS:
        current = read(ROOT / name / "status.json")
        if current["state"] != "training_complete" or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Only immutable completed training tails may be archived")


def checkpoint_inventory():
    require_terminal()
    records, inputs = [], {}
    for name in RUNS:
        directory = ROOT / name
        for filename in ("config.json", "registration.json", "status.json"):
            path = directory / filename
            if not path.is_file():
                raise ValueError("Require the existing training provenance and terminal state")
            inputs[str(path)] = sha256(path)
        path = directory / "latest_checkpoint.pth"
        records.append(original_record(path, sha256(path)))
    return [[record] for record in records], [], inputs


install(template, globals(), {
    'OUTPUT = ROOT / "idle_evidence_archive_v1"': 'OUTPUT = ROOT / "terminal_checkpoint_archive_v1"',
    inspect.getsource(template.require_idle): 'def require_idle():\n    require_terminal()\n',
    inspect.getsource(template.inventory): 'def inventory():\n    return checkpoint_inventory()\n',
    'or len(actual) != 21:': 'or len(actual) != len(RUNS):',
    '"canonical_artifacts_checked": 21': '"canonical_artifacts_checked": len(RUNS)',
    '"preserved_root_links_checked": 11': '"preserved_root_links_checked": 0',
    '"canonical_artifacts": 21': '"canonical_artifacts": len(RUNS)',
    'REPO / "tests/test_single_model_idle_evidence_archive.py"]':
        'Path(template.__file__).resolve(), REPO / "tests/test_single_model_teacher_terminal_archives.py"]',
    'Lossless idle-evidence archival; no checkpoint, baseline, scientific source, registered path/hash or existing archive changes':
        'Lossless archival of10 terminal optimizer-bearing latest checkpoints; all checkpoint bytes, model/optimizer tensors, canonical paths and hashes unchanged; no selected or published alias, baseline bytes, protocol, qualification or existing archive changes',
}, {})


if __name__ == "__main__":
    main()
