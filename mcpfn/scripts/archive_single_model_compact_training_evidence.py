"""Losslessly archive the finished V22 confirmation bank and raw endpoint after its task archive."""

import inspect

import archive_single_model_idle_evidence as template
import archive_single_model_teacher_trial_capacity as capacity
import archive_single_model_compact_training_tasks as tasks
from single_model_registered_adaptation import install


def completed_sources():
    paths, audit = tasks.terminal_evidence()
    archived = tasks.read(tasks.OUTPUT / "independent_recheck.json")
    if (not archived["complete"] or archived["canonical_paths_and_hashes_checked"] != 1856
            or not archived["registered_scientific_evidence_unchanged"] or not archived["no_baseline_or_checkpoint_modified"]
            or tasks.sha256(tasks.OUTPUT / "config.json") != archived["config_sha256"]):
        raise ValueError("Require the completed unchanged1856-path task archive first")
    tasks.verify_hashes(archived["artifact_sha256"])
    paths.extend(tasks.OUTPUT / name for name in ("config.json", "independent_recheck.json"))
    confirmation = tasks.read(tasks.probe.pipeline.CONFIRMATION / "audit.json")
    return paths, audit, confirmation


def evidence_inventory():
    paths, audit, confirmation = completed_sources()
    bank = tasks.probe.pipeline.CONFIRMATION / "synthetic_bank.pt"
    endpoint = tasks.probe.training.OUTPUT / "raw_final_update.pt"
    records = [original_record(bank, confirmation["artifact_sha256"][str(bank)]),
               original_record(endpoint, audit["artifact_sha256"][str(endpoint)])]
    return [[record] for record in records], [], {str(path): sha256(path) for path in paths}


install(template, globals(), {
    'OUTPUT = ROOT / "idle_evidence_archive_v1"': 'OUTPUT = ROOT / "compact_training_evidence_archive_v1"',
    inspect.getsource(template.require_idle): 'def require_idle():\n    completed_sources()\n',
    inspect.getsource(template.inventory): 'def inventory():\n    return evidence_inventory()\n',
    inspect.getsource(template.idle_compressor): 'def idle_compressor():\n    return capacity.capacity_compressor()\n',
    'or len(actual) != 21:': 'or len(actual) != 2:',
    '"canonical_artifacts_checked": 21': '"canonical_artifacts_checked": 2',
    '"preserved_root_links_checked": 11': '"preserved_root_links_checked": 0',
    '"canonical_artifacts": 21': '"canonical_artifacts": 2',
    'REPO / "tests/test_single_model_idle_evidence_archive.py"]':
        'Path(template.__file__).resolve(), Path(capacity.__file__).resolve(), Path(tasks.__file__).resolve(),\n'
        '               REPO / "tests/test_single_model_compact_training_archives.py"]',
}, {})


if __name__ == "__main__":
    main()
