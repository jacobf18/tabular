"""Preserve the fully audited shared-storage V21 endpoint without altering its original bytes."""

import inspect

import archive_single_model_decoder_endpoint as template
import archive_single_model_idle_evidence as idle
import archive_single_model_point_inclusive_tasks as completed
from single_model_registered_adaptation import install


def verified_record():
    paths, audit = completed.terminal_evidence()
    return idle.original_record(SOURCE, audit["artifact_sha256"][str(SOURCE)])


install(template, globals(), {
    'OUTPUT = archive.ROOT / "decoder_endpoint_archive_v1"': 'OUTPUT = archive.ROOT / "point_inclusive_endpoint_archive_v1"',
    'SOURCE = archive.ROOT / "decoder_repair_v14/raw_final_update.pt"': 'SOURCE = archive.ROOT / "point_inclusive_v21/raw_final_update.pt"',
    inspect.getsource(template.source_record): 'def source_record():\n    return verified_record()\n',
    'inputs = [SOURCE.parent / "status.json", archive.ROOT / "decoder_repair_v14_audit/audit.json"]':
        'inputs, _audit = completed.terminal_evidence()',
    'sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(), REPO / "tests/test_single_model_endpoint_archive.py"]':
        'sources = [Path(__file__).resolve(), Path(template.__file__).resolve(), Path(idle.__file__).resolve(),\n'
        '               Path(completed.__file__).resolve(), Path(archive.__file__).resolve(),\n'
        '               REPO / "tests/test_single_model_point_inclusive_archives.py"]',
    '"minimum_free_bytes_during_staging": archive.MINIMUM_FREE': '"minimum_free_bytes_during_staging": idle.MINIMUM_FREE',
    'result = archive.compress_group(OUTPUT / "group_00", [record])':
        'result = idle.idle_compressor()(OUTPUT / "group_00", [record])',
}, {})


if __name__ == "__main__":
    main()
