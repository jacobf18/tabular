"""Losslessly archive the completed structured-teacher preflight's raw gradients."""

import inspect

import archive_single_model_decoder_endpoint as template
import archive_single_model_idle_evidence as idle
from single_model_registered_adaptation import install


def evidence():
    paths = [SOURCE.parent / name for name in ("status.json", "audit.json", "gate.json", "teacher_replay/result.json")]
    terminal, audit, gate, replay = map(read, paths)
    if (terminal["state"] != "complete" or Path(f'/proc/{terminal["pid"]}').exists()
            or not audit["complete"] or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
            or audit["raw_endpoint"]["gradient_vectors"] != 84 or not gate["passed"] or not gate["preflight_only"]
            or gate["audit_sha256"] != sha256(paths[1]) or not audit["teacher_replay"]["complete"]
            or audit["teacher_replay"]["full_grids"] != 28 or not replay["complete"] or len(replay["rows"]) != 28):
        raise ValueError("Require the terminal successful preflight and all independent teacher/optimizer evidence")
    return paths, audit


def verified_record():
    paths, audit = evidence()
    return idle.original_record(SOURCE, audit["artifact_sha256"][str(SOURCE)])


install(template, globals(), {
    'OUTPUT = archive.ROOT / "decoder_endpoint_archive_v1"': 'OUTPUT = archive.ROOT / "structured_teacher_preflight_archive_v1"',
    'SOURCE = archive.ROOT / "decoder_repair_v14/raw_final_update.pt"': 'SOURCE = archive.ROOT / "structured_teacher_preflight_v1/raw_update.pt"',
    inspect.getsource(template.source_record): 'def source_record():\n    return verified_record()\n',
    'inputs = [SOURCE.parent / "status.json", archive.ROOT / "decoder_repair_v14_audit/audit.json"]': 'inputs, _audit = evidence()',
    'sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(), REPO / "tests/test_single_model_endpoint_archive.py"]':
        'sources = [Path(__file__).resolve(), Path(template.__file__).resolve(), Path(idle.__file__).resolve(),\n'
        '               Path(archive.__file__).resolve(), REPO / "tests/test_single_model_teacher_preflight_archive.py"]',
    '"minimum_free_bytes_during_staging": archive.MINIMUM_FREE': '"minimum_free_bytes_during_staging": idle.MINIMUM_FREE',
    'result = archive.compress_group(OUTPUT / "group_00", [record])': 'result = idle.idle_compressor()(OUTPUT / "group_00", [record])',
}, {})


if __name__ == "__main__":
    main()
