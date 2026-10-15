"""Preserve the fully rejected V19 endpoint losslessly before another preflight."""

import inspect

import archive_single_model_decoder_endpoint as template
import archive_single_model_idle_evidence as idle
from single_model_registered_adaptation import install


def terminal_evidence():
    root = archive.ROOT
    paths = [root / name for name in (
        "paired_decoder_v19/status.json", "paired_decoder_v19_pipeline/status.json",
        "paired_decoder_v19_pipeline/result.json", "paired_decoder_v19_audit/audit.json",
        "paired_decoder_v19_endpoint_replay/audit.json", "paired_decoder_v19_confirmation/audit.json",
        "paired_decoder_v19_confirmation/gate.json")]
    trainer, pipeline, result, audit, replay, confirmation, gate = map(read, paths)
    if (trainer["state"] != "training_complete" or pipeline["state"] != "complete"
            or any(Path(f'/proc/{current["pid"]}').exists() for current in (trainer, pipeline))
            or not result["complete"] or result["gate"]["passed"]
            or not result["gate"]["confirmation_evaluated"] or result["campaign"] is not None
            or result["review"] is not None or not audit["complete_training_audit"]
            or audit["through_step"] != 64 or not replay["complete"] or replay["endpoint_grids"] != 56
            or not confirmation["complete"] or confirmation["prediction_grids"] != 1136 or gate["passed"]):
        raise ValueError("Require terminal fully audited and rejected V19, including independent confirmation")
    return paths, audit


def verified_record():
    paths, audit = terminal_evidence()
    return idle.original_record(SOURCE, audit["artifact_sha256"][str(SOURCE)])


install(template, globals(), {
    'OUTPUT = archive.ROOT / "decoder_endpoint_archive_v1"': 'OUTPUT = archive.ROOT / "paired_endpoint_archive_v1"',
    'SOURCE = archive.ROOT / "decoder_repair_v14/raw_final_update.pt"': 'SOURCE = archive.ROOT / "paired_decoder_v19/raw_final_update.pt"',
    inspect.getsource(template.source_record): 'def source_record():\n    return verified_record()\n',
    'inputs = [SOURCE.parent / "status.json", archive.ROOT / "decoder_repair_v14_audit/audit.json"]':
        'inputs, _audit = terminal_evidence()',
    'sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(), REPO / "tests/test_single_model_endpoint_archive.py"]':
        'sources = [Path(__file__).resolve(), Path(template.__file__).resolve(), Path(idle.__file__).resolve(),\n'
        '               Path(archive.__file__).resolve(), REPO / "tests/test_single_model_paired_endpoint_archive.py"]',
    '"minimum_free_bytes_during_staging": archive.MINIMUM_FREE': '"minimum_free_bytes_during_staging": idle.MINIMUM_FREE',
    'result = archive.compress_group(OUTPUT / "group_00", [record])': 'result = idle.idle_compressor()(OUTPUT / "group_00", [record])',
}, {})


if __name__ == "__main__":
    main()
