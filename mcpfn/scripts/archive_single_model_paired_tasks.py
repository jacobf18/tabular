"""Losslessly preserve the completed V19 task tree before the structured-teacher trial."""

import archive_single_model_replicated_tasks as template
from single_model_registered_adaptation import install


install(template, globals(), {
    'SOURCE = ROOT / "replicated_decoder_v18/training_batches"': 'SOURCE = ROOT / "paired_decoder_v19/training_batches"',
    'OUTPUT = ROOT / "replicated_task_archive_v1"': 'OUTPUT = ROOT / "paired_task_archive_v1"',
    '"replicated_decoder_v18/status.json", "replicated_decoder_v18_audit/audit.json",':
        '"paired_decoder_v19/status.json", "paired_decoder_v19_audit/audit.json",',
    '"replicated_decoder_v18_endpoint_replay/audit.json", "replicated_decoder_v18_pipeline/status.json",':
        '"paired_decoder_v19_endpoint_replay/audit.json", "paired_decoder_v19_pipeline/status.json",',
    '"replicated_decoder_v18_pipeline/result.json", "replicated_decoder_v18_confirmation/audit.json"':
        '"paired_decoder_v19_pipeline/result.json", "paired_decoder_v19_confirmation/audit.json"',
    'unsuccessful V18 lifecycle': 'unsuccessful V19 lifecycle',
    'independent V18 audit': 'independent V19 audit',
    'REPO / "tests/test_single_model_replicated_task_archive.py"]':
        'Path(template.__file__).resolve(), REPO / "tests/test_single_model_teacher_terminal_archives.py"]',
}, {})


if __name__ == "__main__":
    main()
