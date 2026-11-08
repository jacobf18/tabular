"""Prepare the original trainer for complete four-draw CPU capture and CUDA updates."""

import single_model_four_draw_trial_budget as budget
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY = budget.POLICY + (
    " Full-trial source integration retains the original optimizer and exact captured "
    "individual gradients. Allocate a168-row FP32 CPU capture bank only after complete "
    "admission; release the preceding CUDA gradient bank before capturing the next admitted "
    "attempt. Transfer and byte-check every defined row using the already measured staging "
    "procedure, release the CPU bank before the unchanged CUDA quarter-protected update, "
    "and retain the most recent actual gradient bank through subsequent data-only "
    "rejections for the terminal optimizer audit. Store a separate exact transfer receipt "
    "for every optimizer attempt. No inference-time change, extra evaluation boundary, "
    "undefined-metric fabrication, reduced replay coverage or scientific execution is "
    "authorized by this source adapter. Full entry-point, qualification and capacity "
    "integration must be complete and separately registered before execution."
)


def adapt_training(source):
    for before, after in (
        ('import preflight_single_model_balanced_native_quarter as preflight', 'import preflight_single_model_four_draw as preflight'),
        ('import os\n', 'import os\nimport resource\n'),
        ('OUTPUT = ROOT / "balanced_native_long_v56"', 'OUTPUT = ROOT / "four_draw_v57"'),
        ('SCRATCH = Path("/mnt/matrixformer_scratch/balanced_native_long_v56")', 'SCRATCH = Path("/mnt/matrixformer_scratch/four_draw_v57")'),
        ('STEPS, BOUNDARIES = 256, (0, 10, 25, 50, 64, 128, 192, 256)', 'STEPS, BOUNDARIES = 128, (0, 5, 13, 25, 32, 64, 96, 128)'),
        ('DATA_SEED, GRADIENT_SEED = 1481100901, 1482100901', 'DATA_SEED, GRADIENT_SEED = 1511100901, 1512100901'),
        ('CONFIRMATION_SEEDS = (1491100901, 1492100901)', 'CONFIRMATION_SEEDS = (1521100901, 1522100901)'),
        ('all256 attempts', 'all128 attempts'),
        ('boundaries0/10/25/50/64/128/192/256', 'boundaries0/5/13/25/32/64/96/128'),
        ('gradients = torch.empty((84, sum(row[2] for row in selected)), device="cuda", dtype=torch.float32)', 'gradients = None'),
        ('            if checker.eligible:\n',
         '            if checker.eligible:\n'
         '                gradients = None\n'
         '                torch.cuda.empty_cache()\n'
         '                captured = torch.empty((168, sum(row[2] for row in selected)), device="cpu", dtype=torch.float32)\n'),
        ('criterion, teachers, checker, gradients)', 'criterion, teachers, checker, captured)'),
        ('                paired.validate_layout(entries, batch["tasks"], before)\n',
         '                paired.validate_layout(entries, batch["tasks"], before)\n'
         '                capture_peak_allocated = torch.cuda.max_memory_allocated()\n'
         '                capture_peak_reserved = torch.cuda.max_memory_reserved()\n'
         '                torch.cuda.empty_cache()\n'
         '                gradients, transfer = staging.stage_for_update(captured, entries, batch["tasks"], before, "cuda")\n'
         '                del captured\n'
         '                transfer_path = output / "gradient_transfers" / f"step_{step:03d}.json"\n'
         '                write_json(transfer_path, transfer)\n'),
        ('                record.update(retention_manifest=str(retention_directory / "manifest.json"),',
         '                record.update(gradient_transfer=str(transfer_path), gradient_transfer_sha256=sha256(transfer_path),\n'
         '                    capture_peak_allocated_bytes=capture_peak_allocated, capture_peak_reserved_bytes=capture_peak_reserved,\n'
         '                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,\n'
         '                    peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,\n'
         '                    cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory, allocator_fraction=config["base"]["memory_fraction"])\n'
         '                record.update(retention_manifest=str(retention_directory / "manifest.json"),'),
        ('or config["duration_extension"] != budget.SPECIFICATION',
         'or config["duration_extension"] != budget.SPECIFICATION\n'
         '            or config["four_draw_batching"] != batching.SPECIFICATION\n'
         '            or config["gradient_staging_policy"] != staging.POLICY'),
    ):
        source = replace_once(source, before, after)
    return replace_assignment(source, "POLICY", POLICY)


def adapt_registration(source):
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "four_draw_batching": batching.SPECIFICATION, "gradient_staging_policy": staging.POLICY,')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update((REPO / "scripts").glob("*four_draw*.py"))\n'
        '    sources.update((REPO / "tests").glob("test_single_model_four_draw*.py"))\n'
        '    sources.update(Path(module.__file__).resolve()')
