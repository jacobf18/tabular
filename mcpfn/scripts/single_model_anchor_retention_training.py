"""Bind full-trial anchor retention to complete training records and independent replay."""

from copy import deepcopy
import json
from pathlib import Path

import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import atomic_npz, sha256, write_json
import single_model_anchor_retention as retention
from single_model_composite_gradient_storage import byte_equal
from single_model_registered_adaptation import replace_once
from train_streamed_large_matrices import append_json


POLICY = ("Add the prospectively fixed native-anchor retention term to all admitted V37 synthetic batches. "
    "The reference remains the frozen V29 initializer throughout training, never the current student or "
    "a probe endpoint. Preserve every original teacher and point metric. Bind each batch's complete retention "
    "manifest in its training log; data-only rejections create no retention artifacts. Independently regenerate "
    "every actual task under the existing training audit, check all eligibility decisions, replay every added "
    "reference grid, and require the complete hash-bound retention audit before qualification. Counts derive "
    "from actual admitted tasks, not a fixed number of native priors per batch. No additional inference model, "
    "held-out change, selection shortcut, training seed reuse or historical full-autograd replay claim.")


def layout(scratch):
    return {"targets": str(Path(scratch) / "anchor_retention_training"),
            "replay": str(Path(scratch) / "anchor_retention_replay"), "provider_persistent": False}


def prepare(scratch):
    for name in ("targets", "replay"):
        Path(layout(scratch)[name]).mkdir()


def directory(scratch, step):
    if type(step) is not int or step not in range(1, 65):
        raise ValueError("Keep the original64 registered optimizer attempts")
    return Path(layout(scratch)["targets"]) / f"step_{step:03d}"


def identity(config):
    result = {"checkpoint": config["parent_checkpoint"], "sha256": config["input_sha256"][config["parent_checkpoint"]]}
    if (result["sha256"] != retention.REFERENCE_SHA256 or config["anchor_retention"] != retention.SPECIFICATION
            or config["retention_backward_sha256"] != retention.BACKWARD_SOURCE_SHA256
            or config["retention_storage"] != layout(config["scratch"])):
        raise ValueError("Bind the exact V29 initializer, prospectively fixed penalty and declared diagnostic paths")
    return result


def frozen_reference(model, expected):
    reference = deepcopy(model).eval().requires_grad_(False)
    retention.verify_model(reference, expected)
    return reference


def adapt_training(source):
    for old, new in {
        'or config["policy"] != POLICY':
            'or config["anchor_retention"] != retention.SPECIFICATION\n'
            '            or config["retention_backward_sha256"] != retention.BACKWARD_SOURCE_SHA256\n'
            '            or config["retention_storage"] != retention_training.layout(SCRATCH)\n'
            '            or config["policy"] != POLICY',
        '        model, view, optimizer, criterion, teachers, sampler, parent_state = setup(config)':
            '        model, view, optimizer, criterion, teachers, sampler, parent_state = setup(config)\n'
            '        retention_identity = retention_training.identity(config)\n'
            '        retention_reference = retention_training.frozen_reference(model, retention_identity)',
        '                with full_network.capture_coordinates(model, view):':
            '                retention_directory = retention_training.directory(SCRATCH, step)\n'
            '                with retention.runtime(original.alignment.mixture.backend, retention_directory,\n'
            '                        retention_reference, retention_identity, query_seed_offset=config["base"]["query_seed_offset"]), \\\n'
            '                        full_network.capture_coordinates(model, view):',
        '                record.update(attempt_kind="optimizer_attempt", admission_manifest=str(admission_path),':
            '                record.update(retention_manifest=str(retention_directory / "manifest.json"),\n'
            '                              retention_manifest_sha256=sha256(retention_directory / "manifest.json"))\n'
            '                record.update(attempt_kind="optimizer_attempt", admission_manifest=str(admission_path),',
    }.items():
        source = replace_once(source, old, new)
    return source


def adapt_registration(source):
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "anchor_retention": retention.SPECIFICATION, '
        '"retention_backward_sha256": retention.BACKWARD_SOURCE_SHA256, '
        '"retention_storage": retention_training.layout(SCRATCH),')
    return replace_once(source, '    SCRATCH.mkdir()', '    SCRATCH.mkdir()\n    retention_training.prepare(SCRATCH)')


class RetentionReplay:
    def __init__(self, training, config, counts, reference, criterion, backend):
        self.training, self.config, self.counts = training, config, counts
        self.reference, self.criterion, self.backend = reference, criterion, backend
        self.expected_identity = identity(config)
        self.verified = retention.verify_model(reference, self.expected_identity)
        self.output = Path(config["retention_storage"]["replay"])
        if not self.output.is_dir() or any(self.output.iterdir()):
            raise ValueError("Use the original empty reserved retention replay directory exactly once")
        self.rows_path = self.output / "tasks.jsonl"
        self.summary_path = self.output / "summary.json"
        self.rows_path.touch()
        self.tasks, self.seen, self.hashes = {}, set(), {}
        self.reference_grids = 0
        trace_path = training.OUTPUT / "training.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        self.hashes[str(trace_path)] = sha256(trace_path)
        expected_steps = list(range(1, training.STEPS + 1))
        if ([row["step"] for row in trace] != expected_steps
                or [batch["step"] for batch in config["batches"]] != expected_steps):
            raise ValueError("Keep every registered attempt, including data-only and finite rejections")
        admitted = set(counts["admitted_steps"])
        expected_directories = {directory(training.SCRATCH, step) for step in admitted}
        if set(Path(config["retention_storage"]["targets"]).iterdir()) != expected_directories:
            raise ValueError("Retention directories must cover exactly admitted attempts")
        for batch, row in zip(config["batches"], trace):
            step = batch["step"]
            if step not in admitted:
                if row["attempt_kind"] != "data_admission_rejected" or any(key.startswith("retention_") for key in row):
                    raise ValueError("Data-only rejection must perform no retention work")
                continue
            manifest_path = directory(training.SCRATCH, step) / "manifest.json"
            if (row["attempt_kind"] != "optimizer_attempt" or row["retention_manifest"] != str(manifest_path)
                    or row["retention_manifest_sha256"] != sha256(manifest_path)):
                raise ValueError("Every actual gradient batch must bind its complete retention manifest")
            manifest = json.loads(manifest_path.read_text())
            if (manifest["complete"] is not True or manifest["policy"] != retention.POLICY
                    or manifest["specification"] != retention.SPECIFICATION or manifest["reference"] != self.verified
                    or manifest["backward_source_sha256"] != retention.BACKWARD_SOURCE_SHA256
                    or manifest["inference_ensemble"] is not False or len(manifest["records"]) != 28
                    or len(batch["tasks"]) != 28):
                raise ValueError("Preserve all28 decisions and the exact frozen reference per admitted batch")
            self.hashes[str(manifest_path)] = sha256(manifest_path)
            for task, decision in zip(batch["tasks"], manifest["records"]):
                path = training.OUTPUT / "training_batches" / f"step_{step:03d}" / f'task_{task["index"]:02d}.pt'
                if str(path) in self.tasks:
                    raise ValueError("Every registered task must have one distinct retention decision")
                self.tasks[str(path)] = (task, decision, manifest_path.parent)
        if len(self.tasks) != counts["training_tasks"]:
            raise ValueError("Retain the full independently derived task count")

    def check(self, path, payload):
        key = str(path)
        if key not in self.tasks or key in self.seen:
            raise ValueError("Audit every independently regenerated training task exactly once")
        task, decision, folder = self.tasks[key]
        episode = self.backend.restore_episode(payload["episode"], "cpu")
        selected = retention.eligible(episode.metadata)
        if (payload["task"] != task or decision["seed"] != episode.metadata["seed"]
                or decision["metadata"] != episode.metadata or decision["eligible"] is not selected
                or decision["shape"] != list(episode.target.shape) or decision["queries"] != int(episode.mask.sum())):
            raise ValueError("Bind native-prior eligibility to the full actual regenerated episode")
        record = {"task": key, "task_sha256": sha256(path), "seed": decision["seed"], "eligible": selected,
                  "queries": int(episode.mask.sum()), "full_episode_and_identity_checked": True}
        self.hashes[key] = record["task_sha256"]
        if selected:
            target_path = folder / f'reference_{decision["seed"]}.pt'
            if decision["artifact"] != str(target_path) or decision["sha256"] != sha256(target_path):
                raise ValueError("Bind the complete archived reference prediction")
            cached = torch.load(target_path, map_location="cpu", weights_only=False)
            if (cached["identity"] != self.expected_identity or cached["specification"] != retention.SPECIFICATION
                    or cached["metadata"] != episode.metadata or not byte_equal(cached["source"], episode.source)
                    or not byte_equal(cached["mask"], episode.mask)):
                raise ValueError("Retain the actual source, original mask and frozen initializer identity")
            retention.validate_target(episode, cached["prediction"])
            device = next(self.reference.parameters()).device
            with torch.no_grad(), compute_context(device):
                prediction = retention.streamed_point_predictions(self.reference, episode.source.to(device),
                    self.criterion, **retention.STREAM_OPTIONS).float().cpu()
            if not byte_equal(prediction, cached["prediction"]):
                raise ValueError("Independent inference must reproduce the whole stored reference grid")
            output = self.output / f'reference_{decision["seed"]}.npz'
            if output.exists():
                raise ValueError("Never overwrite an independently replayed reference grid")
            atomic_npz(output, prediction=prediction.numpy())
            self.hashes.update({str(target_path): decision["sha256"], str(output): sha256(output)})
            record.update(reference=str(target_path), reference_sha256=decision["sha256"],
                          prediction=str(output), prediction_sha256=sha256(output))
            self.reference_grids += 1
        elif any(name in decision for name in ("artifact", "sha256")):
            raise ValueError("Spectral and rating tasks must retain their unchanged original objective")
        append_json(self.rows_path, record)
        self.seen.add(key)

    def finish(self):
        if self.seen != set(self.tasks) or self.summary_path.exists():
            raise ValueError("Finish exactly one complete all-task retention audit")
        if retention.verify_model(self.reference, self.expected_identity) != self.verified:
            raise ValueError("Independent retention inference must preserve the frozen reference bitwise")
        self.hashes[str(self.rows_path)] = sha256(self.rows_path)
        self.hashes[str(self.training.OUTPUT / "config.json")] = sha256(self.training.OUTPUT / "config.json")
        verify_hashes(self.hashes)
        result = {"complete": True, "policy": POLICY, "specification": retention.SPECIFICATION,
            "eligibility_decisions": len(self.seen), "admitted_batches": len(self.counts["admitted_steps"]),
            "reference_grids": self.reference_grids, "reference": self.verified,
            "training_config_sha256": sha256(self.training.OUTPUT / "config.json"),
            "artifact_sha256": self.hashes, "cuda_initialized": torch.cuda.is_initialized(),
            "training_performed": False, "historical_full_autograd_replayed": False,
            "no_inference_ensemble": True, "goal_success_not_assessed": True}
        write_json(self.summary_path, result)
        return {str(self.summary_path): sha256(self.summary_path), str(self.rows_path): sha256(self.rows_path)}


def require_report(training, audit):
    config = training.read(training.OUTPUT / "config.json")
    reference = identity(config)
    output = Path(config["retention_storage"]["replay"])
    report_path, rows_path = output / "summary.json", output / "tasks.jsonl"
    report = training.read(report_path)
    counts = audit["admission_attempts"]
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    expected = {str(training.OUTPUT / "training_batches" / f"step_{step:03d}" / f"task_{index:02d}.pt")
                for step in counts["admitted_steps"] for index in range(28)}
    if (report["complete"] is not True or report["policy"] != POLICY or report["specification"] != retention.SPECIFICATION
            or report["training_config_sha256"] != sha256(training.OUTPUT / "config.json")
            or report["reference"] != {**reference, "state_tensors": 254, "bitwise_equal": True}
            or report["eligibility_decisions"] != counts["training_tasks"] or len(rows) != counts["training_tasks"]
            or {row["task"] for row in rows} != expected or any(row["full_episode_and_identity_checked"] is not True for row in rows)
            or report["admitted_batches"] != len(counts["admitted_steps"])
            or report["reference_grids"] != sum(row["eligible"] for row in rows)
            or report["training_performed"] is not False or report["historical_full_autograd_replayed"] is not False
            or report["no_inference_ensemble"] is not True
            or any(audit["artifact_sha256"].get(str(path)) != sha256(path) for path in (report_path, rows_path))):
        raise ValueError("Qualification requires the complete independently hash-bound retention audit")
    verify_hashes(report["artifact_sha256"])
    return report
