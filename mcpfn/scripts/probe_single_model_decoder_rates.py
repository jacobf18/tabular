"""Compare three fresh-Adam rates on a fixed audited synthetic batch, never on validation."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
import preflight_single_model_targeted_decoder as source
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_json
from recover_single_model_decoder_finite_audit import compare_nested
import single_model_decoder_training as decoder
import single_model_finite_guarded_audit as checks
from single_model_point_proposal_grid import direction_products


OUTPUT = source.ROOT / "decoder_rate_probe_v1"
WAIT_FOR = source.ROOT / "targeted_decoder_v15_pipeline_recovery1"
RATES = (2e-6, 2e-5, 1e-4)
REPEATS = (0, 1)
POLICY = ("A bounded optimizer-feasibility diagnostic, not training or model selection. After the unchanged V15 "
          "qualification/campaign pipeline finishes, reuse the already audited fourteen synthetic training tasks "
          "and all42 decoder gradients from targeted_decoder_preflight_v1, at its identical fixed V14 input. "
          "Recompute actual fused-CUDA AdamW proposals independently at2e-6,2e-5,1e-4 with fresh moments; do not "
          "rescale an old rounded proposal or carry state between rates. Preserve the original composite weights, "
          "0.1 joint protection,2x rate-specific Adam norm bound and exact finite-point rollback. Independently "
          "repeat all before/candidate forwards twice with the original seeds. A rate is training-feasible only "
          "if both repetitions retain all defined point metrics and improve at least one. Reproduce the original "
          "2e-6 endpoint bitwise. CPU-audit every candidate endpoint, optimizer, cached task and saved prediction. "
          "Store duplicate composite gradients by verified reference to the immutable original raw artifact. "
          "No new data or autograd, validation, confirmation, checkpoint nomination, alias, real benchmark, "
          "guard relaxation, inference ensemble, or modification of the ongoing V15 trial.")


def read(path):
    return source.read(path)


def status(output, state, **fields):
    source.status(output, state, **fields)


def compact_snapshot(snapshot, gradients, count):
    if not torch.equal(snapshot["task_gradients"], gradients[:count]):
        raise ValueError("Only exactly duplicated composite gradients may use a shared reference")
    return {key: value for key, value in snapshot.items() if key != "task_gradients"}


def restore_snapshot(snapshot, gradients, count):
    if "task_gradients" in snapshot or gradients.ndim != 2 or not 0 < count < len(gradients):
        raise ValueError("Require the declared compact endpoint and complete shared gradient inventory")
    return {**snapshot, "task_gradients": gradients[:count]}


def fresh_optimizer(view, weight_decay):
    return torch.optim.AdamW([{"params": list(view.parameters()), "lr": source.RATE}], lr=source.RATE,
                             weight_decay=weight_decay, betas=(.9, .999), eps=1e-8, fused=True)


def repeat_decision(before, candidate):
    if set(before) != {str(index) for index in REPEATS} or set(candidate) != set(before):
        raise ValueError("Both original-seed repetitions and every metric slot remain mandatory")
    decisions = {index: source.training.guarded.metric_decision(before[index], candidate[index]) for index in before}
    return {"repetitions": decisions, "finite_feasible": all(row["accepted"] for row in decisions.values())}


def verify(output):
    config = read(output / "config.json")
    if (read(output / "registration.json")["config_sha256"] != sha256(output / "config.json")
            or config["policy"] != POLICY or config["rates"] != list(RATES) or config["repeats"] != list(REPEATS)
            or config["wait_for"] != str(WAIT_FOR) or config["source"] != str(source.OUTPUT)):
        raise ValueError("Never change the declared fixed-batch rate probe")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    source.verify(source.OUTPUT)
    return config


def register(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Require the fresh declared diagnostic")
    source.verify(source.OUTPUT)
    audit, gate = read(source.OUTPUT / "audit.json"), read(source.OUTPUT / "gate.json")
    if (read(source.OUTPUT / "status.json")["state"] != "complete" or not audit["complete"] or not gate["passed"]
            or not gate["preflight_only"] or audit["training_tasks"] != 14 or audit["prediction_grids"] != 28
            or audit["raw_endpoint"]["gradient_vectors"] != 42):
        raise ValueError("Require the complete successful actual decoder optimizer preflight")
    verify_hashes(audit["artifact_sha256"])
    inputs = {**audit["artifact_sha256"], str(source.OUTPUT / "audit.json"): sha256(source.OUTPUT / "audit.json"),
              str(source.OUTPUT / "gate.json"): sha256(source.OUTPUT / "gate.json"), str(source.PARENT): sha256(source.PARENT),
              str(WAIT_FOR / "recovery_registration.json"): sha256(WAIT_FOR / "recovery_registration.json")}
    paths = {Path(__file__).resolve(), REPO / "tests/test_single_model_decoder_rates.py"}
    paths.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                 if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                 and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    output.mkdir()
    write_json(output / "config.json", {"policy": POLICY, "source": str(source.OUTPUT), "rates": list(RATES),
               "repeats": list(REPEATS), "wait_for": str(WAIT_FOR), "input_sha256": inputs,
               "source_sha256": {str(path): sha256(path) for path in paths}, "fresh_data_or_gradients": False})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
               "registered_unix": time.time(), "model_evaluated": False, "validation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


@torch.no_grad()
def evaluate(output, role, repeat, model, criterion, tasks):
    rows, values = [], {}
    for task in tasks:
        cached = torch.load(task["artifact"], map_location="cpu", weights_only=False)
        episode = source.training.alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
        torch.manual_seed(task["gradient_seed"])
        with compute_context(episode.source.device):
            prediction = source.training.streamed_point_predictions(model, criterion, episode.source, episode.mask, **source.training.TRAIN_OPTIONS)
        metrics = source.training.alignment.metric_summary(prediction, episode, source.training.point_metric_terms(prediction, episode))
        path = output / f"{role}_repeat_{repeat}_task_{task['index']:02d}.npz"
        atomic_npz(path, prediction=prediction.cpu().numpy())
        rows.append({"task": task["name"], "index": task["index"], "prediction": str(path), "sha256": sha256(path), "metrics": metrics})
        values.update({f"{task['name']}/{name}": None if metric is None else metric["value"] for name, metric in metrics.items()})
    report = {"role": role, "repeat": repeat, "rows": rows, "values": values}
    write_json(output / f"{role}_repeat_{repeat}.json", report)
    return values


def wait_for_previous(output):
    while True:
        previous = read(WAIT_FOR / "status.json")
        if previous["state"] == "complete":
            return
        if previous["state"] in {"failed", "interrupted"}:
            raise RuntimeError("Never bypass or restart the existing V15 qualification pipeline")
        os.kill(previous["pid"], 0)
        status(output, "waiting_for_v15", pipeline_pid=previous["pid"], pipeline_state=previous["state"])
        time.sleep(30)


@torch.no_grad()
def run(output):
    config = verify(output)
    if read(output / "status.json")["state"] != "registered":
        raise ValueError("Never restart a started rate probe")
    wait_for_previous(output)
    if shutil.disk_usage(source.ROOT).free < 20 * 1024**3:
        raise OSError("Keep18 GiB free plus the compact endpoint/prediction budget")
    source.wait_for_gpu(output, True, False)
    status(output, "loading_fixed_decoder_state")
    original = torch.load(source.OUTPUT / "raw_update.pt", map_location="cpu", weights_only=False, mmap=True)
    parent = torch.load(source.PARENT, map_location="cpu", weights_only=False, mmap=True)["model"]
    manifest = read(source.OUTPUT / "training_batch/manifest.json")
    selected = read(source.OUTPUT / "decoder_inventory.json")
    raw_gradients = original["gradients"]
    base = read(source.OUTPUT / "config.json")["base"]
    with source.training.alignment.mixture.runtime(base, output):
        model, unused_optimizer, criterion, teachers, sampler = source.training.alignment.mixture.backend.setup(base)
        del unused_optimizer, teachers, sampler
        model.load_state_dict(parent, strict=True)
        model.train()
        view = decoder.DecoderParameterView(model)
        if [[name, list(parameter.shape), parameter.numel()] for name, parameter in view.named_parameters()] != selected:
            raise ValueError("The original decoder inventory remains mandatory")
        gradients = raw_gradients.to("cuda")
        before = {}
        for repeat in REPEATS:
            before[str(repeat)] = evaluate(output, "before", repeat, model, criterion, manifest["tasks"])
            compare_nested(before[str(repeat)], original["record"]["point_before"])
        rows = []
        for index, rate in enumerate(RATES):
            directory = output / f"rate_{index}"
            directory.mkdir()
            model.load_state_dict(parent, strict=True)
            model.train()
            view.zero_grad(set_to_none=True)
            optimizer = fresh_optimizer(view, base["weight_decay"])
            if optimizer.state or not torch.equal(source.training.alignment.preflight.flatten(view).cpu(), original["snapshot"]["before"]):
                raise ValueError("Each rate must reset the exact parameters and all Adam moments")
            candidates = {}

            def score_candidate(current):
                source.training.save_tensor(source.training.alignment.preflight.flatten(current).cpu(), directory / "candidate_parameters.pt")
                for repeat in REPEATS:
                    candidates[str(repeat)] = evaluate(directory, "candidate", repeat, model, criterion, manifest["tasks"])
                return candidates["0"]

            status(output, "testing_rate", rate=rate, index=index)
            try:
                record, snapshot = source.training.guarded.protected_update(view, optimizer, gradients,
                    [task["weight"] for task in manifest["tasks"]], rate, before["0"], score_candidate)
            except source.training.guarded.StepEvaluationError as error:
                source.training.save_tensor({"snapshot": compact_snapshot(error.snapshot, raw_gradients, 14),
                    "record": error.record, "source_raw_sha256": sha256(source.OUTPUT / "raw_update.pt"),
                    "config_sha256": sha256(output / "config.json")}, directory / "failed_endpoint.pt")
                raise
            record.update(step=1, accepted_updates=int(record["accepted"]))
            if index == 0 and (not record["accepted"] or not torch.equal(snapshot["after"], original["snapshot"]["after"])):
                raise ValueError("The original2e-6 GPU endpoint must reproduce bitwise before interpreting larger rates")
            frozen = decoder.frozen_state_equal(parent, model.state_dict(), {row[0] for row in selected})
            decisions = repeat_decision(before, candidates) if candidates else None
            digest = source.training.save_tensor({"snapshot": compact_snapshot(snapshot, raw_gradients, 14),
                "record": record, "optimizer": optimizer.state_dict(), "source_raw_sha256": sha256(source.OUTPUT / "raw_update.pt"),
                "config_sha256": sha256(output / "config.json")}, directory / "endpoint.pt")
            row = {"index": index, "rate": rate, "record": record, "repeated": decisions, "frozen": frozen,
                   "endpoint": str(directory / "endpoint.pt"), "endpoint_sha256": digest,
                   "finite_feasible": bool(record["accepted"] and decisions and decisions["finite_feasible"])}
            rows.append(row)
            write_json(directory / "result.json", row)
            del optimizer, snapshot
    verify(output)
    write_json(output / "result.json", {"complete": True, "before": before, "rates": rows,
               "config_sha256": sha256(output / "config.json"), "checkpoint_selected": False,
               "autograd_replayed": False, "goal_success_not_assessed": True})
    status(output, "awaiting_independent_audit")


def audit_predictions(directory, role, repeat, tasks, hashes):
    path = directory / f"{role}_repeat_{repeat}.json"
    hashes[str(path)] = sha256(path)
    report = read(path)
    if report["role"] != role or report["repeat"] != repeat or len(report["rows"]) != 14:
        raise ValueError("Every original role/repetition/task is required")
    values = {}
    for row, task in zip(report["rows"], tasks):
        if row["task"] != task["name"] or row["index"] != task["index"]:
            raise ValueError("Never reorder or omit a prediction slot")
        hashes[row["prediction"]] = row["sha256"]
        cached = torch.load(task["artifact"], map_location="cpu", weights_only=False)
        episode = source.training.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
        with np.load(row["prediction"], allow_pickle=False) as saved:
            prediction = torch.from_numpy(saved["prediction"])
        metrics = source.training.alignment.metric_summary(prediction, episode, source.training.point_metric_terms(prediction, episode))
        compare_nested(metrics, row["metrics"])
        values.update({f"{task['name']}/{name}": None if metric is None else metric["value"] for name, metric in metrics.items()})
    compare_nested(values, report["values"])
    return values


def audit(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Require a separate CPU-only audit")
    verify(output)
    if read(output / "status.json")["state"] != "awaiting_independent_audit" or (output / "audit.json").exists():
        raise ValueError("Audit each complete probe only once")
    result = read(output / "result.json")
    if (not result["complete"] or [row["rate"] for row in result["rates"]] != list(RATES)
            or [row["index"] for row in result["rates"]] != list(range(len(RATES)))
            or result["config_sha256"] != sha256(output / "config.json") or result["checkpoint_selected"]):
        raise ValueError("Every registered rate, including all failures, must remain present")
    original = torch.load(source.OUTPUT / "raw_update.pt", map_location="cpu", weights_only=False, mmap=True)
    manifest = read(source.OUTPUT / "training_batch/manifest.json")
    config = read(source.OUTPUT / "config.json")
    selected = read(source.OUTPUT / "decoder_inventory.json")
    hashes = {str(output / "result.json"): sha256(output / "result.json")}
    logged_before = {}
    for task, logged in zip(config["batch"]["tasks"], manifest["tasks"]):
        hashes[logged["artifact"]] = logged["sha256"]
        cached = torch.load(logged["artifact"], map_location="cpu", weights_only=False)
        logged_before.update(source.task_audit.check_task(cached, task, logged))
    before = {str(repeat): audit_predictions(output, "before", repeat, manifest["tasks"], hashes) for repeat in REPEATS}
    compare_nested(before, result["before"])
    for measured in before.values():
        compare_nested(measured, logged_before)
        compare_nested(measured, original["record"]["point_before"])
    grids, endpoints = 28, []
    for row in result["rates"]:
        directory = output / f"rate_{row['index']}"
        if row["endpoint"] != str(directory / "endpoint.pt") or row != read(directory / "result.json"):
            raise ValueError("Per-rate report identity changed")
        hashes[str(directory / "result.json")] = sha256(directory / "result.json")
        hashes[row["endpoint"]] = row["endpoint_sha256"]
        payload = torch.load(row["endpoint"], map_location="cpu", weights_only=False, mmap=True)
        if (payload["source_raw_sha256"] != sha256(source.OUTPUT / "raw_update.pt")
                or payload["config_sha256"] != sha256(output / "config.json") or payload["record"] != row["record"]):
            raise ValueError("The compact endpoint must bind the exact shared original gradients")
        snapshot = restore_snapshot(payload["snapshot"], original["gradients"], 14)
        record = row["record"]
        if (record["rate"] != row["rate"] or record["step"] != 1 or record["accepted_updates"] != int(record["accepted"])
                or snapshot["optimizer_before"]["state"] or not torch.equal(snapshot["before"], original["snapshot"]["before"])):
            raise ValueError("Never carry parameters or moments between rates")
        checks.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], before["0"], 42)
        endpoint = checks.check_endpoint({**payload, "snapshot": snapshot, "gradients": original["gradients"]}, record,
            [item[2] for item in selected], config["base"]["weight_decay"], source.RATE, payload["optimizer"])
        if row["index"] == 0 and not torch.equal(snapshot["after"], original["snapshot"]["after"]):
            raise ValueError("Baseline replay must exactly reproduce the original accepted endpoint")
        repeated = None
        if record["point_after"] is not None:
            candidate = {str(repeat): audit_predictions(directory, "candidate", repeat, manifest["tasks"], hashes) for repeat in REPEATS}
            compare_nested(candidate["0"], record["point_after"])
            repeated = repeat_decision(before, candidate)
            path = directory / "candidate_parameters.pt"
            hashes[str(path)] = sha256(path)
            parameters = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            expected = source.training.alignment.projection.projected_parameters(snapshot["tentative"], original["gradients"],
                                                                            record["projection"]["raw_gradient_coefficients"])
            lower, upper = torch.nextafter(expected, torch.full_like(expected, -torch.inf)), torch.nextafter(expected, torch.full_like(expected, torch.inf))
            if not ((parameters >= lower) & (parameters <= upper)).all():
                raise ValueError("Scored candidate exceeds the original one-ULP arithmetic allowance")
            products, norm = direction_products(original["gradients"], snapshot["before"].double() - parameters.double())
            np.testing.assert_allclose(products, record["candidate_products"], rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(norm, record["candidate_norm"], rtol=1e-12, atol=1e-12)
            if record["accepted"] and not torch.equal(parameters, snapshot["after"]):
                raise ValueError("Actual accepted parameters must equal the scored candidate")
            grids += 28
        if repeated != row["repeated"] or row["finite_feasible"] != bool(record["accepted"] and repeated and repeated["finite_feasible"]):
            raise ValueError("Strict two-repeat feasibility must reproduce independently")
        if row["frozen"]["frozen_tensors"] != 248 or not row["frozen"]["backbone_and_buffers_bitwise_unchanged"]:
            raise ValueError("Every nondecoder model tensor must remain frozen")
        endpoints.append(endpoint)
    verify_hashes(hashes)
    verify(output)
    write_json(output / "audit.json", {"complete": True, "prediction_grids": grids, "training_tasks": 14,
               "rates": list(RATES), "endpoints": endpoints, "artifact_sha256": hashes, "cuda_initialized": False,
               "autograd_replayed": False, "research_selection_tolerance": 0, "checkpoint_selected": False,
               "goal_success_not_assessed": True})
    lines = ["# Decoder learning-rate feasibility probe", "", POLICY, "",
             "| Adam rate | Accepted first repeat | Both repeats feasible | Candidate/Adam norm |", "|---:|---|---|---:|"]
    for row in result["rates"]:
        record = row["record"]
        ratio = "not scored" if "candidate_norm" not in record else f'{record["candidate_norm"] / record["adam_norm"]:.6f}'
        lines.append(f'| {row["rate"]:.6g} | {record["accepted"]} | {row["finite_feasible"]} | {ratio} |')
    lines.extend(["", "No rate or checkpoint is nominated; these are reused training-batch diagnostics, not held-out performance.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    status(output, "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "audit": audit}[arguments.stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
