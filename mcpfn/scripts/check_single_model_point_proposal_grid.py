"""Fixed stronger-margin/contraction grid on cached synthetic training tasks only."""

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

import check_single_model_point_finite_steps as finite
import diagnose_single_model_point_alignment as alignment
from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_json
from single_model_point_metric_gradients import point_metric_terms
import single_model_point_proposal_grid as grid
from single_model_streamed_point_gradients import streamed_point_predictions
from streamed_large_matrix_training import atomic_torch_save
from train_streamed_large_matrices import append_json, wait_for_gpu


SOURCE = finite.SOURCE
FINITE = alignment.ROOT / "point_finite_steps_v1"
POLICY = ("Fixed nine-candidate grid: joint42-gradient inward margins .001/.01/.1 and actual-FP32 step contractions "
          "1/.5/.25 at fixed V8 and persisted projected80. No new gradient data or neural optimization. "
          "Save every solver failure and every rounded candidate; require all42 realized products nonnegative, "
          "at least one positive, and displacement norm at most twice the original Adam norm before any finite evaluation. "
          "Evaluate each admissible candidate and before on all14 original synthetic training tasks, twice with original seeds. "
          "Training-task feasibility requires every defined point metric to retain on each repeat and some strict improvement. "
          "No case filtering, sign tolerance, candidate promotion, validation/confirmation, or real benchmarks. "
          "The22 held-out retention guards remain unchanged and are not assessed by this diagnostic.")


def read(path):
    return json.loads(Path(path).read_text())


def verify_registration(output):
    config = read(output / "config.json")
    if (read(output / "registration.json")["config_sha256"] != sha256(output / "config.json")
            or config["grid"] != grid.specifications() or config["policy"] != POLICY
            or config["maximum_norm_ratio"] != grid.MAXIMUM_NORM_RATIO or config["repeats"] != grid.REPEATS):
        raise ValueError("Never alter the registered margin/contraction grid")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def register(output):
    if output.exists():
        raise FileExistsError("Never overwrite or restart a diagnostic")
    if shutil.disk_usage(alignment.ROOT).free < 28 * 1024**3:
        raise OSError("Reserve projection/prediction storage while preserving18 GiB free")
    source_config = alignment.verify_registration(SOURCE)
    finite.verify_registration(FINITE)
    if (read(SOURCE / "status.json")["state"] != "complete" or not read(SOURCE / "audit.json")["complete"]
            or read(FINITE / "status.json")["state"] != "complete" or not read(FINITE / "result.json")["complete"]):
        raise ValueError("Require complete prior gradient and finite-step evidence")
    inputs = {**source_config["input_sha256"], **read(SOURCE / "audit.json")["input_sha256"]}
    inputs.update({str(directory / name): sha256(directory / name)
                   for directory, names in ((SOURCE, ("config.json", "registration.json", "audit.json", "result.json")),
                                            (FINITE, ("config.json", "registration.json", "result.json"))) for name in names})
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_point_proposal_grid.py",
               REPO / "tests/test_single_model_point_proposal_grid_run.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    hashes = {**source_config["source_sha256"], **{str(path): sha256(path) for path in sources}}
    verify_hashes(inputs)
    verify_hashes(hashes)
    output.mkdir(parents=True)
    write_json(output / "config.json", {"source_config": source_config, "grid": grid.specifications(), "policy": POLICY,
               "maximum_norm_ratio": grid.MAXIMUM_NORM_RATIO, "repeats": grid.REPEATS,
               "input_sha256": inputs, "source_sha256": hashes})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "grid_projections_performed": False, "grid_predictions_evaluated": False})
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def load_gradients(result, elements, device):
    gradients = torch.empty((len(result["entries"]), elements), device=device, dtype=torch.float32)
    seen = set()
    for task in result["tasks"]:
        saved = torch.load(task["artifact"], map_location="cpu", weights_only=False, mmap=True)
        for vector, entry in zip(saved["gradients"], saved["entries"]):
            if (entry != result["entries"][entry["index"]] or entry["index"] in seen
                    or vector.shape != (elements,) or not torch.isfinite(vector).all()):
                raise ValueError("Original complete parameter-gradient inventory must remain intact")
            seen.add(entry["index"])
            gradients[entry["index"]].copy_(vector)
        del saved
    if seen != set(range(42)) or len(result["entries"]) != 42:
        raise ValueError("Require all fourteen composite and twenty-eight point gradients")
    return gradients


def panel(rows, cases, role):
    selected = [row for row in rows if row["role"] == role]
    if len(selected) != len(cases) * grid.REPEATS:
        raise ValueError("Every original task and repeat must be present")
    result = {}
    for case in cases:
        current = sorted((row for row in selected if row["task"] == case["name"]), key=lambda row: row["repeat"])
        if [row["repeat"] for row in current] != list(range(grid.REPEATS)):
            raise ValueError("Repeated case identity changed")
        names = set(current[0]["metrics"])
        if any(set(row["metrics"]) != names for row in current):
            raise ValueError("Point metric names must remain identical")
        for name in sorted(names):
            records = [row["metrics"][name] for row in current]
            result[f'{case["name"]}/{name}'] = [None if record is None else record["value"] for record in records]
    return result


def project_candidates(output, name, result, snapshot):
    directory = output / name
    gradients = load_gradients(result, snapshot["before"].numel(), "cuda")
    before, tentative = snapshot["before"].to("cuda"), snapshot["tentative"].to("cuda")
    gram, products, norm = alignment.projection.gram_and_products(gradients, before.double() - tentative.double())
    np.testing.assert_allclose(gram, result["raw_gram"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(products, result["adam_products"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(norm, result["adam_norm"], rtol=1e-12, atol=1e-12)
    records = []
    for margin in grid.MARGINS:
        solution, full, error = None, None, None
        try:
            solution = alignment.projection.solve_projection(gram, products, norm, margin_ratio=margin)
            alignment.preflight.verify_certificate(gram, products, norm, solution, margin)
            full = alignment.projection.projected_parameters(tentative, gradients, solution["raw_gradient_coefficients"])
        except (ValueError, AssertionError) as exception:
            error = repr(exception)
        for specification in (item for item in grid.specifications() if item["margin"] == margin):
            record = {**specification, "solver_error": error, "solution": solution, "artifact": None,
                      "admissible": False, "finite_evaluated": False}
            if full is not None:
                parameters = grid.scaled_parameters(before, full, specification["scale"])
                actual, actual_norm = grid.direction_products(gradients, before.double() - parameters.double())
                decision = grid.displacement_decision(actual, actual_norm, norm)
                path = directory / f'{specification["name"]}.pt'
                alignment.storage_guard(output)
                atomic_torch_save({"parameters": parameters.cpu(), "specification": specification, "solution": solution,
                                   "actual_products": actual.tolist(), "actual_norm": actual_norm, "decision": decision}, path)
                record.update(artifact=str(path), sha256=sha256(path), actual_products=actual.tolist(), actual_norm=actual_norm,
                              decision=decision, admissible=decision["admissible"])
                del parameters
            records.append(record)
        del full
    del gradients, before, tentative
    gc.collect()
    torch.cuda.empty_cache()
    write_json(directory / "proposals.json", records)
    return records


def evaluate_role(output, name, role, model, criterion, cases, options):
    rows = []
    for task in cases:
        cached_path = SOURCE / "tasks" / f'task_{task["index"]:02d}.pt'
        cached = torch.load(cached_path, map_location="cpu", weights_only=False)
        episode = alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
        for repeat in range(grid.REPEATS):
            write_json(output / "status.json", {"state": "checking_grid", "checkpoint": name, "role": role,
                       "task": task["name"], "repeat": repeat, "pid": os.getpid(), "updated_unix": time.time()})
            torch.manual_seed(task["gradient_seed"])
            with torch.no_grad(), compute_context(episode.source.device):
                prediction = streamed_point_predictions(model, criterion, episode.source, episode.mask, **options)
            metrics = alignment.metric_summary(prediction, episode, point_metric_terms(prediction, episode))
            path = output / name / f'{role}_task_{task["index"]:02d}_repeat_{repeat}.npz'
            alignment.storage_guard(output)
            atomic_npz(path, prediction=prediction.cpu().numpy())
            row = {"role": role, "task": task["name"], "repeat": repeat, "metrics": metrics,
                   "cached_task": str(cached_path), "cached_task_sha256": sha256(cached_path),
                   "prediction": str(path), "prediction_sha256": sha256(path)}
            rows.append(row)
            append_json(output / name / "cases.jsonl", row)
            del prediction
        del cached, episode
    return rows


def execute(output):
    config = verify_registration(output)
    wait_for_gpu(output, True, False)
    source_config = config["source_config"]
    model, optimizer, criterion, teachers, sampler = alignment.mixture.original_setup(source_config["base"])
    del optimizer, teachers, sampler
    model.train()
    reports = {}
    for name, specification in source_config["states"].items():
        (output / name).mkdir()
        result = read(SOURCE / name / "result.json")
        snapshot = torch.load(result["proposal"], map_location="cpu", weights_only=True, mmap=True)
        checkpoint = torch.load(specification["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        if not torch.equal(alignment.preflight.flatten(model).cpu(), snapshot["before"]):
            raise ValueError("Original checkpoint does not bind its diagnostic displacement")
        write_json(output / "status.json", {"state": "projecting_grid", "checkpoint": name,
                   "pid": os.getpid(), "updated_unix": time.time()})
        proposals = project_candidates(output, name, result, snapshot)
        rows = evaluate_role(output, name, "before", model, criterion, source_config["cases"], source_config["streaming"])
        initial = panel(rows, source_config["cases"], "before")
        for proposal in proposals:
            if not proposal["admissible"]:
                continue
            saved = torch.load(proposal["artifact"], map_location="cpu", weights_only=True, mmap=True)
            alignment.preflight.assign(model, saved["parameters"])
            current = evaluate_role(output, name, proposal["name"], model, criterion,
                                    source_config["cases"], source_config["streaming"])
            rows.extend(current)
            proposal["finite_evaluated"] = True
            proposal["finite"] = grid.finite_decision(initial, panel(current, source_config["cases"], proposal["name"]))
            print(json.dumps({"checkpoint": name, "candidate": proposal["name"],
                              **{key: value for key, value in proposal["finite"].items() if key != "changes"}}), flush=True)
            del saved
        alignment.preflight.assign(model, snapshot["before"])
        reports[name] = {"proposals": proposals, "before_panel": initial, "forward_passes": len(rows),
                         "parameters_restored": True}
        write_json(output / name / "result.json", reports[name])
        del snapshot
        gc.collect()
        torch.cuda.empty_cache()
    verify_registration(output)
    write_json(output / "result.json", {"capture_complete": True, "states": reports,
               "training_restarted": False, "trained_checkpoint_created": False, "real_benchmarks_used": False,
               "validation_or_confirmation_used": False, "goal_achieved": False})


def audit(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Independent audit must use CPU only")
    config = verify_registration(output)
    if not read(output / "result.json")["capture_complete"] or (output / "audit.json").exists():
        raise ValueError("Require a complete unaudited grid")
    cases = config["source_config"]["cases"]
    hashes, summaries = {}, {}
    for name in config["source_config"]["states"]:
        result_path = output / name / "result.json"
        result = read(result_path)
        hashes[str(result_path)] = sha256(result_path)
        source = read(SOURCE / name / "result.json")
        snapshot = torch.load(source["proposal"], map_location="cpu", weights_only=True, mmap=True)
        gradients = load_gradients(source, snapshot["before"].numel(), "cpu")
        gram, original_products, original_norm = np.asarray(source["raw_gram"]), np.asarray(source["adam_products"]), source["adam_norm"]
        for proposal in result["proposals"]:
            if proposal["artifact"] is None:
                if proposal["solver_error"] is None or proposal["admissible"] or proposal["finite_evaluated"]:
                    raise ValueError("Solver failure must remain explicit and unavailable")
                continue
            hashes[proposal["artifact"]] = proposal["sha256"]
            saved = torch.load(proposal["artifact"], map_location="cpu", weights_only=True, mmap=True)
            alignment.preflight.verify_certificate(gram, original_products, original_norm, saved["solution"], proposal["margin"])
            full = alignment.projection.projected_parameters(snapshot["tentative"], gradients, saved["solution"]["raw_gradient_coefficients"])
            expected = grid.scaled_parameters(snapshot["before"], full, proposal["scale"])
            lower, upper = torch.nextafter(expected, torch.full_like(expected, -torch.inf)), torch.nextafter(expected, torch.full_like(expected, torch.inf))
            if not ((saved["parameters"] >= lower) & (saved["parameters"] <= upper)).all():
                raise ValueError("Independent rounded candidate arithmetic exceeds one output ULP")
            products, norm = grid.direction_products(gradients, snapshot["before"].double() - saved["parameters"].double())
            np.testing.assert_allclose(products, saved["actual_products"], rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(norm, saved["actual_norm"], rtol=1e-12, atol=1e-12)
            decision = grid.displacement_decision(products, norm, original_norm)
            alignment.compare_metric_records(decision, saved["decision"])
            if decision["admissible"] != proposal["admissible"] or proposal["admissible"] != proposal["finite_evaluated"]:
                raise ValueError("Never evaluate inadmissible or omit admissible proposals")
            del full, expected, lower, upper, saved
        del gradients, snapshot
        gc.collect()
        path = output / name / "cases.jsonl"
        hashes[str(path)] = sha256(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            hashes[row["cached_task"]] = row["cached_task_sha256"]
            hashes[row["prediction"]] = row["prediction_sha256"]
            cached = torch.load(row["cached_task"], map_location="cpu", weights_only=False)
            episode = alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
            with np.load(row["prediction"], allow_pickle=False) as saved:
                prediction = torch.from_numpy(saved["prediction"])
            metrics = alignment.metric_summary(prediction, episode, point_metric_terms(prediction, episode))
            alignment.compare_metric_records(metrics, row["metrics"])
            del prediction, episode, cached
        initial = panel(rows, cases, "before")
        if initial != result["before_panel"] or len(rows) != result["forward_passes"]:
            raise ValueError("Original repeated baseline panel changed")
        summaries[name] = []
        for proposal in result["proposals"]:
            record = {key: proposal[key] for key in ("name", "margin", "scale", "admissible", "finite_evaluated")}
            if proposal["finite_evaluated"]:
                current = grid.finite_decision(initial, panel(rows, cases, proposal["name"]))
                if current != proposal["finite"]:
                    raise ValueError("Strict every-repeat finite-step feasibility does not reproduce")
                record.update({key: value for key, value in current.items() if key != "changes"})
            summaries[name].append(record)
    verify_hashes(hashes)
    verify_registration(output)
    write_json(output / "audit.json", {"complete": True, "states": summaries, "input_sha256": hashes,
               "cuda_initialized": False, "autograd_replayed": False, "original_source_geometry_previously_audited": True,
               "every_available_candidate_arithmetic_and_direction_recomputed": True,
               "every_saved_prediction_cpu_rescored": True, "checkpoint_promoted": False, "goal_achieved": False})
    lines = ["# Stronger point-protection margins and contracted steps", "", POLICY, "",
             "| Fixed state | Margin | Step scale | Admissible | Improved | Worsened | Every repeat retained |",
             "|---|---:|---:|:---:|---:|---:|:---:|"]
    for name, records in summaries.items():
        for record in records:
            lines.append(f'| {name} | {record["margin"]} | {record["scale"]} | {record["admissible"]} | '
                         f'{record.get("improved_slots", "—")} | {record.get("worsened_slots", "—")} | {record.get("all_repeats_retained", "—")} |')
    lines.extend(["", "Training-task feasibility is not held-out retention or qualification. All22 guards and independent confirmation remain mandatory.",
                  "No model is promoted, no optimizer training resumes, and the full single-model benchmark goal remains unmet.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = "registered" if args.stage == "run" else "capture_complete"
        if read(output / "status.json")["state"] != expected:
            raise ValueError("Never restart a started stage")
        try:
            write_json(output / "status.json", {"state": "starting" if args.stage == "run" else "auditing",
                       "pid": os.getpid(), "updated_unix": time.time()})
            (execute if args.stage == "run" else audit)(output)
            write_json(output / "status.json", {"state": "capture_complete" if args.stage == "run" else "complete",
                       "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
