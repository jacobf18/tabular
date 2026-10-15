"""Fifty fresh mixed-prior batches with strong point protection and finite-metric rollback."""

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import torch

from audit_single_matrixformer_structured_training import close_values
import check_single_model_point_proposal_grid as diagnostic
import diagnose_single_model_point_alignment as alignment
from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_finite_guarded_update as guarded
from single_model_point_metric_gradients import point_metric_terms
from single_model_streamed_point_gradients import streamed_point_metric_losses, streamed_point_predictions
from streamed_large_matrix_training import TRAIN_OPTIONS, atomic_torch_save, learning_rate, rng_state
from train_single_matrixformer_column_aligned import repeated_bank
import train_single_matrixformer_family_projection as original
from train_streamed_large_matrices import append_json, wait_for_gpu


ROOT = alignment.ROOT
OUTPUT = ROOT / "finite_guarded_v13"
EVIDENCE = ROOT / "point_proposal_grid_v1"
STEPS = 50
BOUNDARIES = (0, 10, 25, 50)
DATA_SEED, GRADIENT_SEED = 171100901, 172100901
CONFIRMATION_SEEDS = (181100901, 182100901)
POLICY = ("One ordinary V8-initialized network; fifty fresh fourteen-task batches with the original V12 first-fifty "
          "shapes, component cycles, .25 legacy/.25 structured/.5 rating weights and first-fifty learning rates "
          "of the original100-step schedule. Original teachers, Huber/NLL/CDF/query losses and normalization unchanged. "
          "Strong .1 joint composite-plus-point protection, actual FP32 norm cap2x, and exact finite training-point "
          "retention before accepting each update. Rejected attempts restore parameters/moments/rate and advance "
          "to the next fresh registered batch; no fallback or repeated fitting of the rejected data.")
QUALIFICATION = ("Complete fifty attempted batches, audit every cached task, all scheduled raw-prediction scores, "
                 "the unchanged22-guard minimax selection, and the last attempted update's full gradient/projection "
                 "snapshot. Select only a trained strict improvement versus V8 retaining every original guard, "
                 "earlier boundary wins exact ties. Independently confirm the selected checkpoint once on the "
                 "reserved284 fresh cases under the unchanged gate, without fallback. Only then run every frozen "
                 "real campaign and its full per-comparison review. Logged earlier gradient certificates are not "
                 "independent full-vector replays; only the final attempted endpoint retains full raw vectors.")


def read(path):
    return json.loads(Path(path).read_text())


def batches():
    result = deepcopy(original.training_batches()[:STEPS])
    for batch in result:
        for task in batch["tasks"]:
            offset = (batch["step"] - 1) * 1400 + task["index"] * 100
            task["gradient_seed"] = GRADIENT_SEED + offset
            specification = task["specification"].get("structured", task["specification"])
            specification["seed"] = DATA_SEED + offset
    return result


def confirmation_cases():
    return json.loads(json.dumps(repeated_bank(CONFIRMATION_SEEDS[0], confirmation=True)
                                 + alignment.mixture.structured_bank(CONFIRMATION_SEEDS[1], confirmation=True)))


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(nested) for nested in value)
    return 0


def save_tensor(payload, path):
    required = tensor_bytes(payload) + 4 * 1024**2
    if shutil.disk_usage(path.parent).free - required < 18 * 1024**3:
        raise OSError("Do not consume the18 GiB reserve, including the pending atomic write")
    atomic_torch_save(payload, path)
    return sha256(path)


def verify_registration(output):
    config = read(output / "config.json")
    if (sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]
            or config["batches"] != batches() or config["steps"] != STEPS or config["boundaries"] != list(BOUNDARIES)
            or config["confirmation_cases"] != confirmation_cases() or config["policy"] != POLICY
            or config["qualification"] != QUALIFICATION or config["guarded_policy"] != guarded.POLICY
            or config["margin"] != .1 or guarded.MARGIN != .1 or config["learning_rate_horizon"] != 100):
        raise ValueError("Never alter the registered training, rollback, selection or confirmation recipe")
    alignment.fixed.validate_recipe(config["base"])
    if config["base"] != read(alignment.fixed.PRODUCER / "config.json"):
        raise ValueError("Original architecture, teachers, scalar objective or normalization changed")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def register(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Use only the new finite-guarded trial location")
    if shutil.disk_usage(ROOT).free < 38 * 1024**3:
        raise OSError("Reserve the terminal full-gradient endpoint and18 GiB free space")
    diagnostic.verify_registration(EVIDENCE)
    evidence = read(EVIDENCE / "audit.json")
    if read(EVIDENCE / "status.json")["state"] != "complete" or not evidence["complete"]:
        raise ValueError("Require the completed independent stronger-margin audit")
    for name in alignment.STATES:
        candidate = next(row for row in evidence["states"][name] if row["name"] == "margin_0.1_scale_1")
        if not candidate["admissible"] or not candidate["finite_feasible"] or candidate["defined_slots"] != 28:
            raise ValueError("The fixed strong full step must retain every repeated point metric in both diagnostic states")
    verify_hashes(evidence["input_sha256"])
    base = read(alignment.fixed.PRODUCER / "config.json")
    old_configs = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    previous_seeds = set().union(*(alignment.fixed.seeds_in(read(path)) for path in old_configs))
    new_seeds = alignment.fixed.seeds_in([batches(), confirmation_cases()])
    if len(new_seeds) != 1684 or previous_seeds.intersection(new_seeds):
        raise ValueError("Reserve1400 fresh data/backward seeds and284 independent confirmation seeds")
    inputs = [alignment.fixed.PARENT, alignment.fixed.PRODUCER / "config.json",
              alignment.fixed.PRODUCER / "synthetic_bank.pt", alignment.fixed.PRODUCER / "reference.json",
              alignment.fixed.PRODUCER / "validation_0.json", EVIDENCE / "config.json", EVIDENCE / "audit.json",
              EVIDENCE / "result.json", ROOT / "domain_gradient_probe_v1/parameter_inventory.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_finite_guarded_update.py",
               REPO / "tests/test_single_model_finite_guarded_training.py"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"base": base, "model": base["model"], "steps": STEPS, "batches": batches(), "boundaries": list(BOUNDARIES),
              "confirmation_cases": confirmation_cases(), "confirmation_seeds": list(CONFIRMATION_SEEDS),
              "learning_rate_horizon": 100, "margin": guarded.MARGIN, "policy": POLICY, "qualification": QUALIFICATION,
              "guarded_policy": guarded.POLICY, "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in inputs}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in old_configs},
              "raw_endpoint_policy": "Persist full raw gradients and actual optimizer/projection at the final attempted batch, or at a caught protected-step evaluation failure; no claim to replay every earlier full gradient"}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "training_started": False, "confirmation_evaluated": False, "real_benchmarks_evaluated": False})
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def checkpoint(output, model, optimizer, step, accepted, best, reference, selected=False):
    payload = {"model": model.state_dict(), "step": step, "epoch": step, "attempted_batches": step,
               "gradient_updates_performed": accepted, "config_sha256": sha256(output / "config.json"),
               "bank_sha256": reference["bank_sha256"], "best_step": best["step"], "best_score": best["score"],
               "preflight_only": False, "independent_confirmation_passed": False,
               "step_semantics": "Registered attempted batch index; accepted optimizer updates are counted separately"}
    if not selected:
        payload.update(optimizer=optimizer.state_dict(), rng=rng_state())
    return save_tensor(payload, output / ("selected_checkpoint.pth" if selected else "latest_checkpoint.pth"))


def validate(output, model, optimizer, criterion, step, accepted, best, bank, reference, baseline, config):
    directory = output / f"validation_{step:03d}"
    directory.mkdir()
    with alignment.mixture.runtime(config["base"], directory):
        metrics, cases = alignment.mixture.backend.evaluate_bank(model, criterion, bank)
    if len(cases) != 152:
        raise ValueError("Retain every original synthetic development case")
    if step == 0:
        close_values(metrics, baseline)
    decision = alignment.fixed.selection_decision(metrics, baseline, reference["references"], config["base"])
    selected = original.selected_boundary(best, step, decision) if accepted else dict(best)
    if selected != best:
        best = selected
        digest = checkpoint(output, model, optimizer, step, accepted, best, reference, selected=True)
    else:
        digest = None
    row = {"step": step, "accepted_updates": accepted, "student": metrics, "cases": cases,
           "decision": decision, "best_step": best["step"], "selected_checkpoint_sha256": digest}
    write_json(output / f"validation_{step}.json", row)
    append_json(output / "validation.jsonl", {key: value for key, value in row.items() if key != "cases"})
    print(json.dumps({"validation": step, "accepted_updates": accepted, "score": decision["score"],
                      "failed_guards": sum(not item["retained"] for item in decision["comparisons"]), "best_step": best["step"]}), flush=True)
    return best


def capture_batch(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients):
    entries = [None] * 14
    records, values = [], {}
    for task in batch["tasks"]:
        write_json(output / "status.json", {"state": "capturing_gradients", "active_step": batch["step"], "task": task["name"],
                   "pid": os.getpid(), "updated_unix": time.time()})
        episode = sampler.sample(step=1, **task["specification"])
        expected_seed = task["specification"].get("structured", task["specification"])["seed"]
        if episode.metadata.get("seed") != expected_seed or episode.metadata.get("gradient_data") != "synthetic":
            raise ValueError("Every registered training task must bind its fresh synthetic seed")
        teacher, routing = alignment.mixture.backend.teacher_target(teachers, criterion, episode)
        path = directory / f'task_{task["index"]:02d}.pt'
        digest = save_tensor({"episode": alignment.mixture.backend.cache_episode(episode), "teacher": teacher.detach().float().cpu(),
                              "routing": routing, "task": task}, path)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(task["gradient_seed"])
        loss = alignment.mixture.backend.backward_episode(model, criterion, episode, teacher, 1.)
        gradients[task["index"]].copy_(alignment.preflight.flatten(model, gradients=True))
        entries[task["index"]] = {"task": task["name"], "kind": "composite", "index": task["index"]}
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(task["gradient_seed"])
        with compute_context(episode.source.device):
            prediction, terms, losses = streamed_point_metric_losses(model, criterion, episode, **TRAIN_OPTIONS)
        metrics = alignment.metric_summary(prediction.detach(), episode, terms)
        for name, record in metrics.items():
            values[f'{task["name"]}/{name}'] = None if record is None else record["value"]
        defined = [(name, loss) for name, loss in losses.items() if loss is not None]
        for position, (name, objective) in enumerate(defined):
            optimizer.zero_grad(set_to_none=True)
            objective.backward(retain_graph=position + 1 < len(defined))
            index = len(entries)
            gradients[index].copy_(alignment.preflight.flatten(model, gradients=True))
            entries.append({"task": task["name"], "kind": name, "index": index})
        optimizer.zero_grad(set_to_none=True)
        records.append({**task, "artifact": str(path), "sha256": digest, "metrics_before": metrics,
                        "composite_loss": loss, "metadata": deepcopy(episode.metadata)})
        del prediction, terms, losses, defined, objective, episode, teacher
    write_json(directory / "manifest.json", {"step": batch["step"], "tasks": records, "entries": entries})
    return entries, records, values


def evaluate_training(model, criterion, records):
    values = {}
    for record in records:
        cached = torch.load(record["artifact"], map_location="cpu", weights_only=False)
        episode = alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
        torch.manual_seed(record["gradient_seed"])
        with torch.no_grad(), compute_context(episode.source.device):
            prediction = streamed_point_predictions(model, criterion, episode.source, episode.mask, **TRAIN_OPTIONS)
        terms = point_metric_terms(prediction, episode)
        metrics = alignment.metric_summary(prediction, episode, terms)
        for name, metric in metrics.items():
            values[f'{record["name"]}/{name}'] = None if metric is None else metric["value"]
        del cached, episode, prediction, terms
    return values


def train(output):
    config = verify_registration(output)
    wait_for_gpu(output, True, False)
    reference = read(alignment.fixed.PRODUCER / "reference.json")
    baseline = read(alignment.fixed.PRODUCER / "validation_0.json")["student"]
    bank = torch.load(alignment.fixed.PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    score = alignment.fixed.selection_decision(baseline, baseline, reference["references"], config["base"])["initialization_score"]
    best, accepted, stop = {"step": 0, "score": score}, 0, []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda received, frame: stop.append(received))
    with alignment.mixture.runtime(config["base"], output):
        model, optimizer, criterion, teachers, sampler = alignment.mixture.backend.setup(config["base"])
        inventory = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
        if inventory != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json"):
            raise ValueError("Require the unchanged original single-network architecture")
        write_json(output / "parameter_inventory.json", inventory)
        best = validate(output, model, optimizer, criterion, 0, accepted, best, bank, reference, baseline, config)
        checkpoint(output, model, optimizer, 0, accepted, best, reference)
        gradients = torch.empty((42, sum(row[2] for row in inventory)), device="cuda", dtype=torch.float32)
        for batch in config["batches"]:
            step, started = batch["step"], time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            directory = output / "training_batches" / f"step_{step:03d}"
            directory.mkdir(parents=True)
            entries, tasks, before_metrics = capture_batch(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients)
            rate = learning_rate(step, 100, 1e-5, 5)
            write_json(output / "status.json", {"state": "protecting_update", "active_step": step, "accepted_updates": accepted,
                       "pid": os.getpid(), "updated_unix": time.time()})
            try:
                record, snapshot = guarded.protected_update(model, optimizer, gradients[:len(entries)],
                    [task["weight"] for task in batch["tasks"]], rate, before_metrics,
                    lambda current: evaluate_training(current, criterion, tasks))
            except guarded.StepEvaluationError as error:
                save_tensor({"snapshot": error.snapshot, "record": error.record, "gradients": gradients[:len(entries)].cpu(),
                             "entries": entries, "batch": batch, "task_manifest": str(directory / "manifest.json")},
                            output / "failed_update.pt")
                raise
            accepted += int(record["accepted"])
            record.update(step=step, accepted_updates=accepted, task_manifest=str(directory / "manifest.json"),
                          task_manifest_sha256=sha256(directory / "manifest.json"), seconds=time.monotonic() - started,
                          peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
            if step == STEPS or stop:
                endpoint = {"snapshot": snapshot, "record": record, "gradients": gradients[:len(entries)].cpu(),
                            "entries": entries, "batch": batch, "task_manifest": str(directory / "manifest.json"),
                            "config_sha256": sha256(output / "config.json")}
                record["raw_endpoint_sha256"] = save_tensor(endpoint, output / "raw_final_update.pt")
            append_json(output / "training.jsonl", record)
            print(json.dumps({"step": step, "accepted_updates": accepted, "accepted": record["accepted"],
                              "reason": record["reason"], "seconds": record["seconds"], "best": best}), flush=True)
            del snapshot
            if step in BOUNDARIES:
                best = validate(output, model, optimizer, criterion, step, accepted, best, bank, reference, baseline, config)
            if step % 5 == 0 or stop:
                checkpoint(output, model, optimizer, step, accepted, best, reference)
            if stop:
                break
        del gradients
        gc.collect()
    verify_registration(output)
    write_json(output / "status.json", {"state": "interrupted" if stop else "training_complete", "step": step,
               "accepted_updates": accepted, "best": best, "qualification_pending": True,
               "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(output / "status.json")["state"] != "registered":
            raise ValueError("Never restart a started trial")
        try:
            write_json(output / "status.json", {"state": "starting", "pid": os.getpid(), "updated_unix": time.time()})
            train(output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
