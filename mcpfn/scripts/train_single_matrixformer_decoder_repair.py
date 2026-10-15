"""Fifty fresh decoder-only mixed-prior attempts with the original V8 retention baseline."""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import torch

from audit_single_matrixformer_structured_training import close_values
from dwip_teacher_mean_training import verify_hashes
import preflight_single_model_decoder_training as preflight
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_decoder_training as decoder
from streamed_large_matrix_training import learning_rate, rng_state
import train_single_matrixformer_finite_guarded as original
from train_single_matrixformer_column_aligned import repeated_bank
from train_streamed_large_matrices import append_json, wait_for_gpu


ROOT = original.ROOT
OUTPUT = ROOT / "decoder_repair_v14"
PARENT = preflight.PARENT
STEPS, BOUNDARIES = 50, (0, 10, 25, 50)
DATA_SEED, GRADIENT_SEED = 201100901, 202100901
CONFIRMATION_SEEDS = (211100901, 212100901)
POLICY = ("One ordinary network initialized from the unqualified but fully audited V13 final training state, "
          "not its one-step decoder preflight. Freeze all encoder/attention parameters and buffers; train only the "
          "original complete decoder. Fifty fresh fourteen-task batches preserve V13's shape/component schedule, "
          ".25 legacy/.25 structured/.5 ratings mixture, original V8 parent/specialist teachers, losses and normalization. "
          "Recompute all composite and defined-point gradients and fresh fused-Adam moments in decoder coordinates; "
          "same first-fifty original100-step learning rates,0.1 protection margin,2x actual-norm bound and strict "
          "finite training-point acceptance/rollback. Rejections move to the next fresh batch. No inference ensemble.")
QUALIFICATION = ("Independently audit all fifty attempted batches and accepted counts, cached data, scheduled full "
                 "development predictions, unchanged22-guard V8-baseline selection, bitwise-frozen V13 backbone, "
                 "and full final restricted-gradient/optimizer endpoint. Replay that endpoint independently and "
                 "any selected full model on all152 development cases without reselection. Select only a trained "
                 "strict minimax improvement retaining every original V8 guard; earlier boundary wins ties. "
                 "The unqualified V13 initialization never becomes a best alias. Confirm the sole selected model "
                 "once on284 fresh reserved cases with the original gate, still anchored to V8, then independently "
                 "audit all1136 confirmation grids before publishing any alias. Only afterward run all frozen "
                 "tabular, recommender and LLM campaigns and every required comparison. No guard relaxation or "
                 "fallback. Earlier gradient certificates are logs, not independent full-vector trajectory replays.")


def read(path):
    return json.loads(Path(path).read_text())


def status(output, state, **fields):
    write_json(output / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def batches():
    result = deepcopy(original.original.training_batches()[:STEPS])
    for batch in result:
        for task in batch["tasks"]:
            offset = (batch["step"] - 1) * 1400 + task["index"] * 100
            task["gradient_seed"] = GRADIENT_SEED + offset
            task["specification"].get("structured", task["specification"])["seed"] = DATA_SEED + offset
    return result


def confirmation_cases():
    return json.loads(json.dumps(repeated_bank(CONFIRMATION_SEEDS[0], confirmation=True)
                                 + original.alignment.mixture.structured_bank(CONFIRMATION_SEEDS[1], confirmation=True)))


def verify_registration(output):
    config = read(output / "config.json")
    if (sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["qualification"] != QUALIFICATION
            or config["batches"] != batches() or config["boundaries"] != list(BOUNDARIES) or config["steps"] != STEPS
            or config["confirmation_cases"] != confirmation_cases() or config["parent_checkpoint"] != str(PARENT)
            or config["retention_baseline"] != str(original.alignment.fixed.PARENT)
            or config["base"] != read(original.alignment.fixed.PRODUCER / "config.json")
            or config["guarded_policy"] != original.guarded.POLICY or config["margin"] != .1
            or config["learning_rate_horizon"] != 100 or config["mutable_prefix"] != "decoder."):
        raise ValueError("Never change the registered decoder continuation or its original V8 qualification baseline")
    original.alignment.fixed.validate_recipe(config["base"])
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def register(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Use the fresh declared decoder-repair trial")
    if shutil.disk_usage(ROOT).free < 25 * 1024**3:
        raise OSError("Reserve all fifty task batches, decoder endpoint and18 GiB free space")
    preflight.verify(preflight.OUTPUT)
    audit, gate = read(preflight.OUTPUT / "audit.json"), read(preflight.OUTPUT / "gate.json")
    if (read(preflight.OUTPUT / "status.json")["state"] != "complete" or not audit["complete"]
            or not gate["passed"] or not gate["preflight_only"] or audit["training_tasks"] != 14
            or audit["prediction_grids"] != 28 or not audit["frozen"]["backbone_and_buffers_bitwise_unchanged"]
            or not audit["raw_endpoint"]["accepted"] or audit["raw_endpoint"]["gradient_vectors"] != 42):
        raise ValueError("Require the complete successful actual decoder optimizer preflight, not only a counterfactual")
    verify_hashes(audit["artifact_sha256"])
    previous = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(original.alignment.fixed.seeds_in(read(path)) for path in previous))
    fresh = original.alignment.fixed.seeds_in([batches(), confirmation_cases()])
    if len(fresh) != 1684 or fresh.intersection(used):
        raise ValueError("Reserve all1400 new data/backward seeds and284 independent confirmation seeds")
    base = read(original.alignment.fixed.PRODUCER / "config.json")
    paths = [PARENT, original.OUTPUT / "validation_50.json", ROOT / "finite_guarded_v13_audit/audit.json",
             original.alignment.fixed.PARENT, original.alignment.fixed.PRODUCER / "config.json",
             original.alignment.fixed.PRODUCER / "synthetic_bank.pt", original.alignment.fixed.PRODUCER / "reference.json",
             original.alignment.fixed.PRODUCER / "validation_0.json", preflight.OUTPUT / "config.json",
             preflight.OUTPUT / "audit.json", preflight.OUTPUT / "gate.json", ROOT / "domain_gradient_probe_v1/parameter_inventory.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_decoder_repair.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "qualification": QUALIFICATION, "base": base, "model": base["model"],
              "parent_checkpoint": str(PARENT), "parent_selection_qualified": False,
              "retention_baseline": str(original.alignment.fixed.PARENT), "mutable_prefix": "decoder.",
              "steps": STEPS, "batches": batches(), "boundaries": list(BOUNDARIES),
              "confirmation_cases": confirmation_cases(), "confirmation_seeds": list(CONFIRMATION_SEEDS),
              "learning_rate_horizon": 100, "margin": .1, "guarded_policy": original.guarded.POLICY,
              "best_step_zero_semantics": "No eligible trained boundary; best score sentinel is the frozen V8 reference score, never promotion of the V13 training input",
              "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in previous}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "training_started": False, "confirmation_evaluated": False, "real_benchmarks_evaluated": False})
    status(output, "registered")


def setup(config):
    model, old_optimizer, criterion, teachers, sampler = original.alignment.mixture.backend.setup(config["base"])
    del old_optimizer
    parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(parent["model"], strict=True)
    model.train()
    view = decoder.DecoderParameterView(model)
    optimizer = torch.optim.AdamW([{"params": list(view.parameters()), "lr": config["base"]["peak_lr"]}],
        lr=config["base"]["peak_lr"], weight_decay=config["base"]["weight_decay"], betas=(.9, .999), eps=1e-8, fused=True)
    return model, view, optimizer, criterion, teachers, sampler, parent["model"]


def selected_boundary(best, step, accepted, decision):
    return original.original.selected_boundary(best, step, decision) if step > 0 and accepted > 0 else dict(best)


def checkpoint(output, model, optimizer, step, accepted, best, reference, selected=False):
    config = read(output / "config.json")
    payload = {"model": model.state_dict(), "step": step, "epoch": step, "attempted_batches": step,
               "gradient_updates_performed": accepted, "parent_training_updates": 46,
               "parent_checkpoint_sha256": config["input_sha256"][str(PARENT)],
               "config_sha256": sha256(output / "config.json"), "bank_sha256": reference["bank_sha256"],
               "best_step": best["step"], "best_score": best["score"], "preflight_only": False,
               "independent_confirmation_passed": False, "decoder_only_training": True,
               "retention_baseline_sha256": config["input_sha256"][str(original.alignment.fixed.PARENT)]}
    if not selected:
        payload.update(optimizer=optimizer.state_dict(), rng=rng_state())
    return original.save_tensor(payload, output / ("selected_checkpoint.pth" if selected else "latest_checkpoint.pth"))


def validate(output, model, optimizer, step, accepted, best, criterion, bank, reference, baseline, parent_state, config):
    directory = output / f"validation_{step:03d}"
    directory.mkdir()
    with original.alignment.mixture.runtime(config["base"], directory):
        metrics, cases = original.alignment.mixture.backend.evaluate_bank(model, criterion, bank)
    if len(cases) != 152:
        raise ValueError("Every original development case remains mandatory")
    if step == 0:
        close_values(metrics, read(original.OUTPUT / "validation_50.json")["student"])
    frozen = decoder.frozen_state_equal(parent_state, model.state_dict(), {name for name, parameter in model.named_parameters() if name.startswith("decoder.")})
    decision = original.alignment.fixed.selection_decision(metrics, baseline, reference["references"], config["base"])
    chosen = selected_boundary(best, step, accepted, decision)
    digest = checkpoint(output, model, optimizer, step, accepted, chosen, reference, selected=True) if chosen != best else None
    row = {"step": step, "accepted_updates": accepted, "student": metrics, "cases": cases, "decision": decision,
           "frozen": frozen, "best_step": chosen["step"], "selected_checkpoint_sha256": digest}
    write_json(output / f"validation_{step}.json", row)
    append_json(output / "validation.jsonl", {key: value for key, value in row.items() if key != "cases"})
    print(json.dumps({"validation": step, "accepted_updates": accepted, "score": decision["score"],
                      "failed_guards": sum(not row["retained"] for row in decision["comparisons"]), "best_step": chosen["step"]}), flush=True)
    return chosen


def train(output):
    config = verify_registration(output)
    wait_for_gpu(output, True, False)
    reference = read(original.alignment.fixed.PRODUCER / "reference.json")
    baseline = read(original.alignment.fixed.PRODUCER / "validation_0.json")["student"]
    bank = torch.load(original.alignment.fixed.PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    anchor = original.alignment.fixed.selection_decision(baseline, baseline, reference["references"], config["base"])["score"]
    best, accepted, stop = {"step": 0, "score": anchor}, 0, []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda received, frame: stop.append(received))
    with original.alignment.mixture.runtime(config["base"], output):
        model, view, optimizer, criterion, teachers, sampler, parent_state = setup(config)
        inventory = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
        selected = [[name, list(parameter.shape), parameter.numel()] for name, parameter in view.named_parameters()]
        if inventory != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json") or selected != [row for row in inventory if row[0].startswith("decoder.")]:
            raise ValueError("The original network and exact decoder coordinate subset are mandatory")
        write_json(output / "parameter_inventory.json", inventory)
        write_json(output / "decoder_inventory.json", selected)
        best = validate(output, model, optimizer, 0, accepted, best, criterion, bank, reference, baseline, parent_state, config)
        checkpoint(output, model, optimizer, 0, accepted, best, reference)
        gradients = torch.empty((42, sum(row[2] for row in selected)), device="cuda", dtype=torch.float32)
        for batch in config["batches"]:
            step, started = batch["step"], time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            directory = output / "training_batches" / f"step_{step:03d}"
            directory.mkdir(parents=True)
            with decoder.decoder_capture_coordinates(model, view):
                entries, tasks, before = original.capture_batch(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients)
            rate = learning_rate(step, 100, 1e-5, 5)
            status(output, "protecting_decoder_update", active_step=step, accepted_updates=accepted)
            try:
                record, snapshot = original.guarded.protected_update(view, optimizer, gradients[:len(entries)],
                    [task["weight"] for task in batch["tasks"]], rate, before,
                    lambda current: original.evaluate_training(model, criterion, tasks))
            except original.guarded.StepEvaluationError as error:
                original.save_tensor({"snapshot": error.snapshot, "record": error.record, "gradients": gradients[:len(entries)].cpu(),
                    "entries": entries, "batch": batch, "task_manifest": str(directory / "manifest.json")}, output / "failed_update.pt")
                raise
            accepted += int(record["accepted"])
            record.update(step=step, accepted_updates=accepted, task_manifest=str(directory / "manifest.json"),
                          task_manifest_sha256=sha256(directory / "manifest.json"), seconds=time.monotonic() - started,
                          peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
            if step == STEPS or stop:
                payload = {"snapshot": snapshot, "record": record, "gradients": gradients[:len(entries)].cpu(),
                           "entries": entries, "batch": batch, "task_manifest": str(directory / "manifest.json"),
                           "config_sha256": sha256(output / "config.json")}
                record["raw_endpoint_sha256"] = original.save_tensor(payload, output / "raw_final_update.pt")
            append_json(output / "training.jsonl", record)
            print(json.dumps({"step": step, "accepted_updates": accepted, "accepted": record["accepted"],
                              "reason": record["reason"], "seconds": record["seconds"], "best": best}), flush=True)
            del snapshot
            if step in BOUNDARIES:
                best = validate(output, model, optimizer, step, accepted, best, criterion, bank, reference, baseline, parent_state, config)
                checkpoint(output, model, optimizer, step, accepted, best, reference)
            elif stop:
                checkpoint(output, model, optimizer, step, accepted, best, reference)
            if stop:
                break
    verify_registration(output)
    status(output, "interrupted" if stop else "training_complete", step=step, accepted_updates=accepted,
           best=best, qualification_pending=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        register(output)
    else:
        with (output / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(output / "status.json")["state"] != "registered":
                raise ValueError("Never restart a registered decoder trial")
            try:
                train(output)
            except BaseException as error:
                status(output, "failed", error=repr(error))
                raise
