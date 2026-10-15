"""One hundred targeted-prior decoder attempts retaining the original V8 baseline."""

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
import preflight_single_model_targeted_decoder as preflight
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_decoder_training as decoder
from streamed_large_matrix_training import learning_rate, rng_state
import train_single_matrixformer_finite_guarded as original
from train_single_matrixformer_column_aligned import repeated_bank
from train_streamed_large_matrices import append_json, wait_for_gpu


ROOT = original.ROOT
OUTPUT = ROOT / "targeted_decoder_v15"
INITIAL_RUN = ROOT / "decoder_repair_v14"
PARENT = preflight.PARENT
STEPS, BOUNDARIES = 100, (0, 10, 25, 50, 100)
DATA_SEED, GRADIENT_SEED = 241100901, 242100901
CONFIRMATION_SEEDS = (251100901, 252100901)
COMPONENT_CYCLES = {
    "mar": ("low_rank_mcar",) * 4 + ("slow_tail_mcar",) * 2 + ("steep_tail_mcar", "anchor_latent_mcar"),
    "mnar_skip_logic": ("steep_tail_mcar",) * 4 + ("anchor_latent_mcar",) * 2 + ("low_rank_mcar", "slow_tail_mcar"),
}
POLICY = ("One ordinary network initialized from the fully audited unqualified V14 final state, never its targeted "
          "one-step preflight. Freeze all original nondecoder parameters and buffers. One hundred fresh fourteen-task "
          "attempts retain .25 legacy/.25 structured/.5 ratings domain mass and every family. Allocate structured "
          "composite weights .10 MAR/.10 skip logic/.005 for each other family. On an eight-step component cycle, "
          "MAR uses low-rank4/slow-tail2/steep-tail1/anchor1; skip logic uses steep-tail4/anchor2/low-rank1/slow-tail1. "
          "Cross every emphasized component with both240x80 and1024x256 requests, both legacy/area-scaled regimes "
          "and both.2/.6 missingness rates within64 attempts. Other tasks preserve the original100-step shape/component "
          "schedule, V8 teachers, original losses and normalization. Fresh decoder Adam uses the original100-step "
          "1e-5 peak/warmup5 schedule,0.1 joint composite/point protection,2x norm bound and strict finite-point "
          "parameter/moment/preceding-rate rollback. Rejections advance to a fresh batch. No inference ensemble.")
QUALIFICATION = ("Independently audit all100 attempted batches, accepted counts and1400 cached tasks, every scheduled "
                 "development grid, unchanged22-guard V8-baseline decisions, and bitwise-frozen V14 nondecoder state. "
                 "Verify the full final decoder-gradient/optimizer endpoint and independently reconstruct/replay it. "
                 "Only a trained strict minimax improvement retaining all original V8 guards at0/10/25/50/100 may be "
                 "selected; step0 and zero accepted updates cannot qualify and earlier ties win. Independently replay "
                 "the sole selected152-case development boundary without reselection, then confirm it once on284 "
                 "reserved fresh cases against V8, never V14, with the unchanged gate. Audit all1136 confirmation "
                 "grids before publishing an alias or running the unchanged complete frozen real campaign/review. "
                 "No fallback, guard relaxation or hidden benchmark regression. Earlier gradient certificates remain "
                 "logged evidence, not independent full-vector trajectory replays.")


def read(path):
    return json.loads(Path(path).read_text())


def status(output, state, **fields):
    write_json(output / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def batches():
    result = deepcopy(original.original.training_batches()[:STEPS])
    for batch in result:
        for task in batch["tasks"]:
            if "structured" in task["specification"]:
                task["weight"] = .1 if task["name"] in COMPONENT_CYCLES else .005
            if task["name"] in COMPONENT_CYCLES:
                cycle, position = divmod(batch["step"] - 1, 8)
                task["specification"]["structured"].update(
                    component=COMPONENT_CYCLES[task["name"]][position],
                    shape=list(((240, 80), (1024, 256))[(cycle + position) % 2]),
                    regime=("area_scaled", "legacy")[(cycle // 2 + position) % 2],
                    requested_missingness=(.6, .2)[(cycle // 4 + position) % 2])
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
    if shutil.disk_usage(ROOT).free < 29 * 1024**3:
        raise OSError("Reserve all100 task batches, decoder endpoint, validation and18 GiB free space")
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
    if len(fresh) != 3084 or fresh.intersection(used):
        raise ValueError("Reserve all2800 new data/backward seeds and284 independent confirmation seeds")
    base = read(original.alignment.fixed.PRODUCER / "config.json")
    paths = [PARENT, INITIAL_RUN / "validation_50.json", ROOT / "decoder_repair_v14_audit/audit.json",
             original.alignment.fixed.PARENT, original.alignment.fixed.PRODUCER / "config.json",
             original.alignment.fixed.PRODUCER / "synthetic_bank.pt", original.alignment.fixed.PRODUCER / "reference.json",
             original.alignment.fixed.PRODUCER / "validation_0.json", preflight.OUTPUT / "config.json",
             preflight.OUTPUT / "audit.json", preflight.OUTPUT / "gate.json", ROOT / "domain_gradient_probe_v1/parameter_inventory.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_targeted_decoder_training.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "qualification": QUALIFICATION, "base": base, "model": base["model"],
              "parent_checkpoint": str(PARENT), "parent_selection_qualified": False,
              "retention_baseline": str(original.alignment.fixed.PARENT), "mutable_prefix": "decoder.",
              "steps": STEPS, "batches": batches(), "boundaries": list(BOUNDARIES),
              "confirmation_cases": confirmation_cases(), "confirmation_seeds": list(CONFIRMATION_SEEDS),
              "learning_rate_horizon": 100, "margin": .1, "guarded_policy": original.guarded.POLICY,
              "best_step_zero_semantics": "No eligible trained boundary; best score sentinel is the frozen V8 reference score, never promotion of the V14 training input",
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
               "gradient_updates_performed": accepted, "parent_training_updates": 95,
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
        close_values(metrics, read(INITIAL_RUN / "validation_50.json")["student"])
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
