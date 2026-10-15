"""Compare V26/V27 on both consumed synthetic banks without training or qualification."""

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_csv, write_json
import single_model_fixed_contraction as fixed
import single_model_structured_mixture as mixture
import relocate_single_model_idle_diagnostics_to_scratch as storage
from train_streamed_large_matrices import wait_for_gpu


ROOT = fixed.ROOT
OUTPUT = ROOT / "v26_v27_consumed_bank_comparison_v1"
SCRATCH = storage.SCRATCH / "v26_v27_consumed_bank_comparison_v1"
TRIALS = {
    "v26": {"run": "top_two_blocks_v26", "lifecycle": "top_two_blocks_v26_identity_recovery", "step": 64,
            "updates": 64, "grids": 1216,
            "checkpoint_sha256": "5096341d7ed637848b4eb8807a81c96777849fac22e67d3213a1b9ae3cce242a",
            "result_sha256": "79a78a9123684afc52593b9a0a98a4f3bac2c58943acbc81be15944a9b713543"},
    "v27": {"run": "general_imputation_teacher_v27", "lifecycle": "general_imputation_teacher_v27_pipeline", "step": 10,
            "updates": 63, "grids": 1368,
            "checkpoint_sha256": "ef775cef8dee57338132f1756ad39042995950976d5df15c18445b205b99002d",
            "result_sha256": "ba9c32f648f06787788e723a6c80069d7af5ecfdf157d7abfff553f440a89270"},
}
METRICS = ("rmse", "mae", "missbench_nrmse", "uci_nrmse", "common_nrmse")
POLICY = ("Post-hoc matched two-checkpoint/two-consumed-bank diagnostic only. Reuse the independently audited "
    "V26-on-V26 and V27-on-V27 predictions; infer only V26-on-V27 and V27-on-V26, all284 cases per cell,568 new "
    "complete grids. Independently CPU-rescore all1136 grids in the four cells, retaining every case, full "
    "observed/query prediction, original normalization, canonical metric, truth-defined coverage and reference. "
    "Both banks remain consumed and diagnosis-only, never future training or fresh confirmation. Predict each "
    "cell with one ordinary checkpoint; no inference ensemble, training, bank regeneration, retry, checkpoint "
    "publication, candidate reselection, new teacher or changed qualification. Comparing the two checkpoints "
    "isolates their net continuation differences on identical inputs, not the teacher-policy change alone. "
    "Keep configurations/audits on checkpoint storage and new prediction directories on UUID-bound local "
    "scratch, not provider-persistent storage or backup. Preserve18/6/50 GiB reserves plus64/16/1536 MiB "
    "checkpoint/root/scratch write budgets. No real benchmark or claim of goal success.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def cells():
    return [{"bank": bank, "model": model, "name": f"{model}_on_{bank}", "reuse": bank == model}
            for bank in TRIALS for model in TRIALS]


def require_capacity():
    storage.require_mount()
    for path, reserve, budget in ((ROOT, 18, 64), (Path("/"), 6, 16), (storage.SCRATCH, 50, 1536)):
        if shutil.disk_usage(path).free - budget * 1024**2 < reserve * 1024**3:
            raise OSError("Preserve original reserves and declared diagnostic write budgets")


def require_stopped(directory, states):
    current = read(directory / "status.json")
    if current["state"] not in states or Path(f'/proc/{current["pid"]}').exists():
        raise ValueError("Use only completed, stopped original scientific owners")


def validate_history(trial, result, audit, replay, confirmation, manifest):
    if (not result["complete"] or result["gate"]["passed"] is not False
            or result["campaign"] is not None or result["review"] is not None
            or not audit["complete"] or not audit["complete_training_audit"] or audit["through_step"] != 64
            or audit["attempted_batches"] != 64 or audit["accepted_updates"] != trial["updates"]
            or audit["best"]["step"] != trial["step"] or audit["prediction_grids"] != trial["grids"]
            or audit["independently_regenerated_training_tasks"] != 1792
            or audit["independently_regenerated_rating_contexts"] != 128
            or audit.get("top_two_block_training") is not True
            or not replay["complete"] or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 152
            or not confirmation["complete"] or confirmation["cases"] != 284 or confirmation["prediction_grids"] != 1136
            or not result["gate"]["confirmation_evaluated"] or manifest["selected_step"] != trial["step"]
            or manifest["checkpoint_sha256"] != trial["checkpoint_sha256"]):
        raise ValueError("Require both complete audited but unqualified original lifecycles")


def history():
    evidence, banks, bases = {}, {}, []
    for label, trial in TRIALS.items():
        run, lifecycle = ROOT / trial["run"], ROOT / trial["lifecycle"]
        confirmation = ROOT / (trial["run"] + "_confirmation")
        replay = ROOT / (trial["run"] + "_endpoint_replay")
        audit_path = ROOT / (trial["run"] + "_audit") / "audit.json"
        for directory, states in ((run, {"training_complete"}), (lifecycle, {"complete"}),
                (confirmation, {"complete"}), (replay, {"complete"}),
                (audit_path.parent.with_name(audit_path.parent.name + "_task_replay"), {"complete"})):
            require_stopped(directory, states)
            evidence[str(directory / "status.json")] = sha256(directory / "status.json")
        result, audit = read(lifecycle / "result.json"), read(audit_path)
        replay_audit, confirmation_audit = read(replay / "audit.json"), read(confirmation / "audit.json")
        manifest, gate, config = read(confirmation / "manifest.json"), read(confirmation / "gate.json"), read(run / "config.json")
        validate_history(trial, result, audit, replay_audit, confirmation_audit, manifest)
        if (sha256(lifecycle / "result.json") != trial["result_sha256"] or result["gate"] != gate
                or gate["audit_sha256"] != sha256(confirmation / "audit.json")
                or gate["training_audit_sha256"] != sha256(audit_path)
                or gate["replay_audit_sha256"] != sha256(replay / "audit.json")
                or manifest["cases"] != config["confirmation_cases"]
                or manifest["retention_baseline"] != str(fixed.PARENT)
                or manifest["checkpoint"] != str(run / "selected_checkpoint.pth")):
            raise ValueError("Preserve exact closed results, original references, bank order and sole selections")
        if ((run / "best_validation_checkpoint.pth").exists()
                or (ROOT / (trial["run"] + "_benchmarks")).exists() or (ROOT / (trial["run"] + "_review")).exists()):
            raise ValueError("Never reinterpret a diagnostic as a published or qualified model")
        verify_hashes(config["source_sha256"])
        verify_hashes(confirmation_audit["artifact_sha256"])
        paths = [lifecycle / "result.json", audit_path, run / "config.json", run / "selected_checkpoint.pth",
                 replay / "audit.json", replay / "gate.json"]
        paths += [confirmation / name for name in ("config.json", "registration.json", "audit.json", "gate.json", "manifest.json",
                                                    "synthetic_bank.pt", "reference.json", "initialization.json", "student.json")]
        evidence.update({str(path): sha256(path) for path in paths})
        evidence.update(confirmation_audit["artifact_sha256"])
        if evidence[str(run / "selected_checkpoint.pth")] != trial["checkpoint_sha256"]:
            raise ValueError("Only the exact original selected checkpoint may be diagnosed")
        banks[label] = {"directory": str(confirmation), "bank": str(confirmation / "synthetic_bank.pt"),
                       "bank_sha256": gate["bank_sha256"], "cases": manifest["cases"],
                       "checkpoint": manifest["checkpoint"], "checkpoint_sha256": trial["checkpoint_sha256"]}
        bases.append(config["base"])
    if bases[0] != bases[1]:
        raise ValueError("Require the identical original inference and normalization recipe")
    evidence.update(bases[0]["input_sha256"])
    inventory = ROOT / "domain_gradient_probe_v1/parameter_inventory.json"
    evidence[str(inventory)] = sha256(inventory)
    verify_hashes(evidence)
    return banks, bases[0], evidence


def register():
    if OUTPUT.exists() or OUTPUT.is_symlink() or SCRATCH.exists() or SCRATCH.is_symlink():
        raise FileExistsError("Never overwrite or repeat a consumed-bank diagnostic")
    require_capacity()
    banks, base, evidence = history()
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_consumed_bank_comparison.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "trials": TRIALS, "cells": cells(), "banks": banks, "base": base,
              "scratch": str(SCRATCH), "scratch_is_provider_persistent": False,
              "input_sha256": evidence, "source_sha256": {str(path): sha256(path) for path in sources}}
    SCRATCH.mkdir()
    OUTPUT.mkdir()
    for cell in cells():
        if not cell["reuse"]:
            destination = SCRATCH / cell["name"]
            destination.mkdir()
            (OUTPUT / cell["name"]).symlink_to(destination, target_is_directory=True)
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
               "registered_unix": time.time(), "training_performed": False, "neural_evaluation_started": False,
               "confirmation_consumed_here": False, "both_banks_already_consumed": True})
    status("registered")


def verify():
    config = read(OUTPUT / "config.json")
    if (read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["trials"] != TRIALS or config["cells"] != cells()
            or config["scratch"] != str(SCRATCH) or config["scratch_is_provider_persistent"] is not False):
        raise ValueError("Preserve the exact registered diagnostic scope and all original inputs")
    storage.require_mount()
    if (OUTPUT.is_symlink() or SCRATCH.is_symlink() or not OUTPUT.is_dir() or not SCRATCH.is_dir()
            or OUTPUT.stat().st_dev != ROOT.stat().st_dev or SCRATCH.stat().st_dev != storage.SCRATCH.stat().st_dev
            or OUTPUT.stat().st_dev == SCRATCH.stat().st_dev):
        raise ValueError("Keep canonical metadata and local scratch on their original separate filesystems")
    for cell in cells():
        if not cell["reuse"]:
            path, target = OUTPUT / cell["name"], SCRATCH / cell["name"]
            if not path.is_symlink() or path.readlink() != target or path.resolve() != target or not target.is_dir():
                raise ValueError("Preserve exact declared scratch targets and canonical prediction paths")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def validate_bank(bank, specifications):
    if len(bank) != len(specifications) or len(bank) != 284:
        raise ValueError("Every one of the284 original cases is required")
    for cached, specification in zip(bank, specifications):
        metadata = cached["metadata"]
        if (metadata["seed"] != specification.get("structured", specification)["seed"]
                or metadata["domain"] != specification["domain"]
                or bool(metadata.get("structured", False)) != ("structured" in specification)):
            raise ValueError("Never reorder, regenerate or replace any consumed case")


def validate_report(report, bank, directory, new):
    if len(report["cases"]) != len(bank) or len(bank) != 284:
        raise ValueError("Every full prediction grid and case metric must be retained")
    for index, (record, cached) in enumerate(zip(report["cases"], bank)):
        metadata = {key: value for key, value in record.items() if key not in ("student", "prediction_file", "prediction_sha256")}
        if metadata != {"index": index, **cached["metadata"]}:
            raise ValueError("Match the entire original case metadata, not just a displayed label")
        path = Path(record["prediction_file"])
        if (not path.resolve().is_relative_to(directory.resolve())
                or (new and path != directory / "student_predictions_000" / f"case_{index:03d}.npz")):
            raise ValueError("Use only original audited or exactly declared new prediction paths")
    if mixture.aggregate(report["cases"], "student") != report["student"]:
        raise ValueError("Keep the original complete-case aggregation")


@torch.no_grad()
def evaluate():
    config = verify()
    if read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart a begun neural diagnostic")
    require_capacity()
    wait_for_gpu(OUTPUT, True, False)
    status("evaluating")
    for cell in cells():
        if cell["reuse"]:
            continue
        directory = OUTPUT / cell["name"]
        if any(directory.iterdir()):
            raise FileExistsError("Never overwrite or resume a started diagnostic cell")
        bank_spec = config["banks"][cell["bank"]]
        model_spec = config["banks"][cell["model"]]
        bank = torch.load(bank_spec["bank"], map_location="cpu", weights_only=True)
        validate_bank(bank, bank_spec["cases"])
        status("evaluating_cell", cell=cell["name"])
        checkpoint = torch.load(model_spec["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        if checkpoint["step"] != TRIALS[cell["model"]]["step"] or sha256(model_spec["checkpoint"]) != model_spec["checkpoint_sha256"]:
            raise ValueError("Never diagnose a later unselected checkpoint")
        with mixture.runtime(config["base"], directory):
            model, optimizer, criterion, teachers, sampler = mixture.backend.setup(config["base"])
            del optimizer, teachers, sampler
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval().requires_grad_(False)
            inventory = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
            if inventory != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json") or sum(row[2] for row in inventory) != 77998496:
                raise ValueError("Preserve the ordinary original single-network architecture")
            metrics, records = mixture.backend.evaluate_bank(model, criterion, bank)
            result = {"cell": cell, "bank_sha256": bank_spec["bank_sha256"], "checkpoint_sha256": model_spec["checkpoint_sha256"],
                      "student": metrics, "cases": records, "single_checkpoint": True, "training_performed": False,
                      "fresh_confirmation": False, "selection_performed": False}
            validate_report(result, bank, directory, True)
            write_json(directory / "result.json", result)
            del model, criterion
        del checkpoint, bank
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"cell": cell["name"], "complete_grids": len(records)}), flush=True)
    verify()
    status("awaiting_independent_cpu_audit", new_grids=568)


def comparisons(bank_label, reports, original):
    values = {model: reports[model]["student"] for model in TRIALS}
    values.update({"original_V8": original["initialization"]["student"], "general": original["reference"]["general"],
                   "blend_or_specialist": original["reference"]["teacher"]})
    summary, paired = [], []
    groups = [("legacy", ("imputation",)), ("recsys", ("recsys",)), ("structured_overall", ("structured", "overall"))]
    groups += [(family, ("structured", "families", family)) for family in mixture.FAMILIES]
    for group, path in groups:
        selected = {}
        for role, metrics in values.items():
            for part in path:
                metrics = metrics[part]
            selected[role] = metrics
        for metric in (("rmse", "mae") if group == "recsys" else METRICS):
            scores = {role: metrics[metric] for role, metrics in selected.items()}
            summary.append({"bank": bank_label, "group": group, "metric": metric, **scores,
                            "v27_minus_v26": scores["v27"] - scores["v26"],
                            "relative_change": None if scores["v26"] == 0 else scores["v27"] / scores["v26"] - 1})
    for index in range(284):
        first, second = reports["v26"]["cases"][index], reports["v27"]["cases"][index]
        metadata = {key: value for key, value in first.items() if key not in ("student", "prediction_file", "prediction_sha256")}
        other = {key: value for key, value in second.items() if key not in ("student", "prediction_file", "prediction_sha256")}
        if metadata != other:
            raise ValueError("Paired comparisons require identical full case metadata")
        initial, reference = original["initialization"]["cases"][index], original["reference"]["cases"][index]
        scores = {"v26": first["student"], "v27": second["student"], "original_V8": initial["student"],
                  "general": reference["general"], "blend_or_specialist": reference["teacher"]}
        row = {**metadata, "bank": bank_label, **scores,
               "v27_minus_v26": {metric: None if first["student"].get(metric) is None or second["student"].get(metric) is None
                                  else second["student"][metric] - first["student"][metric] for metric in METRICS}}
        mixture.require_truth_coverage([row], tuple(scores))
        paired.append(row)
    return summary, paired


def audit():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Independently rescore every full grid with CPU-only access")
    config = verify()
    require_stopped(OUTPUT, {"awaiting_independent_cpu_audit"})
    if (OUTPUT / "audit.json").exists():
        raise FileExistsError("Never replace a completed independent diagnostic audit")
    require_capacity()
    status("auditing")
    hashes, summary, paired = {}, [], []
    grids = 0
    for bank_label, bank_spec in config["banks"].items():
        bank = torch.load(bank_spec["bank"], map_location="cpu", weights_only=True)
        validate_bank(bank, bank_spec["cases"])
        original_directory = Path(bank_spec["directory"])
        original = {name: read(original_directory / f"{name}.json") for name in ("reference", "initialization")}
        reports = {}
        for model_label in TRIALS:
            reused = model_label == bank_label
            directory = original_directory if reused else OUTPUT / f"{model_label}_on_{bank_label}"
            path = directory / ("student.json" if reused else "result.json")
            report = read(path)
            if not reused and (report["bank_sha256"] != bank_spec["bank_sha256"]
                    or report["checkpoint_sha256"] != config["banks"][model_label]["checkpoint_sha256"]
                    or report["cell"] != next(cell for cell in cells() if cell["name"] == directory.name)
                    or report["single_checkpoint"] is not True or report["training_performed"] is not False
                    or report["fresh_confirmation"] is not False or report["selection_performed"] is not False):
                raise ValueError("Preserve exact inference-only cell provenance")
            validate_report(report, bank, directory, not reused)
            count, recorded = fixed.raw_audit(bank, [(report, ("student",))])
            if count != 284:
                raise ValueError("Independently score every full observed/query grid")
            grids += count
            hashes.update(recorded)
            hashes[str(path)] = sha256(path)
            reports[model_label] = report
        rows, cases = comparisons(bank_label, reports, original)
        summary.extend(rows)
        paired.extend(cases)
        del bank
    if grids != 1136 or len(paired) != 568 or len(summary) != 144:
        raise ValueError("Retain the complete four-cell, two-bank comparison scope")
    verify_hashes(hashes)
    verify()
    write_csv(OUTPUT / "comparisons.csv", summary)
    write_json(OUTPUT / "cases.json", paired)
    result = {"complete": True, "config_sha256": sha256(OUTPUT / "config.json"), "prediction_grids": grids,
              "new_grids": 568, "reused_grids": 568, "paired_cases": len(paired), "summary_comparisons": len(summary),
              "cuda_initialized": False, "bank_regenerated": False, "training_performed": False,
              "selection_performed": False, "qualification_changed": False, "new_teacher_or_retention_reference": False,
              "teacher_change_alone_causally_isolated": False, "goal_success_not_assessed": True,
              "artifact_sha256": {**hashes, str(OUTPUT / "comparisons.csv"): sha256(OUTPUT / "comparisons.csv"),
                                  str(OUTPUT / "cases.json"): sha256(OUTPUT / "cases.json")}}
    write_json(OUTPUT / "audit.json", result)
    lines = ["# V26/V27 matched consumed-bank diagnostic", "", POLICY, "",
             "All four cells retain284 cases;568 new grids and568 reused grids are independently CPU-scored.",
             "No checkpoint is trained, selected or qualified here; both banks remain consumed.", "",
             "| Consumed bank | Metric | V26 | V27 | V27 minus V26 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summary:
        if row["group"] in {"legacy", "structured_overall", "recsys"} and row["metric"] in {"rmse", "common_nrmse", "mae"}:
            lines.append(f'| {row["bank"]} | {row["group"]}/{row["metric"]} | {row["v26"]:.9f} | {row["v27"]:.9f} | {row["v27_minus_v26"]:+.9f} |')
    lines.extend(["", "All144 aggregate comparisons and568 paired cases, including regressions, are retained in the CSV/JSON.",
                  "This measures the net continuation difference on identical inputs, not a causal teacher-only effect."])
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    status("complete", new_grids=568, audited_grids=grids, qualification_changed=False)
    print(json.dumps({key: value for key, value in result.items() if key != "artifact_sha256"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "evaluate", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register()
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"evaluate": evaluate, "audit": audit}[arguments.stage]()
            except BaseException as error:
                status("failed", error=repr(error))
                raise
