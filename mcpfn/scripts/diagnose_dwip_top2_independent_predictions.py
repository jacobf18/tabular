"""One preregistered independent-synthetic check of the existing fixed pair."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from diagnose_dwip_matched_objective_gradients import require_completed_stages
from diagnose_dwip_top2_fixed_predictions import compare_fixed_predictions, top_two
from run_dwip_budgeted_queue_v1 import immutable_json, RESERVE_BYTES
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO


EXPERIMENT = "dwip_top2_independent_complementarity_20260917"
FIXED = "dwip_top2_fixed_complementarity_20260917"
INVENTORY = "dwip_top2_fixed_parameter_average_20260917/final_saved_model_inventory.json"
PROVENANCE = "dwip_spectral_anchor_mix25_huber1_continuous_64k_20260917/continuous_full_comparison.json"
GENERAL = "rowcol_output_remedies_paired_20260907/independent_validation.pth"
PANELS = ("independent_spectral", "independent_general")
SOURCES = (
    "scripts/diagnose_dwip_top2_independent_predictions.py",
    "scripts/diagnose_dwip_top2_fixed_predictions.py",
    "scripts/diagnose_dwip_matched_objective_gradients.py",
    "scripts/run_dwip_budgeted_queue_v1.py",
    "scripts/run_dwip_continuous_spectrum.py",
    "scripts/run_dwip_transfer_evaluation.py",
    "scripts/run_rowcol_controlled_experiments.py",
)


def read_json(path):
    return json.loads(path.read_text())


def unchanged_pair(inventory, registered):
    members = top_two(inventory)
    fields = ("queue", "sha256", "fixed_macro_rmse", "case_digest")
    if len(registered) != 2 or any(
            any(member[field] != original[field] for field in fields)
            for member, original in zip(members, registered)):
        raise ValueError("Require the exact previously registered fixed pair, not a new selection")
    return members


def independent_comparison(panel, cases, first, second):
    if panel not in PANELS:
        raise ValueError("Only the two registered independent synthetic panels are permitted")
    comparison = compare_fixed_predictions(cases, first, second)
    comparison["non_monitor_group_count"] = comparison.pop("selector_group_count")
    comparison["non_monitor_groups_mean_beats_both_raw"] = comparison.pop("selector_groups_mean_beats_both_raw")
    comparison.update(
        panel=panel, case_count=len(cases),
        scope="Independent-synthetic cached prediction diagnostic, not a selector, parameter average or deployed model. Existing holdouts and constituent outcomes were previously observed; this is not a fresh blinded holdout.",
    )
    return comparison


def evidence_decision(panels):
    if set(panels) != set(PANELS):
        raise ValueError("Require both complete preregistered synthetic panels")
    return {
        "supports_studying_single_model_compression": all(
            panels[name]["equal_prediction_mean_beats_both_raw"] for name in PANELS),
        "checkpoint_created_or_promoted": False,
        "ensemble_deployment_proposed": False,
        "rule": "Both predeclared independent macro RMSEs must strictly beat both constituents before formulating a single-model compression study. This diagnostic cannot select or promote a checkpoint.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    queues = REPO / "runs/objective_queues"
    root = queues / EXPERIMENT
    root.mkdir(exist_ok=True)
    if (root / "results.json").exists():
        raise ValueError("Diagnostic already complete; preserve existing evidence")
    if shutil.disk_usage(root).free < RESERVE_BYTES + 32 * 1024**2:
        raise RuntimeError("Preserve the20GiB disk reserve plus32MiB diagnostic allowance")
    inventory_path, fixed_path = queues / INVENTORY, queues / FIXED / "prespecification.json"
    provenance_path = queues / PROVENANCE
    inventory, fixed, provenance = (read_json(path) for path in (inventory_path, fixed_path, provenance_path))
    members = unchanged_pair(inventory["inventory"], fixed["members"])
    inputs = {}

    def verify(path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Registered input changed: {path}")
        inputs[str(path)] = actual
        return actual

    for path in (inventory_path, fixed_path, provenance_path, *(REPO / source for source in SOURCES)):
        verify(path)
    for path, expected in inventory["input_sha256"].items():
        verify(path, expected)
    for path, expected in fixed["input_sha256"].items():
        verify(path, expected)
    panels = {name: {"case_paths": [], "predictions": {}} for name in PANELS}
    for label, member in zip(("rank1", "rank2"), members):
        queue = queues / member["queue"]
        require_completed_stages(queue)
        verify(member["checkpoint"], member["sha256"])
        for panel, case_path, prediction_name in (
                (PANELS[0], queue / "holdout.pth", "dwip1e5_selected_predictions.pth"),
                (PANELS[1], queues / GENERAL, "dwip1e5_general_predictions.pth")):
            prediction_path = queue / prediction_name
            for path in (case_path, prediction_path):
                if path == queues / GENERAL:
                    verify(path)
                else:
                    verify(path, provenance["input_sha256"][str(path)])
            panels[panel]["case_paths"].append(str(case_path))
            panels[panel]["predictions"][label] = str(prediction_path)
    registration = {
        "members": members, "panels": panels, "input_sha256": inputs,
        "case_count": {PANELS[0]: 72, PANELS[1]: 64},
        "expected_case_digest": {
            PANELS[0]: "7230a76121a5892186d46740106f8a51d6a803770c2d67eed57cbb964135e68a",
            PANELS[1]: "2d851df2aed755fe28fcc90cbb432ff873e657ee8ffe1d38798811339154dbf2",
        },
        "general_case_provenance": "The shared general cache is newly file-hashed here, not inherited from the full-comparison input map. Its entire case-content digest is frozen before evaluation and all scored targets/indices must exactly match both existing prediction caches.",
        "arithmetic": "One equal FP64 prediction mean rounded to FP32. Reproduce both raw macro/group RMSEs at1e-12. Preserve existing scored targets, indices and monitor-only exclusions.",
        "decision_rule": evidence_decision({name: {"equal_prediction_mean_beats_both_raw": False} for name in PANELS})["rule"],
        "known_outcomes": "Raw synthetic and public benchmark outcomes and the fixed-prediction gain are already known. Independent prediction mixtures have not been computed before registration. No blinding claim; no benchmark input affects ranking or arithmetic.",
        "limits": "CPU-only, two frozen raw models, two predeclared synthetic populations. No new pair/weight, parameter average, retention/public benchmark evaluation, optimizer, training, checkpoint creation/promotion or ensemble deployment.",
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "threads": 2},
    }
    immutable_json(root / "prespecification.json", registration)
    if args.register_only:
        immutable_json(root / "registration_status.json", {
            "state": "registered", "evaluation_started": False,
            "utc": datetime.now(timezone.utc).isoformat(),
            "prespecification_sha256": sha256(root / "prespecification.json"),
        })
        print(json.dumps({"registered": str(root), "members": [item["queue"] for item in members]}, indent=2))
        return
    registration_status = read_json(root / "registration_status.json")
    if registration_status["prespecification_sha256"] != sha256(root / "prespecification.json"):
        raise ValueError("Require the unchanged separate registration before evaluation")
    torch.set_num_threads(2)
    comparisons = {}
    for panel, paths in panels.items():
        populations = [torch.load(path, map_location="cpu", weights_only=True)["cases"] for path in paths["case_paths"]]
        if any(len(cases) != registration["case_count"][panel] or
               case_digest(cases) != registration["expected_case_digest"][panel] for cases in populations):
            raise ValueError(f"Registered independent case population differs: {panel}")
        if registration["expected_case_digest"][panel] == members[0]["case_digest"]:
            raise ValueError("Independent population is the fixed selection population")
        payloads = {label: torch.load(path, map_location="cpu", weights_only=True)
                    for label, path in paths["predictions"].items()}
        comparisons[panel] = independent_comparison(panel, populations[0], payloads["rank1"], payloads["rank2"])
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise ValueError(f"Input changed during the read-only diagnostic: {path}")
    result = {"utc": datetime.now(timezone.utc).isoformat(), "panels": comparisons,
              "decision": evidence_decision(comparisons), "prespecification_sha256": sha256(root / "prespecification.json")}
    immutable_json(root / "results.json", result)
    immutable_json(root / "status.json", {"state": "complete", "gpu_evaluation_started": False,
                   "checkpoint_created_or_promoted": False, "updated_utc": result["utc"]})
    print(json.dumps({"summary": {name: {key: value for key, value in panel.items() if key != "groups"}
                                 for name, panel in comparisons.items()}, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
