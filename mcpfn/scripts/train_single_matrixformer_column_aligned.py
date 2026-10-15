"""Register and train a prospective column-aligned single-network continuation."""

import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import sys
import time

import torch

import diagnose_single_model_confirmation_nrmse as diagnostic
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_column_objective as column
from train_single_matrixformer import storage_guard
import train_single_matrixformer_parent_teacher as parent
from train_streamed_large_matrices import wait_for_gpu


balanced, continuation = parent.balanced, parent.continuation


def evidence(directory, parent_recipe):
    config, summary = balanced.read(directory / "config.json"), balanced.read(directory / "summary.json")
    diagnostic.full.verify_config(config, directory)
    if (balanced.read(directory / "status.json")["state"] != "complete" or not summary["complete"]
            or summary["cases"] != 30 or summary["pooled_parity_checks"] != 120 or not summary["pooled_parity_passed"]
            or not summary["model_state_unchanged"] or config["run"] != str(parent_recipe)
            or summary["config_sha256"] != sha256(directory / "config.json")
            or any(summary[field] for field in ("training_performed", "checkpoint_selection_performed", "gate_changed"))
            or any(config[field] for field in ("training_performed", "checkpoint_selection_performed", "real_benchmark_data_used", "gate_changed"))):
        raise ValueError("Require the completed immutable V5 full-context metric diagnostic")
    cases = [balanced.read(directory / "cases" / f"{index:03d}.json") for index in range(30)]
    if any(case["index"] != index for index, case in enumerate(cases)):
        raise ValueError("Require every original diagnostic case index")
    rows = [row for row in diagnostic.full.cpu.summarize(cases) if row["family"] == "ALL"]
    if rows != summary["summary"]:
        raise ValueError("Diagnostic aggregates do not reproduce")
    roles = {row["role"]: row for row in rows}
    if not (roles["candidate"]["pooled_rmse"] > roles["initialization"]["pooled_rmse"]
            and all(roles["candidate"][name] < roles["initialization"][name]
                    for name in ("missbench_nrmse", "uci_nrmse"))):
        raise ValueError("Require the verified full-context metric-order reversal")
    audit = balanced.read(directory / "saved_prediction_audit.json")
    if (not audit["complete"] or audit["prediction_cases_audited"] != 30 or not audit["all_saved_scores_reproduce"]
            or audit["cuda_initialized"] or audit["original_gate_changed"]
            or audit["config_sha256"] != sha256(directory / "config.json")
            or audit["summary_sha256"] != sha256(directory / "summary.json")):
        raise ValueError("Require the completed independent saved-array audit")
    artifacts = [directory / name for name in ("config.json", "registration.json", "summary.json", "saved_prediction_audit.json")]
    artifacts.extend(directory / "cases" / f"{index:03d}.json" for index in range(30))
    return {"directory": str(directory), "roles": roles,
            "interpretation": "Reused synthetic evidence motivates prospective normalized point supervision; no retroactive gate changes or benchmark-derived selection",
            "artifacts_sha256": {str(path): sha256(path) for path in artifacts}}


def repeated_bank(seed, *, confirmation=False):
    return [case for replica in range(2)
            for case in continuation.bank_specs(seed + 1000 * replica, confirmation=confirmation)]


def make_config(parent_recipe, metric_diagnostic, after_campaign):
    base = balanced.read(parent_recipe / "config.json")
    continuation.verify_run(base, parent_recipe)
    if (base["version"] != 5 or base["training_teacher_policy"] != parent.TEACHER_POLICY
            or balanced.read(parent_recipe / "status.json")["state"] != "training_complete"):
        raise ValueError("Require the completed V5 frozen-parent recipe")
    report = evidence(metric_diagnostic, parent_recipe)
    predecessor = after_campaign / "cache_launcher_config.json"
    if balanced.read(predecessor)["run"] != str(parent_recipe) or not balanced.campaign_ready(after_campaign):
        raise ValueError("Require the completed unchanged V5 campaign")
    config = deepcopy(base)
    config.update(version=6, seed=36100501, data_seed=37100501, validation_seed=38100501,
                  confirmation_seed=39100501, preflight_seed=40100501,
                  validation_cases=repeated_bank(38100501), confirmation_cases=repeated_bank(39100501, confirmation=True),
                  column_objective=deepcopy(column.COLUMN_OBJECTIVE), column_metric_evidence=report,
                  primary_metrics=[list(pair) for pair in column.PRIMARY_METRICS],
                  retention_metrics=[list(pair) for pair in column.RETENTION_METRICS],
                  bank_replicates=2, parent_teacher_recipe=str(parent_recipe), after_campaign=str(after_campaign),
                  objective="Original V5 objectives plus weight-1 frozen-parent quadratic upper bound on the mean of two truth-defined column NRMSE risks (std >=1e-3 and >=1e-4), anchored at parent query RMSE with 1e-3 denominator floor; rating loss unchanged",
                  selection="Minimize worst normalized MissBench/UCI NRMSE and rating RMSE/MAE ratio to original frozen references; require all four plus pooled imputation RMSE and fixed-common-column NRMSE <= initialization; step zero eligible",
                  confirmation="92 fresh full cases; all six retention metrics <= initialization and general control, and strict four-metric minimax improvement versus initialization",
                  experiment="Prospective metric-aligned sibling from the same confirmed V2; synthetic-only normalized point supervision, two fresh replicas of each original bank design, stricter metric retention; unchanged priors, teacher identities, model, dropout, optimizer and 400-update schedule",
                  execution="Process-local adapters add the registered point objective and column scoring; restore callbacks on errors; independent confirmation uses the new registered six-metric gate; all historical gates unchanged")
    config["phases"][0]["name"] = "column_aligned"
    old = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]}
    new = [case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]]
    if (len(new) != 148 or len(set(new)) != 148 or old.intersection(new)
            or config["data_seed"] + 16 * config["steps"] + 1 >= min(new)):
        raise ValueError("Require disjoint fresh 56-case selection and 92-case confirmation banks")
    config["input_sha256"].update({str(parent_recipe / "config.json"): sha256(parent_recipe / "config.json"),
                                   str(predecessor): sha256(predecessor), **report["artifacts_sha256"]})
    sources = {Path(__file__).resolve(), REPO / "scripts/confirm_single_matrixformer_column_aligned.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    sources.update(Path(path) for path in diagnostic.metric.SCORERS.values())
    config["source_sha256"].update({str(path): sha256(path) for path in sources})
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def validate_recipe(config):
    if (config["version"] != 6 or config["column_objective"] != column.COLUMN_OBJECTIVE
            or config["training_teacher_policy"] != parent.TEACHER_POLICY
            or config["probability_loss_scales"] != balanced.PROBABILITY_SCALES
            or config["bank_replicates"] != 2
            or config["primary_metrics"] != [list(pair) for pair in column.PRIMARY_METRICS]
            or config["retention_metrics"] != [list(pair) for pair in column.RETENTION_METRICS]
            or len(config["validation_cases"]) != 56 or len(config["confirmation_cases"]) != 92):
        raise ValueError("Require the registered prospective column-aligned recipe")


def train(config, output, *, resume=False):
    validate_recipe(config)
    with parent.training_teachers(), column.evaluation_metrics(), column.training_objective():
        continuation.train(config, output, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        if (any(value is None for value in (arguments.parent_recipe, arguments.diagnostic, arguments.after_campaign))
                or arguments.wait or arguments.resume):
            parser.error("Registration requires the V5 recipe, metric diagnostic and predecessor; no wait/resume")
        balanced.register(make_config(arguments.parent_recipe.resolve(), arguments.diagnostic.resolve(),
                                       arguments.after_campaign.resolve()), output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = balanced.read(output / "status.json")
        if (status["state"] != "registered" and not arguments.resume) or status["state"] == "training_complete":
            raise ValueError("Never overwrite started training or resume completed training")
        try:
            config = balanced.read(output / "config.json")
            continuation.verify_run(config, output)
            validate_recipe(config)
            if not balanced.campaign_ready(Path(config["after_campaign"])):
                raise RuntimeError("Predecessor campaign must be terminal")
            wait_for_gpu(output, arguments.wait, False)
            storage_guard(output)
            if not arguments.resume:
                write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                                          "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
                write_json(output / "status.json", {"state": "preflight", "pid": os.getpid(), "updated_unix": time.time()})
                with parent.training_teachers(), column.training_objective():
                    balanced.preflight(config, output)
            train(config, output, resume=arguments.resume)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                                "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
