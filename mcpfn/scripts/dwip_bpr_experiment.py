"""Synthetic-only prior/optimization experiments with frozen DWIP inference."""
import argparse
import fcntl
import json
import math
import resource
from pathlib import Path
import shutil
import time
from dataclasses import replace

import numpy as np
import torch

from dwip_bpr_evaluation import CAMPAIGN, INITIAL, INITIAL_SHA256, MF_BASELINE, ROOT, compare_goal, evaluate_public, selection_key, split_hashes
from dwip_recsys_experiment import compute_context, training_episode, update_ema, candidate_parameters, save_training_checkpoint
from dwip_recsys_provenance import POLICY, TRAINING_DATA, require_synthetic_checkpoint, require_synthetic_episode
from dwip_bpr_optimization import configure_trainable_parameters, protected_optimizer_step
from dwip_bpr_window_training import recomputed_window_logits, windowed_training_logits
from evaluate_dwip_recsys import sha256, write_json
from tabimpute.interface import ImputePFN
from tabimpute.prior.recsys_ranking_objective import raw_query_predictions, synthetic_ranking_objective


def make_imputer(checkpoint):
    require_synthetic_checkpoint(checkpoint)
    config = json.loads((INITIAL.parent / "config.json").read_text())["model"]
    return ImputePFN(device="cuda", checkpoint_path=str(checkpoint), entry_wise_features=False,
                     json_config={"config": config}, model_family="new_stable", prediction_stat="mean",
                     precision_mode="amp_bfloat16", max_num_rows=2000, max_num_cols=1000, max_num_chunks=1, verbose=False)


def state_signature(model):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in model.state_dict().items()}


def registered_baselines(root):
    registration = json.loads((root / "baseline_registration.json").read_text())
    if registration["split_sha256"] != split_hashes() or registration["mf_sha256"] != sha256(MF_BASELINE):
        raise ValueError("Frozen split or MF reference changed")
    if registration["bpr_sha256"] != sha256(root / "bpr_baseline.json"):
        raise ValueError("Frozen BPR reference changed")
    return json.loads(MF_BASELINE.read_text()), json.loads((root / "bpr_baseline.json").read_text()), registration


@torch.no_grad()
def training_validation(imputer, work, step, args):
    output = work / f"validation_details_{step}" if getattr(args, "save_validation_details", False) else None
    return evaluate_public(imputer, output=output)


def episode_arguments(args):
    defaults = json.loads((INITIAL.parent / "config.json").read_text())["arguments"]
    defaults.update({key: value for key, value in vars(args).items()})
    defaults["selection_objective"] = "rating_guarded_bpr"
    anchor_mode = defaults.get("teacher_anchor_mode", "prediction_mse")
    if anchor_mode not in ("prediction_mse", "macro_risk_budget"):
        raise ValueError("Unknown synthetic-teacher anchor mode")
    if anchor_mode == "macro_risk_budget":
        weight = defaults.get("teacher_anchor_weight", 0.)
        mae_weight = defaults["mae_weight"]
        if not math.isfinite(weight) or weight <= 0 or defaults["point_objective"] != "expected_user_rmse":
            raise ValueError("Risk-budget anchoring requires positive finite teacher weight and expected-user-RMSE rating risk")
        if not math.isfinite(mae_weight) or mae_weight <= 0:
            raise ValueError("Risk-budget anchoring requires positive finite MAE weight")
    affinity_range = (defaults.get("exposure_affinity_low", 0.0), defaults.get("exposure_affinity_high", 1.0))
    if not all(math.isfinite(value) for value in affinity_range) or not 0 <= affinity_range[0] <= affinity_range[1]:
        raise ValueError("Exposure-affinity range must be finite, nonnegative and ordered")
    quality_range = (defaults.get("exposure_quality_low", 0.0), defaults.get("exposure_quality_high", 1.5))
    if not all(math.isfinite(value) for value in quality_range) or not 0 <= quality_range[0] <= quality_range[1]:
        raise ValueError("Exposure-quality range must be finite, nonnegative and ordered")
    popularity_range = (defaults.get("item_popularity_low", 1.4), defaults.get("item_popularity_high", 2.8))
    if not all(math.isfinite(value) for value in popularity_range) or not 0 <= popularity_range[0] <= popularity_range[1]:
        raise ValueError("Item-popularity range must be finite, nonnegative and ordered")
    candidate_scope = defaults.get("ranking_candidate_scope", "all")
    if candidate_scope not in ("all", "topk_union"):
        raise ValueError("Unknown top-k candidate scope")
    if candidate_scope != "all" and defaults.get("ranking_objective") not in ("topk_expected_pairwise", "topk_soft_choice", "topk_finite_gap"):
        raise ValueError("Candidate restriction requires the top-k expected pairwise objective")
    if defaults.get("ranking_objective") == "topk_finite_gap":
        if (defaults.get("ranking_relevance_target") != "visible_item_frequency" or candidate_scope != "topk_union"
                or defaults.get("negative_only_ranking") or defaults.get("center_negative_only_ranking")):
            raise ValueError("Finite-gap ranking requires synthetic visible frequencies and ordinary two-sided top-k union")
        margin = defaults.get("ranking_finite_gap_margin", .01)
        top_k = defaults.get("ranking_top_k", 10)
        if (not math.isfinite(margin) or margin <= 0 or isinstance(top_k, bool) or not isinstance(top_k, int)
                or top_k < 1 or top_k * margin > 4):
            raise ValueError("Require positive finite and bounded-rating-feasible top-k margin")
    if defaults.get("ranking_objective") == "topk_soft_choice":
        if (defaults.get("ranking_relevance_target") != "visible_item_frequency" or candidate_scope != "topk_union"
                or defaults.get("negative_only_ranking") or defaults.get("center_negative_only_ranking")):
            raise ValueError("Soft choice requires synthetic visible frequencies and ordinary two-sided top-k union")
        pseudocount = defaults.get("ranking_frequency_pseudocount", .5)
        if not math.isfinite(pseudocount) or pseudocount <= 0:
            raise ValueError("Require positive finite synthetic frequency pseudo-count")
    if defaults.get("center_negative_only_ranking") and (not defaults.get("negative_only_ranking")
                                                        or defaults.get("ranking_objective") != "topk_expected_pairwise"):
        raise ValueError("Gradient centering requires one-sided top-k ranking")
    if defaults.get("training_rows") is not None:
        if defaults["training_rows"] < 2:
            raise ValueError("Require at least two synthetic training rows")
        defaults["rows"] = defaults["training_rows"]
    if defaults.get("recompute_window_gradients") and defaults.get("project_rating_gradient"):
        raise ValueError("Window replay and protected-update projection cannot be combined")
    if defaults.get("offload_window_activations") and not defaults.get("recompute_window_gradients"):
        raise ValueError("Activation offload requires window-gradient replay")
    if defaults.get("project_rating_gradient") and defaults["rating_weight"] == 0 and defaults["teacher_anchor_weight"] == 0:
        raise ValueError("Projection requires a nonzero synthetic rating objective")
    return argparse.Namespace(**defaults)


def ranking_episode(args, step):
    selected = args
    if not 0 <= args.catalog_fraction <= 1:
        raise ValueError("Catalog prior mixture must lie in [0,1]")
    if args.catalog_fraction and np.random.default_rng(args.seed * 1000000 + step).random() < args.catalog_fraction:
        selected = argparse.Namespace(**{**vars(args), "context_ordering": "first_observation_catalog"})
    windows = getattr(args, "training_item_windows", 1)
    if windows < 1:
        raise ValueError("Require a positive synthetic item-window count")
    paired = getattr(args, "paired_context_windows", False)
    if paired and (windows != 2 or args.context_ordering != "first_observation"):
        raise ValueError("Paired contexts require two windows and a first-observation base prior")
    if windows > 1:
        if args.ranking_query_mode != "user_catalog" or args.negatives_per_user < args.cols * windows:
            raise ValueError("Joint windows require a complete synthetic user catalog")
        if args.cols * windows > args.parent_cols:
            raise ValueError("Synthetic parent must cover all joint item windows")
        selected = argparse.Namespace(**{**vars(selected), "cols": args.cols * windows})
        if paired and selected.context_ordering == "first_observation":
            selected = argparse.Namespace(**{**vars(selected), "context_ordering": "first_observation_paired",
                                             "paired_window_cols": args.cols})
        episode = training_episode(selected, step, preserve_user_item_axes=True)
        episode.metadata.update({"training_item_windows": windows, "training_window_cols": args.cols})
        return episode
    return training_episode(selected, step)


def objective(logits, episode, criterion, args, teacher_prediction=None):
    return synthetic_ranking_objective(logits, episode, criterion, ranking_weight=args.ranking_weight,
                                       rating_weight=args.rating_weight, hard_negatives=args.hard_negatives,
                                       temperature=args.temperature, nll_weight=args.nll_weight, mae_weight=args.mae_weight,
                                       point_objective=args.point_objective, macro_query_scope=args.macro_query_scope,
                                       teacher_prediction=teacher_prediction, teacher_anchor_weight=args.teacher_anchor_weight,
                                       ranking_objective=args.ranking_objective, negative_only_ranking=args.negative_only_ranking,
                                       return_components=args.project_rating_gradient,
                                       ranking_boundary_weight=args.ranking_boundary_weight, ranking_top_k=args.ranking_top_k,
                                       center_negative_only_ranking=args.center_negative_only_ranking,
                                       ranking_candidate_scope=getattr(args, "ranking_candidate_scope", "all"),
                                       ranking_frequency_pseudocount=getattr(args, "ranking_frequency_pseudocount", .5),
                                       teacher_anchor_mode=getattr(args, "teacher_anchor_mode", "prediction_mse"),
                                       ranking_finite_gap_margin=getattr(args, "ranking_finite_gap_margin", .01))


def student_training_logits(model, episode, args, window_cols):
    if getattr(args, "recompute_window_gradients", False):
        return recomputed_window_logits(model, episode.source, episode.query, args.activation_checkpointing, window_cols,
                                        offload_activations=getattr(args, "offload_window_activations", False))
    return windowed_training_logits(model, episode.source, episode.query, args.activation_checkpointing, window_cols), None


@torch.no_grad()
def synthetic_teacher_predictions(teacher, episode, window_cols=0):
    require_synthetic_episode(episode)
    teacher.model.eval()
    with compute_context():
        logits = windowed_training_logits(teacher.model, episode.source, episode.rating_query, window_cols=window_cols)
    prediction, _, _, _ = raw_query_predictions(logits, replace(episode, query=episode.rating_query), teacher.bar_distribution)
    return prediction.detach()


def preflight(args):
    args = episode_arguments(args)
    if args.output is None or args.output.exists():
        raise ValueError("Preflight requires a new output file")
    torch.manual_seed(args.seed)
    imputer = make_imputer(args.checkpoint)
    teacher = make_imputer(args.teacher_checkpoint) if args.teacher_anchor_weight else None
    if teacher is not None:
        teacher.model.requires_grad_(False)
    model = imputer.model
    optimization_scope = configure_trainable_parameters(model, getattr(args, "trainable_scope", "full"))
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.02, fused=True) if args.project_rating_gradient else None
    original_signature = state_signature(model)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    fractions = (0., 1.) if 0 < args.catalog_fraction < 1 else (args.catalog_fraction,)
    variants = []
    for fraction in fractions:
        selected_args = argparse.Namespace(**{**vars(args), "catalog_fraction": fraction})
        model.zero_grad(set_to_none=True)
        episode = ranking_episode(selected_args, 20)
        require_synthetic_episode(episode)
        window_cols = args.cols if args.training_item_windows > 1 else 0
        target = synthetic_teacher_predictions(teacher, episode, window_cols) if teacher is not None else None
        with compute_context():
            logits, replay = student_training_logits(model, episode, args, window_cols)
        result = objective(logits, episode, imputer.bar_distribution, args, target)
        loss, diagnostics = result[:2]
        protected = torch.autograd.grad(result[2]["protected"], parameters, retain_graph=True, allow_unused=True) if args.project_rating_gradient else None
        loss.backward()
        if replay is not None:
            diagnostics["verified_gradient_window_replays"] = replay.backward(logits.grad, compute_context)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        if args.project_rating_gradient:
            diagnostics.update(protected_optimizer_step(optimizer, parameters, protected))
        variants.append({"finite_loss": bool(torch.isfinite(loss)), "gradient_norm": norm.item(), "loss": loss.item(),
                         "episode": episode.metadata, "diagnostics": diagnostics})
        del logits, loss, target, episode, result, protected, replay
    teacher_frozen = teacher is None or all(value.grad is None and not value.requires_grad for value in teacher.model.parameters())
    frozen_gradients_absent = all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    passed = all(value["finite_loss"] for value in variants) and teacher_frozen and frozen_gradients_absent and state_signature(model) == original_signature and peak < 128 and host_peak < 160
    report = {"passed": passed, "synthetic_only": True, "real_data_loaded": False,
              "architecture_unchanged": state_signature(model) == original_signature, "teacher_frozen": teacher_frozen,
              "optimization_scope": optimization_scope,
              "frozen_parameter_gradients_absent": frozen_gradients_absent,
              "peak_allocated_gib": peak, "peak_host_rss_gib": host_peak, **variants[-1], "variants": variants,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    write_json(args.output, report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise ValueError("Synthetic training preflight failed")


def train(args):
    args = episode_arguments(args)
    provenance = require_synthetic_checkpoint(args.checkpoint)
    if sha256(INITIAL) != INITIAL_SHA256:
        raise ValueError("Registered synthetic starting checkpoint changed")
    if not 0 <= args.ema_decay < 1 or args.steps < 1 or args.validate_every < 1:
        raise ValueError("Invalid optimization schedule")
    mf, bpr, registration = registered_baselines(args.root)
    work = args.root / args.name
    work.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    imputer = make_imputer(args.checkpoint)
    model, criterion = imputer.model, imputer.bar_distribution
    optimization_scope = configure_trainable_parameters(model, getattr(args, "trainable_scope", "full"))
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    teacher_provenance = require_synthetic_checkpoint(args.teacher_checkpoint) if args.teacher_anchor_weight else None
    teacher = make_imputer(args.teacher_checkpoint) if args.teacher_anchor_weight else None
    if teacher is not None:
        teacher.model.requires_grad_(False)
    signature = state_signature(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.02, fused=True)
    averaged = {name: value.detach().clone() for name, value in model.state_dict().items()} if args.ema_decay else None
    sources = [Path(__file__), Path(__file__).with_name("dwip_bpr_evaluation.py"),
               Path(__file__).with_name("dwip_recsys_experiment.py"), Path(__file__).with_name("dwip_recsys_provenance.py"),
               Path(__file__).with_name("evaluate_dwip_recsys_candidate.py"), Path(__file__).with_name("dwip_recsys_ranking_audit.py"),
               Path(__file__).with_name("dwip_bpr_ranking_profile.py"),
               Path(__file__).with_name("dwip_bpr_optimization.py"),
               Path(__file__).with_name("dwip_bpr_window_training.py"),
               ROOT / "src/tabimpute/prior/recsys_rating_episode.py", ROOT / "src/tabimpute/prior/recsys_ranking_objective.py",
               ROOT / "src/tabimpute/prior/recsys_soft_choice_objective.py",
               ROOT / "src/tabimpute/prior/recsys_finite_gap_objective.py",
               ROOT / "src/tabimpute/prior/recsys_teacher_risk_budget.py",
               ROOT / "src/tabimpute/prior/recsys_clustered_factors.py", ROOT / "src/tabimpute/model/model_new_stable.py",
               ROOT / "src/tabimpute/interface.py", ROOT / "src/tabimpute/model/bar_distribution.py"]
    manifest = {"model": json.loads((INITIAL.parent / "config.json").read_text())["model"],
                "architecture": "Unchanged DWIP v2-2", "state_signature": signature,
                "optimization_scope": optimization_scope,
                "initial_checkpoint": str(args.checkpoint.resolve()), "initial_sha256": provenance["checkpoint_sha256"],
                "initial_provenance": provenance, "training_data": TRAINING_DATA, "training_policy": POLICY,
                "teacher": {"checkpoint": str(args.teacher_checkpoint.resolve()), "provenance": teacher_provenance,
                            "data": "Synthetic episodes only", "frozen": True} if teacher is not None else None,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "source_sha256": {str(source.resolve()): sha256(source) for source in sources},
                "baseline_registration": registration, "normalization": "two_way_user_scale",
                "inference": {"rows": 2000, "cols": 1000, "statistic": "mean", "precision": "amp_bfloat16"},
                "selection": "Inner only: strict MF rating guardrails, then worst BPR NDCG/Recall ratio, then mean ratio",
                "outer_evaluated": False}
    write_json(work / "config.json", manifest)
    snapshot = work / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copyfile(source, snapshot / source.name)
    started = time.monotonic()
    initial = training_validation(imputer, work, 0, args)
    torch.cuda.empty_cache()
    best_score = selection_key(initial, mf["validation"], bpr["validation"])
    write_json(work / "validation_0.json", {"step": 0, "score": best_score, "improved": True, "datasets": initial})
    save_training_checkpoint(work / "best_validation_checkpoint.pth", {"model": model.state_dict(), "step": 0}, args.storage_reserve_gib)
    print(json.dumps({"step": 0, "score": best_score}), flush=True)
    with (work / "training.jsonl").open("x") as log:
        for step in range(1, args.steps + 1):
            model.train()
            warmup = min(1.0, step / min(50, max(1, args.steps // 10)))
            decay = .1 + .9 * (1 + math.cos(math.pi * step / args.steps)) / 2
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup * decay
            episode = ranking_episode(args, step)
            require_synthetic_episode(episode)
            if not episode.rating_query.any():
                raise ValueError("Synthetic episode has no rating queries")
            optimizer.zero_grad(set_to_none=True)
            window_cols = args.cols if args.training_item_windows > 1 else 0
            teacher_prediction = synthetic_teacher_predictions(teacher, episode, window_cols) if teacher is not None else None
            with compute_context():
                logits, replay = student_training_logits(model, episode, args, window_cols)
            result = objective(logits, episode, criterion, args, teacher_prediction)
            loss, diagnostics = result[:2]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite synthetic objective")
            protected = torch.autograd.grad(result[2]["protected"], parameters, retain_graph=True, allow_unused=True) if args.project_rating_gradient else None
            loss.backward()
            if replay is not None:
                diagnostics["verified_gradient_window_replays"] = replay.backward(logits.grad, compute_context)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            if args.project_rating_gradient:
                diagnostics.update(protected_optimizer_step(optimizer, parameters, protected))
            else:
                optimizer.step()
            if averaged is not None:
                update_ema(averaged, model.state_dict(), args.ema_decay)
            if step == 1 or step % 10 == 0 or diagnostics.get("protected_step_rolled_back", False):
                entry = {"step": step, "loss": loss.item(), **diagnostics, **episode.metadata,
                         "grad_norm": grad_norm.item(), "seconds": time.monotonic() - started}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps(entry), flush=True)
            del logits, loss, episode, teacher_prediction, result, protected, replay
            if step % args.validate_every == 0 or step == args.steps:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                with candidate_parameters(model, averaged):
                    if state_signature(model) != signature:
                        raise ValueError("Architecture/state signature changed")
                    results = training_validation(imputer, work, step, args)
                    score = selection_key(results, mf["validation"], bpr["validation"])
                    improved = score < best_score
                    if improved:
                        best_score = score
                        save_training_checkpoint(work / "best_validation_checkpoint.pth", {"model": model.state_dict(), "step": step}, args.storage_reserve_gib)
                write_json(work / f"validation_{step}.json", {"step": step, "score": score, "improved": improved,
                           "datasets": results, "comparisons": compare_goal(results, mf["validation"], bpr["validation"])})
                print(json.dumps({"step": step, "validation_score": score, "improved": improved}), flush=True)
    if args.save_last:
        with candidate_parameters(model, averaged):
            save_training_checkpoint(work / "last_ema_checkpoint.pth", {"model": model.state_dict(), "step": args.steps}, args.storage_reserve_gib)
    write_json(work / "status.json", {"state": "complete", "steps": args.steps, "best_score": best_score,
                                      "seconds": time.monotonic() - started, "goal_achieved": False})


def evaluate(args):
    mf, bpr, registration = registered_baselines(args.root)
    provenance = require_synthetic_checkpoint(args.checkpoint)
    if args.output is None:
        raise ValueError("Evaluation requires a new output directory")
    if args.partition == "test":
        gate = json.loads((args.checkpoint.parent / "inner_gate.json").read_text())
        if gate["checkpoint_sha256"] != sha256(args.checkpoint) or gate["passed"] is not True:
            raise ValueError("Outer confirmation requires an eligible full inner gate")
        if gate["baseline_registration"] != registration:
            raise ValueError("Gate baseline registration changed")
    args.output.mkdir(parents=True, exist_ok=False)
    imputer = make_imputer(args.checkpoint)
    results = evaluate_public(imputer, args.partition, args.output, audit_cornac=True)
    comparisons = compare_goal(results, mf[args.partition], bpr[args.partition])
    passed = all(value["better"] for values in comparisons.values() for value in values.values())
    report = {"partition": args.partition, "checkpoint_sha256": provenance["checkpoint_sha256"],
              "baseline_registration": registration, "datasets": results, "comparisons": comparisons,
              "passed": passed, "state_signature": state_signature(imputer.model),
              "score": selection_key(results, mf[args.partition], bpr[args.partition])}
    write_json(args.output / "results.json", report)
    if args.partition == "validation" and args.checkpoint.name == "best_validation_checkpoint.pth":
        status = json.loads((args.checkpoint.parent / "status.json").read_text())
        payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True)
        selected = json.loads((args.checkpoint.parent / f"validation_{payload['step']}.json").read_text())
        history = [json.loads(path.read_text()) for path in args.checkpoint.parent.glob("validation_[0-9]*.json")]
        if status["state"] != "complete" or report["score"] != selected["score"] or report["score"] != min(value["score"] for value in history):
            raise ValueError("Checkpoint selection does not reproduce")
        write_json(args.checkpoint.parent / "inner_gate.json", report)
    print(json.dumps({"passed": passed, "score": report["score"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "preflight"))
    parser.add_argument("--root", type=Path, default=CAMPAIGN)
    parser.add_argument("--name", default="matched_bpr_weight1_200")
    parser.add_argument("--checkpoint", type=Path, default=INITIAL)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--ranking-weight", type=float, default=1)
    parser.add_argument("--ranking-boundary-weight", type=float, default=0)
    parser.add_argument("--ranking-objective", choices=("pairwise", "expected_listwise", "topk_expected_pairwise", "topk_soft_choice", "topk_finite_gap"), default="pairwise")
    parser.add_argument("--ranking-finite-gap-margin", type=float, default=.01)
    parser.add_argument("--ranking-frequency-pseudocount", type=float, default=.5)
    parser.add_argument("--ranking-top-k", type=int, default=10)
    parser.add_argument("--ranking-candidate-scope", choices=("all", "topk_union"), default="all")
    parser.add_argument("--negative-only-ranking", action="store_true")
    parser.add_argument("--center-negative-only-ranking", action="store_true")
    parser.add_argument("--project-rating-gradient", action="store_true")
    parser.add_argument("--trainable-scope", choices=("full", "exposure_embeddings"), default="full")
    parser.add_argument("--rating-weight", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--hard-negatives", type=int, default=0)
    parser.add_argument("--teacher-anchor-weight", type=float, default=0)
    parser.add_argument("--teacher-anchor-mode", choices=("prediction_mse", "macro_risk_budget"), default="prediction_mse")
    parser.add_argument("--teacher-checkpoint", type=Path, default=INITIAL)
    parser.add_argument("--catalog-fraction", type=float, default=0)
    parser.add_argument("--exposure-affinity-low", type=float, default=0.0)
    parser.add_argument("--exposure-affinity-high", type=float, default=1.0)
    parser.add_argument("--exposure-quality-low", type=float, default=0.0)
    parser.add_argument("--exposure-quality-high", type=float, default=1.5)
    parser.add_argument("--item-popularity-low", type=float, default=1.4)
    parser.add_argument("--item-popularity-high", type=float, default=2.8)
    parser.add_argument("--training-item-windows", type=int, default=1)
    parser.add_argument("--training-rows", type=int)
    parser.add_argument("--recompute-window-gradients", action="store_true")
    parser.add_argument("--offload-window-activations", action="store_true")
    parser.add_argument("--paired-context-windows", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--save-validation-details", action="store_true")
    parser.add_argument("--ranking-query-mode", choices=("global", "user_matched", "user_catalog"), default="user_matched")
    parser.add_argument("--ranking-relevance-target", choices=("conditional", "item_marginal_positive", "visible_item_frequency"), default="conditional")
    parser.add_argument("--ranking-user-scope", choices=("local_positive", "parent_positive"), default="local_positive")
    parser.add_argument("--negatives-per-user", type=int, default=16)
    parser.add_argument("--ranking-max-users", type=int, default=0)
    parser.add_argument("--ema-decay", type=float, default=.95)
    parser.add_argument("--storage-reserve-gib", type=float, default=8)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    with (args.root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "train":
            train(args)
        elif args.action == "preflight":
            preflight(args)
        else:
            evaluate(args)


if __name__ == "__main__":
    main()
