"""Read-only comparison on shared synthetic cases, never a real-data benchmark."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from audit_dwip_bpr_campaign import inference_signature, lineage
from dwip_bpr_experiment import episode_arguments, make_imputer, objective, ranking_episode, synthetic_teacher_predictions
from dwip_bpr_window_training import windowed_training_logits
from dwip_recsys_experiment import compute_context
from dwip_recsys_provenance import require_synthetic_checkpoint, require_synthetic_episode
from evaluate_dwip_recsys import sha256, write_json
from tabimpute.prior.recsys_ranking_objective import raw_query_predictions


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        if isinstance(value, torch.Tensor):
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def synthetic_ranking_utilities(prediction, episode, top_k=10):
    require_synthetic_episode(episode)
    if episode.source.ndim != 3 or episode.source.shape[0] != 1 or episode.query.shape != episode.source.shape or episode.query.dtype != torch.bool:
        raise ValueError("Require one synthetic user-item matrix and an aligned Boolean query mask")
    if episode.rating_query.shape != episode.query.shape or episode.rating_query.dtype != torch.bool or (episode.rating_query & ~episode.query).any():
        raise ValueError("Synthetic rating queries must be a Boolean subset of all queries")
    if top_k < 1 or prediction.ndim != 1 or prediction.numel() != int(episode.query.sum()):
        raise ValueError("Require positive top-k and scores aligned to synthetic queries")
    probabilities = episode.relevance_probabilities
    if probabilities is None or probabilities.shape != episode.query.shape or probabilities.requires_grad:
        raise ValueError("Require detached synthetic relevance probabilities aligned to queries")
    relevance = probabilities[episode.query]
    if not torch.isfinite(prediction).all() or not torch.isfinite(relevance).all() or torch.any((relevance < 0) | (relevance > 1)):
        raise ValueError("Require finite scores and synthetic probabilities in [0,1]")
    indices = episode.query.nonzero()
    users, items = indices[:, 1], indices[:, 2]
    shape = episode.source.shape[1:]
    additional = ~episode.rating_query[episode.query]
    active = torch.bincount(users[additional], minlength=shape[0]) > 0
    dense_scores = prediction.new_full(shape, -torch.inf)
    dense_scores[users, items] = prediction.clamp(1, 5)
    dense_relevance = prediction.new_zeros(shape)
    dense_relevance[users, items] = relevance.to(prediction.dtype)
    scores, relevance = dense_scores[active], dense_relevance[active]
    included = relevance.sum(1) > 0
    if not included.any():
        raise ValueError("Require ranking users with positive synthetic relevance mass")
    scores, relevance = scores[included], relevance[included]
    count = min(top_k, shape[1])
    selected = scores.argsort(dim=1, descending=True, stable=True)[:, :count]
    selected_relevance = relevance.gather(1, selected)
    ideal_relevance = relevance.sort(dim=1, descending=True).values[:, :count]
    discounts = 1 / (torch.arange(count, device=prediction.device, dtype=prediction.dtype) + 2).log2()
    expected_dcg = (selected_relevance * discounts).sum(1)
    ideal_dcg = (ideal_relevance * discounts).sum(1)
    selected_scores = scores.gather(1, selected)
    valid_counts = selected_scores.isfinite().sum(1).clamp_min(1)
    return {"ranking_users": int(included.sum()), "top_k": top_k,
            "expected_dcg_efficiency": float((expected_dcg / ideal_dcg).mean()),
            "expected_positive_mass_at_k": float(selected_relevance.sum(1).mean()),
            "ideal_expected_positive_mass_at_k": float(ideal_relevance.sum(1).mean()),
            "expected_positive_mass_recall": float((selected_relevance.sum(1) / relevance.sum(1)).mean()),
            "top_k_score_ceiling_fraction": float(((selected_scores >= 5).sum(1) / valid_counts).mean())}


def synthetic_context_references(episode):
    require_synthetic_episode(episode)
    global_counts = episode.global_col_support
    if global_counts is None or global_counts.shape != (episode.source.shape[-1],) or not torch.isfinite(global_counts).all() or (global_counts < 0).any():
        raise ValueError("Require finite nonnegative synthetic parent item counts")
    local = torch.isfinite(episode.source[0]).sum(0)
    items = episode.query.nonzero()[:, 2]
    utilities = {}
    for name, counts in (("visible_item_count", local), ("global_parent_item_count", global_counts)):
        scores = counts.float().log1p()
        scores = 1 + 3.5 * scores / scores.max().clamp_min(1)
        utilities[name] = synthetic_ranking_utilities(scores[items], episode)
    relevance = episode.relevance_probabilities[episode.query]
    return {"utilities": utilities, "support_density": float(torch.isfinite(episode.source).float().mean()),
            "local_item_context_zero_fraction": float((local == 0).float().mean()),
            "query_fraction_without_local_item_context": float((local[items] == 0).float().mean()),
            "query_relevance_mass_without_local_item_context": float((relevance * (local[items] == 0)).sum() / relevance.sum())}


@torch.no_grad()
def run(args):
    configuration = json.loads(args.training_config.read_text())
    arguments = episode_arguments(argparse.Namespace(**configuration["arguments"]))
    if arguments.project_rating_gradient or len(args.checkpoints) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require multiple checkpoints, distinct synthetic seeds and an unprojected objective")
    labels = [checkpoint.parent.name for checkpoint in args.checkpoints]
    if len(set(labels)) != len(labels) or "synthetic_teacher" in labels:
        raise ValueError("Require distinct checkpoint labels")
    historical_seeds = set()
    provenance = {}
    for checkpoint in args.checkpoints:
        provenance[str(checkpoint.resolve())] = require_synthetic_checkpoint(checkpoint)
        for ancestor in lineage(checkpoint):
            path = Path(ancestor["checkpoint"]).parent / "config.json"
            if path.exists():
                historical_seeds.add(json.loads(path.read_text()).get("arguments", {}).get("seed"))
    if any(seed in historical_seeds or seed < 1 for seed in args.seeds):
        raise ValueError("Synthetic diagnostic seeds must be positive and distinct from ancestral training seeds")
    original_signature = inference_signature()
    frozen_signature = json.loads((Path(arguments.root) / "fixed_protocol.json").read_text())["inference_signature"]
    if original_signature != frozen_signature:
        raise ValueError("Registered architecture/inference implementation changed")
    teacher_checkpoint = Path(arguments.teacher_checkpoint) if arguments.teacher_anchor_weight else None
    teacher_provenance = require_synthetic_checkpoint(teacher_checkpoint) if teacher_checkpoint is not None else None
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {Path(source) for source in configuration["source_sha256"]} | {Path(__file__)}
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copyfile(source, snapshot / source.name)
    manifest = {"synthetic_only": True, "real_data_loaded": False, "neural_training": False,
                "training_config": str(args.training_config.resolve()), "training_config_sha256": sha256(args.training_config),
                "checkpoint_provenance": provenance, "seeds": args.seeds, "inference_signature": original_signature,
                "teacher_checkpoint": str(teacher_checkpoint.resolve()) if teacher_checkpoint is not None else None,
                "teacher_provenance": teacher_provenance,
                "source_sha256": {str(source.resolve()): sha256(source) for source in sources},
                "interpretation": "Ratios of expected synthetic utility on training-style unobserved slates; not exact expectations of NDCG/Recall, Cornac metrics, or a goal gate"}
    write_json(args.output / "manifest.json", manifest)
    teacher = make_imputer(teacher_checkpoint) if teacher_checkpoint is not None else None
    candidates = {label: make_imputer(checkpoint) for label, checkpoint in zip(labels, args.checkpoints)}
    models = {label: imputer.model for label, imputer in candidates.items()}
    if teacher is not None:
        models["synthetic_teacher"] = teacher.model
    for model in models.values():
        model.eval().requires_grad_(False)
    original_states = {label: tensor_digest(model.state_dict()) for label, model in models.items()}
    torch.cuda.reset_peak_memory_stats()
    records = []
    window_cols = arguments.cols if arguments.training_item_windows > 1 else 0
    for seed in args.seeds:
        for fraction in (0., 1.):
            selected = argparse.Namespace(**{**vars(arguments), "seed": seed, "catalog_fraction": fraction})
            episode = ranking_episode(selected, 1)
            require_synthetic_episode(episode)
            fingerprint = tensor_digest(vars(episode))
            context_references = synthetic_context_references(episode)
            target = synthetic_teacher_predictions(teacher, episode, window_cols) if teacher is not None else None
            for label, imputer in candidates.items():
                with compute_context():
                    logits = windowed_training_logits(imputer.model, episode.source, episode.query, window_cols=window_cols)
                loss, diagnostics = objective(logits, episode, imputer.bar_distribution, selected, target)
                prediction, _, _, _ = raw_query_predictions(logits, episode, imputer.bar_distribution)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite synthetic diagnostic objective")
                record = {"checkpoint": label, "seed": seed, "mode": episode.metadata["context_ordering"],
                          "shared_episode_sha256": fingerprint, "loss": float(loss),
                          "context_references": context_references,
                          "diagnostics": diagnostics, "utilities": synthetic_ranking_utilities(prediction, episode)}
                records.append(record)
                print(json.dumps(record), flush=True)
                del logits, loss, prediction
            if tensor_digest(vars(episode)) != fingerprint:
                raise ValueError("Read-only synthetic evaluation changed its shared episode")
            del episode, target
    unchanged = all(tensor_digest(model.state_dict()) == original_states[label] for label, model in models.items())
    if not unchanged or inference_signature() != original_signature or any(parameter.grad is not None for model in models.values() for parameter in model.parameters()):
        raise ValueError("Synthetic diagnostic changed a model or the frozen inference implementation")
    averages = {}
    for label in labels:
        selected = [record for record in records if record["checkpoint"] == label]
        averages[label] = {"loss": float(np.mean([record["loss"] for record in selected])),
                           **{key: float(np.mean([record["utilities"][key] for record in selected]))
                              for key in selected[0]["utilities"] if key not in ("ranking_users", "top_k")},
                           **{key: float(np.mean([record["diagnostics"][key] for record in selected]))
                              for key in ("ranking", "rating_total", "synthetic_teacher_anchor")}}
    write_json(args.output / "results.json", {**manifest, "model_states_unchanged": unchanged,
               "model_state_sha256": original_states, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "gradients_enabled": torch.is_grad_enabled(), "cases": records, "averages": averages})
    print(json.dumps({"averages": averages, "model_states_unchanged": unchanged}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[901, 902, 903])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    root = Path(json.loads(args.training_config.read_text())["arguments"]["root"])
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
