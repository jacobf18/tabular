"""Train unchanged DWIP on synthetic data only, with inner-only selection."""
import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import importlib.resources
import io
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from cornac.eval_methods import BaseMethod
from cornac.models import MF

from evaluate_dwip_recsys import (
    CHECKPOINT, OUTPUT, DATASETS, FrozenRatings, METRIC_NAMES, metrics,
    normalize_training_ratings, sha256, user_rating_metrics, write_json,
)
from tabimpute.interface import ImputePFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.prior.recsys_rating_episode import sample_rating_episode, rating_objective, observed_user_scales
from dwip_recsys_provenance import (
    POLICY, TRAINING_DATA, require_synthetic_checkpoint, require_synthetic_episode,
    write_synthetic_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "runs/recsys_goal_20260920"


def inner_split(train_users, train_items, train_ratings, seed=20260920, fraction=0.1):
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must lie in (0,1)")
    random = np.random.default_rng(seed)
    order = random.permutation(len(train_ratings))
    validation = order[:int(len(order) * fraction)]
    context = order[int(len(order) * fraction):]
    known = np.isin(train_users[validation], train_users[context]) & np.isin(train_items[validation], train_items[context])
    validation = validation[known]
    return context, validation


def load_split(folder, partition):
    with np.load(OUTPUT / folder / "split.npz", allow_pickle=False) as data:
        shape = tuple(data["shape"])
        users, items, ratings = (data[f"train_{key}"].copy() for key in ("users", "items", "ratings"))
        if partition == "validation":
            context, validation = inner_split(users, items, ratings)
            queries = (users[validation], items[validation], ratings[validation])
            users, items, ratings = users[context], items[context], ratings[context]
        elif partition == "test":
            queries = tuple(data[f"test_{key}"].copy() for key in ("users", "items", "ratings"))
        else:
            raise ValueError("Unknown evaluation partition")
    observed = np.full(shape, np.nan)
    observed[users, items] = ratings
    if not np.isnan(observed[queries[0], queries[1]]).all():
        raise ValueError("Query leakage into context")
    return observed, (users, items, ratings), queries


def cornac_split(context, queries):
    train = list(zip(context[0].tolist(), context[1].tolist(), context[2].tolist()))
    test = list(zip(queries[0].tolist(), queries[1].tolist(), queries[2].tolist()))
    return BaseMethod.from_splits(train_data=train, test_data=test, rating_threshold=4.0,
                                  exclude_unknowns=True, seed=123)


def training_episode(args, step, *, seed=None, preserve_user_item_axes=False):
    seed = args.seed * 1000000 + step if seed is None else seed
    rows, cols = (args.rows, args.cols) if preserve_user_item_axes or step % 2 else (args.cols, args.rows)
    return sample_rating_episode(seed, rows=rows, cols=cols, device="cuda", query_limit=getattr(args, "query_limit", 1024),
                                 density_range=(args.density_low, args.density_high), parent_rows=max(args.parent_rows, rows),
                                 normalization=args.normalization, point_target=args.point_target,
                                 item_popularity_sigma_range=(args.item_popularity_low, args.item_popularity_high),
                                 exposure_affinity_range=(getattr(args, "exposure_affinity_low", 0.0),
                                                          getattr(args, "exposure_affinity_high", 1.0)),
                                 exposure_quality_range=(getattr(args, "exposure_quality_low", 0.0),
                                                         getattr(args, "exposure_quality_high", 1.5)),
                                 parent_cols=args.parent_cols, context_ordering=args.context_ordering,
                                 paired_window_cols=getattr(args, "paired_window_cols", 0),
                                 query_sampling=args.query_sampling,
                                 ordinal_targets=args.mae_weight > 0 or getattr(args, "point_objective", "mse") == "expected_user_rmse",
                                 preference_family=getattr(args, "preference_family", "gaussian"),
                                 exclude_unknown_queries=getattr(args, "exclude_unknown_queries", False),
                                 uniform_noise_range=(getattr(args, "uniform_noise_low", 0.0), getattr(args, "uniform_noise_high", 0.0)),
                                 user_noise_log_std_range=(getattr(args, "user_noise_log_std_low", 0.0), getattr(args, "user_noise_log_std_high", 0.0)),
                                 ranking_query_mode=getattr(args, "ranking_query_mode", "global"),
                                 negatives_per_user=getattr(args, "negatives_per_user", 16),
                                 ranking_max_users=getattr(args, "ranking_max_users", 0),
                                 ranking_user_scope=getattr(args, "ranking_user_scope", "local_positive"),
                                 relevance_targets=getattr(args, "ranking_objective", "pairwise") in ("expected_listwise", "topk_expected_pairwise", "topk_soft_choice", "topk_finite_gap"),
                                 relevance_target=getattr(args, "ranking_relevance_target", "conditional"))


def run_mf(root):
    results = {}
    for partition in ("validation", "test"):
        results[partition] = {}
        for spec in DATASETS.values():
            _, context, queries = load_split(spec["folder"], partition)
            split = cornac_split(context, queries)
            model = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123,
                       num_threads=1)
            result, _ = split.evaluate(model=model, metrics=metrics(), user_based=True)
            results[partition][spec["folder"]] = {key: float(result.metric_avg_results[key]) for key in METRIC_NAMES}
    write_json(root / "mf_baseline.json", results)
    return results


def compute_context():
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()


def training_logits(model, source, query, activation_checkpointing=False):
    if not activation_checkpointing:
        return model(source, decode_mask=query)
    features = model.feature_encoder(source)
    cls_rows = model.feature_encoder.last_num_cls_rows
    cls_cols = model.feature_encoder.last_num_cls_cols
    for block in model.transformer_blocks:
        features = checkpoint(block, features, num_cls_rows=cls_rows, num_cls_cols=cls_cols, use_reentrant=False)
    features = features[:, cls_rows:, cls_cols:, :]
    if query.shape != features.shape[:-1]:
        raise ValueError("Query mask does not match feature geometry")
    return model.decoder(features[query])


@torch.no_grad()
def update_ema(averaged, current, decay):
    if not 0 < decay < 1:
        raise ValueError("EMA decay must be strictly between zero and one")
    for name, value in current.items():
        if value.is_floating_point():
            averaged[name].lerp_(value.detach(), 1 - decay)
        else:
            averaged[name].copy_(value)


@contextmanager
def candidate_parameters(model, averaged):
    if averaged is None:
        yield
        return
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    try:
        model.load_state_dict(averaged, strict=True)
        yield
    finally:
        model.load_state_dict(original, strict=True)


def inference_scaling(observed, mode="column"):
    if mode == "column":
        return normalize_training_ratings(observed)
    if mode not in ("global", "shrunk_column"):
        raise ValueError("Unknown normalization")
    values = observed[np.isfinite(observed)]
    if not values.size:
        return np.zeros(observed.shape[1]), np.ones(observed.shape[1])
    std = values.std()
    std = 1.0 if std <= 1e-4 else std
    means = np.full(observed.shape[1], values.mean())
    if mode == "shrunk_column":
        counts = np.isfinite(observed).sum(0)
        means = (np.nansum(observed, axis=0) + 10 * values.mean()) / (counts + 10)
    return means, np.full(observed.shape[1], std)


def inference_components(observed, mode="column"):
    if mode not in ("two_way", "two_way_user_scale"):
        return *inference_scaling(observed, mode), np.zeros(observed.shape[0])
    rows, cols = np.where(np.isfinite(observed))
    values = observed[rows, cols]
    if not values.size:
        return np.zeros(observed.shape[1]), np.ones(observed.shape[1]), np.zeros(observed.shape[0])
    mean = values.mean()
    row_counts = np.bincount(rows, minlength=observed.shape[0])
    col_counts = np.bincount(cols, minlength=observed.shape[1])
    row_bias = np.zeros(observed.shape[0])
    col_bias = np.zeros(observed.shape[1])
    for _ in range(30):
        row_bias = np.bincount(rows, weights=values - mean - col_bias[cols], minlength=observed.shape[0]) / (row_counts + 3)
        col_bias = np.bincount(cols, weights=values - mean - row_bias[rows], minlength=observed.shape[1]) / (col_counts + 3)
    std = np.std(values - mean - row_bias[rows] - col_bias[cols])
    std = std if std > 1e-4 else 1.0
    return mean + col_bias, np.full(observed.shape[1], std), row_bias


def inference_row_scales(observed, mode, means, stds, row_offsets):
    if mode != "two_way_user_scale":
        return None
    ratings = torch.from_numpy(observed)
    return observed_user_scales(ratings, torch.isfinite(ratings), torch.as_tensor(means),
                                torch.as_tensor(stds), torch.as_tensor(row_offsets)).numpy()


def block_spans(size, block=1000):
    return ImputePFN._get_chunk_spans(None, size, min(size, block))


@torch.no_grad()
def rating_validation(model, criterion, partition="validation", normalization="column", block_rows=1000, block_cols=1000):
    if block_rows < 1 or block_cols < 1:
        raise ValueError("Require positive evaluation block dimensions")
    model.eval()
    results = {}
    device = next(model.parameters()).device
    for spec in DATASETS.values():
        observed, _, queries = load_split(spec["folder"], partition)
        means, stds, row_offsets = inference_components(observed, normalization)
        source = (observed - row_offsets[:, None] - means) / stds
        row_scales = inference_row_scales(observed, normalization, means, stds, row_offsets)
        if row_scales is not None:
            source = source / row_scales[:, None]
        users, items, truth = queries
        predictions = np.zeros(len(truth), dtype=np.float64)
        counts = np.zeros(len(truth), dtype=np.int32)
        for row_start, row_end in block_spans(source.shape[0], block_rows):
            for col_start, col_end in block_spans(source.shape[1], block_cols):
                selected = (users >= row_start) & (users < row_end) & (items >= col_start) & (items < col_end)
                if not selected.any():
                    continue
                tensor = torch.tensor(source[row_start:row_end, col_start:col_end], device=device, dtype=torch.float32).unsqueeze(0)
                mask = torch.zeros(tensor.shape, dtype=torch.bool, device=device)
                mask[0, users[selected] - row_start, items[selected] - col_start] = True
                with compute_context():
                    logits = model(tensor, decode_mask=mask)
                local = torch.zeros(tensor.shape, device=device)
                local[mask] = criterion.mean(logits.float())
                predicted = local[0, users[selected] - row_start, items[selected] - col_start].cpu().numpy()
                if row_scales is not None:
                    predicted = predicted * row_scales[users[selected]]
                predictions[selected] += predicted * stds[items[selected]] + means[items[selected]] + row_offsets[users[selected]]
                counts[selected] += 1
        if not (counts > 0).all():
            raise ValueError("Incomplete validation prediction coverage")
        results[spec["folder"]] = user_rating_metrics(users, truth, np.clip(predictions / counts, 1, 5))
    return results


def selection_score(results, baseline, objective="mean_rmse"):
    if objective == "mean_rmse":
        return float(np.mean([results[name]["RMSE"] / baseline[name]["RMSE"] for name in results]))
    if objective == "worst_rating":
        return max(results[name][metric] / baseline[name][metric] for name in results for metric in ("MAE", "RMSE"))
    raise ValueError("Unknown checkpoint-selection objective")


def save_checkpoint(path, payload, reserve_gib=18):
    if not math.isfinite(reserve_gib) or reserve_gib <= 0:
        raise ValueError("Require a finite positive checkpoint reserve")
    temporary = path.with_suffix(".tmp")
    if temporary.exists() or path.is_symlink():
        raise ValueError("Preserve unexpected checkpoint path")
    with io.BytesIO() as buffer:
        torch.save(payload, buffer)
        if shutil.disk_usage(path.parent).free - buffer.tell() < reserve_gib * 1024**3:
            raise RuntimeError(f"Atomic checkpoint would breach {reserve_gib:g} GiB reserve; preserve all historical artifacts")
        buffer.seek(0)
        with temporary.open("xb") as handle:
            shutil.copyfileobj(buffer, handle)
    temporary.replace(path)


def save_training_checkpoint(path, payload, reserve_gib=18):
    save_checkpoint(path, payload, reserve_gib)
    write_synthetic_receipt(path)


def train(args):
    if getattr(args, "real_adaptation", False):
        raise ValueError("Real-data adaptation is prohibited; synthetic training only")
    provenance = require_synthetic_checkpoint(args.checkpoint)
    if not 0 <= args.ema_decay < 1:
        raise ValueError("EMA decay must be zero (disabled) or strictly between zero and one")
    work = args.root / args.name
    work.mkdir(parents=True, exist_ok=False)
    config = json.loads((CHECKPOINT.parent / "config.json").read_text())["model"]
    torch.manual_seed(args.seed)
    model = TabImputeModelStable(**config).cuda().float()
    payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    del payload
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    criterion = FullSupportBarDistribution(torch.load(borders, weights_only=True, map_location="cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02, fused=True)
    averaged = {name: value.detach().clone() for name, value in model.state_dict().items()} if args.ema_decay else None
    sources = [Path(__file__), Path(__file__).with_name("dwip_recsys_provenance.py"),
               ROOT / "src/tabimpute/prior/recsys_rating_episode.py",
               ROOT / "src/tabimpute/prior/recsys_clustered_factors.py"]
    manifest = {"model": config, "architecture": "Dwip v2-2; unchanged network/state-dict shapes; normalization registered separately",
                "initial_checkpoint": str(args.checkpoint.resolve()), "initial_sha256": provenance["checkpoint_sha256"],
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "source_sha256": {str(path): sha256(path) for path in sources},
                "selection": args.selection_objective + "; inner validation only; outer test not used",
                "training_data": TRAINING_DATA, "training_policy": POLICY,
                "initial_provenance": provenance,
                "normalization": args.normalization,
                "selected_weights": "Exponential moving average; single network at inference" if args.ema_decay else "Raw optimizer weights",
                "storage": "Best model only" if args.model_only else "Best model plus latest optimizer checkpoint",
                "storage_reserve_gib": args.storage_reserve_gib}
    write_json(work / "config.json", manifest)
    snapshot = work / "source_snapshot"
    snapshot.mkdir()
    for source in manifest["source_sha256"]:
        shutil.copyfile(source, snapshot / Path(source).name)
    baseline = json.loads((args.root / "mf_baseline.json").read_text())["validation"]
    initial = rating_validation(model, criterion, normalization=args.normalization)
    best_score = selection_score(initial, baseline, args.selection_objective)
    write_json(work / "initial_validation.json", {"score": best_score, "datasets": initial})
    write_json(work / "validation_0.json", {"step": 0, "score": best_score, "improved": True, "datasets": initial})
    save_training_checkpoint(work / "best_validation_checkpoint.pth", {"model": model.state_dict(), "step": 0}, args.storage_reserve_gib)
    print(json.dumps({"step": 0, "score": best_score, "datasets": initial}), flush=True)
    started = time.monotonic()
    with (work / "training.jsonl").open("x") as log:
        for step in range(1, args.steps + 1):
            model.train()
            warmup = min(1.0, step / min(200, max(1, args.steps // 10)))
            decay = 0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup * decay
            episode = training_episode(args, step)
            require_synthetic_episode(episode)
            if not episode.rating_query.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with compute_context():
                logits = training_logits(model, episode.source, episode.query, args.activation_checkpointing)
            loss, diagnostics = rating_objective(logits, episode, criterion, args.nll_weight,
                                                 args.ranking_weight, args.mae_weight,
                                                 point_objective=args.point_objective, macro_query_scope=args.macro_query_scope,
                                                 mean_rating_weight=args.mean_rating_weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            if averaged is not None:
                update_ema(averaged, model.state_dict(), args.ema_decay)
            if step % 25 == 0 or step == 1:
                entry = {"step": step, "loss": loss.item(), **diagnostics, **episode.metadata,
                         "grad_norm": grad_norm.item(), "seconds": time.monotonic() - started}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps(entry), flush=True)
            if step % args.validate_every == 0 or step == args.steps:
                with candidate_parameters(model, averaged):
                    results = rating_validation(model, criterion, normalization=args.normalization)
                    score = selection_score(results, baseline, args.selection_objective)
                    improved = score < best_score
                    if improved:
                        best_score = score
                        save_training_checkpoint(work / "best_validation_checkpoint.pth", {"model": model.state_dict(), "step": step}, args.storage_reserve_gib)
                write_json(work / f"validation_{step}.json", {"step": step, "score": score, "improved": improved, "datasets": results})
                if not args.model_only:
                    save_training_checkpoint(work / "latest_checkpoint.pth", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                                                   "step": step, "best_score": best_score, "rng": torch.get_rng_state(),
                                                                   "cuda_rng": torch.cuda.get_rng_state(), "ema": averaged}, args.storage_reserve_gib)
                print(json.dumps({"step": step, "validation_score": score, "improved": improved, "datasets": results}), flush=True)
    write_json(work / "status.json", {"state": "complete", "steps": args.steps, "best_score": best_score,
                                      "seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baseline", "train"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--name", default="exposed_noise_mse_lr1e5")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--cols", type=int, default=160)
    parser.add_argument("--density-low", type=float, default=0.01)
    parser.add_argument("--density-high", type=float, default=0.12)
    parser.add_argument("--nll-weight", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--normalization", choices=("column", "global", "shrunk_column", "two_way", "two_way_user_scale"), default="column")
    parser.add_argument("--model-only", action="store_true", help="Keep only the best model; no optimizer resumption")
    parser.add_argument("--point-target", choices=("observed", "expected_rating"), default="observed")
    parser.add_argument("--point-objective", choices=("mse", "user_rmse", "expected_user_rmse"), default="mse")
    parser.add_argument("--macro-query-scope", choices=("decoded", "parent"), default="decoded")
    parser.add_argument("--query-limit", type=int, default=1024)
    parser.add_argument("--preference-family", choices=("gaussian", "clustered", "mixed"), default="gaussian")
    parser.add_argument("--exclude-unknown-queries", action="store_true")
    parser.add_argument("--uniform-noise-low", type=float, default=0.0)
    parser.add_argument("--uniform-noise-high", type=float, default=0.0)
    parser.add_argument("--user-noise-log-std-low", type=float, default=0.0)
    parser.add_argument("--user-noise-log-std-high", type=float, default=0.0)
    parser.add_argument("--item-popularity-low", type=float, default=.7)
    parser.add_argument("--item-popularity-high", type=float, default=1.9)
    parser.add_argument("--selection-objective", choices=("mean_rmse", "worst_rating"), default="worst_rating")
    parser.add_argument("--parent-rows", type=int, default=1024)
    parser.add_argument("--parent-cols", type=int)
    parser.add_argument("--context-ordering", choices=("iid", "first_observation"), default="iid")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--query-sampling", choices=("uniform_cells", "user_balanced"), default="uniform_cells")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--mae-weight", type=float, default=0.0)
    parser.add_argument("--mean-rating-weight", type=float, default=0.0)
    parser.add_argument("--storage-reserve-gib", type=float, default=18)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    with (args.root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "baseline":
            print(json.dumps(run_mf(args.root), indent=2))
        else:
            train(args)


if __name__ == "__main__":
    main()
