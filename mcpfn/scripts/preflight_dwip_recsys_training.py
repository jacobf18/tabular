"""Measure larger-context synthetic DWIP updates before a longer experiment."""
import argparse
import importlib.resources
import json
from pathlib import Path
import time

import torch

from dwip_recsys_experiment import compute_context, training_logits, update_ema, candidate_parameters
from dwip_recsys_provenance import require_synthetic_checkpoint, require_synthetic_episode
from evaluate_dwip_recsys import CHECKPOINT, sha256, write_json
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.prior.recsys_rating_episode import sample_rating_episode, rating_objective


def preflight_preference_family(requested, step):
    return ("gaussian", "clustered")[step % 2] if requested == "mixed" else requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=320)
    parser.add_argument("--normalization", default="shrunk_column")
    parser.add_argument("--point-target", default="expected_rating")
    parser.add_argument("--point-objective", choices=("mse", "user_rmse", "expected_user_rmse"), default="mse")
    parser.add_argument("--macro-query-scope", choices=("decoded", "parent"), default="decoded")
    parser.add_argument("--query-limit", type=int, default=1024)
    parser.add_argument("--preference-family", choices=("gaussian", "clustered", "mixed"), default="gaussian")
    parser.add_argument("--exclude-unknown-queries", action="store_true")
    parser.add_argument("--uniform-noise-low", type=float, default=0.0)
    parser.add_argument("--uniform-noise-high", type=float, default=0.0)
    parser.add_argument("--user-noise-log-std-low", type=float, default=0.0)
    parser.add_argument("--user-noise-log-std-high", type=float, default=0.0)
    parser.add_argument("--ranking-weight", type=float, default=.05)
    parser.add_argument("--item-popularity-low", type=float, default=.7)
    parser.add_argument("--item-popularity-high", type=float, default=1.9)
    parser.add_argument("--density-low", type=float, default=.01)
    parser.add_argument("--density-high", type=float, default=.12)
    parser.add_argument("--parent-rows", type=int, default=1024)
    parser.add_argument("--parent-cols", type=int)
    parser.add_argument("--context-ordering", choices=("iid", "first_observation"), default="iid")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--query-sampling", choices=("uniform_cells", "user_balanced"), default="uniform_cells")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mae-weight", type=float, default=0.0)
    parser.add_argument("--mean-rating-weight", type=float, default=0.0)
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    if args.updates < 1 or (args.preference_family == "mixed" and args.updates < 2):
        raise ValueError("Preflight requires updates covering every requested preference family")
    provenance = require_synthetic_checkpoint(args.checkpoint)
    if not 0 <= args.ema_decay < 1:
        raise ValueError("EMA decay must be zero (disabled) or strictly between zero and one")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(41)
    torch.backends.cuda.enable_cudnn_sdp(False)
    config = json.loads((CHECKPOINT.parent / "config.json").read_text())["model"]
    model = TabImputeModelStable(**config).cuda().float().train()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    if len(model.state_dict()) != 254:
        raise ValueError("Architecture changed")
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    criterion = FullSupportBarDistribution(torch.load(borders, map_location="cuda", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.02, fused=True)
    averaged = {name: value.detach().clone() for name, value in model.state_dict().items()} if args.ema_decay else None
    records = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.updates):
        torch.cuda.synchronize()
        started = time.monotonic()
        episode = sample_rating_episode(91000000 + step, args.rows, args.cols, device="cuda",
                                        query_limit=args.query_limit, parent_rows=max(args.parent_rows, args.rows),
                                        normalization=args.normalization, point_target=args.point_target,
                                        density_range=(args.density_low, args.density_high),
                                        item_popularity_sigma_range=(args.item_popularity_low, args.item_popularity_high),
                                        parent_cols=args.parent_cols, context_ordering=args.context_ordering,
                                        query_sampling=args.query_sampling,
                                        ordinal_targets=args.mae_weight > 0 or args.point_objective == "expected_user_rmse",
                                        preference_family=preflight_preference_family(args.preference_family, step),
                                        exclude_unknown_queries=args.exclude_unknown_queries,
                                        uniform_noise_range=(args.uniform_noise_low, args.uniform_noise_high),
                                        user_noise_log_std_range=(args.user_noise_log_std_low, args.user_noise_log_std_high))
        require_synthetic_episode(episode)
        optimizer.zero_grad(set_to_none=True)
        with compute_context():
            logits = training_logits(model, episode.source, episode.query, args.activation_checkpointing)
        loss, diagnostics = rating_objective(logits, episode, criterion, nll_weight=.1,
                                             ranking_weight=args.ranking_weight, mae_weight=args.mae_weight,
                                             point_objective=args.point_objective, macro_query_scope=args.macro_query_scope,
                                             mean_rating_weight=args.mean_rating_weight)
        if args.normalization == "two_way_user_scale":
            if episode.row_scales is None or not torch.isfinite(episode.row_scales).all():
                raise ValueError("Require finite context-derived user scales")
            diagnostics.update(row_scale_min=episode.row_scales.min().item(), row_scale_max=episode.row_scales.max().item())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        if averaged is not None:
            update_ema(averaged, model.state_dict(), args.ema_decay)
        torch.cuda.synchronize()
        records.append({"step": step, "seconds": time.monotonic() - started, "grad_norm": gradient.item(),
                        "loss": loss.item(), **diagnostics, **episode.metadata})
    with candidate_parameters(model, averaged), torch.no_grad(), compute_context():
        prediction = model(episode.source, decode_mask=episode.query)
        candidate_finite = torch.isfinite(prediction).all().item()
    peak = torch.cuda.max_memory_allocated()
    finite = all(torch.isfinite(value).all().item() and value.dtype == torch.float32 for value in model.state_dict().values())
    passed = finite and candidate_finite and peak < 128 * 1024**3
    report = {"passed": passed, "architecture_tensors": len(model.state_dict()), "finite_fp32_state": finite,
              "preference_family_schedule": "Alternating gaussian/clustered coverage" if args.preference_family == "mixed" else args.preference_family,
              "synthetic_provenance": provenance,
              "finite_selected_candidate": candidate_finite,
              "peak_allocated_bytes": peak, "updates": records,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "checkpoint_sha256": sha256(args.checkpoint),
              "source_sha256": {str(path): sha256(path) for path in (
                  Path(__file__), Path(__file__).with_name("dwip_recsys_experiment.py"),
                  Path(__file__).parents[1] / "src/tabimpute/prior/recsys_rating_episode.py",
                  Path(__file__).parents[1] / "src/tabimpute/prior/recsys_clustered_factors.py",
                  Path(__file__).with_name("dwip_recsys_provenance.py"))}}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Larger-context training failed its finite-state or memory preflight")


if __name__ == "__main__":
    main()
