"""Verify joint versus replayed DWIP gradients on one exclusively synthetic CUDA episode."""
import argparse
import json
import math
from pathlib import Path
import shutil

import torch

from audit_dwip_bpr_campaign import inference_signature
from dwip_bpr_evaluation import CAMPAIGN, INITIAL
from dwip_bpr_experiment import make_imputer, objective, state_signature, synthetic_teacher_predictions
from dwip_bpr_window_training import recomputed_window_logits, windowed_training_logits
from dwip_recsys_experiment import compute_context
from dwip_recsys_provenance import require_synthetic_checkpoint, require_synthetic_episode
from evaluate_dwip_recsys import sha256, write_json
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def verify(checkpoint, output, offload_activations=False):
    if output.exists():
        raise ValueError("Preserve existing verification results")
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    provenance = require_synthetic_checkpoint(checkpoint)
    args = argparse.Namespace(**json.loads((checkpoint.parent / "config.json").read_text())["arguments"])
    imputer = make_imputer(checkpoint)
    teacher = make_imputer(INITIAL)
    teacher.model.requires_grad_(False)
    model = imputer.model
    model.train()
    signature = state_signature(model)
    episode = sample_rating_episode(69000020, rows=128, cols=128, parent_rows=256, parent_cols=256,
        device="cuda", query_limit=256, density_range=(.1, .2), normalization="two_way_user_scale",
        point_target="expected_rating", ordinal_targets=True, exclude_unknown_queries=True,
        context_ordering="first_observation_paired", paired_window_cols=64,
        ranking_query_mode="user_catalog", ranking_user_scope="parent_positive",
        relevance_targets=True, ranking_max_users=8, negatives_per_user=128)
    require_synthetic_episode(episode)
    target = synthetic_teacher_predictions(teacher, episode, 64)
    torch.manual_seed(691)
    with compute_context():
        logits = windowed_training_logits(model, episode.source, episode.query, True, 64)
    reference_logits = logits.detach().clone()
    loss = objective(logits, episode, imputer.bar_distribution, args, target)[0]
    reference_loss = loss.item()
    loss.backward()
    reference_cpu_rng = torch.get_rng_state()
    reference_cuda_rng = torch.cuda.get_rng_state()
    reference = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    del logits, loss
    model.zero_grad(set_to_none=True)
    torch.manual_seed(691)
    with compute_context():
        logits, replay = recomputed_window_logits(model, episode.source, episode.query, True, 64, offload_activations)
    identical_logits = torch.equal(logits.detach(), reference_logits)
    loss = objective(logits, episode, imputer.bar_distribution, args, target)[0]
    replayed_loss = loss.item()
    loss.backward()
    windows = replay.backward(logits.grad, compute_context)
    current = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    if current.keys() != reference.keys():
        raise ValueError("Replayed model gradient coverage differs")
    reference_squared = sum(value.square().sum().item() for value in reference.values())
    current_squared = sum(value.square().sum().item() for value in current.values())
    difference_squared = sum((current[name] - value).square().sum().item() for name, value in reference.items())
    dot = sum((current[name] * value).sum().item() for name, value in reference.items())
    relative_error = math.sqrt(difference_squared / reference_squared)
    cosine = dot / math.sqrt(reference_squared * current_squared)
    rng_preserved = torch.equal(torch.get_rng_state(), reference_cpu_rng) and torch.equal(torch.cuda.get_rng_state(), reference_cuda_rng)
    unchanged = state_signature(model) == signature and inference_signature() == json.loads((CAMPAIGN / "fixed_protocol.json").read_text())["inference_signature"]
    teacher_frozen = all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.model.parameters())
    passed = identical_logits and abs(replayed_loss - reference_loss) < 1e-5 and relative_error < .02 and cosine > .999 and rng_preserved and unchanged and teacher_frozen
    report = {"passed": passed, "synthetic_only": True, "real_data_loaded": False,
        "offload_window_activations": offload_activations,
        "checkpoint_provenance": provenance, "teacher_frozen": teacher_frozen,
        "reference_loss": reference_loss, "replayed_loss": replayed_loss, "identical_logits": identical_logits,
        "relative_gradient_l2_error": relative_error, "gradient_cosine": cosine, "rng_preserved": rng_preserved,
        "gradient_tensors": len(reference), "verified_replayed_windows": windows,
        "architecture_inference_unchanged": unchanged, "episode": episode.metadata,
        "precision_note": "Joint versus sequential BF16 parameter-gradient accumulation can differ by rounding; exact per-window logits are mandatory.",
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
            (Path(__file__).name, "dwip_bpr_window_training.py", "dwip_bpr_experiment.py")},
        "goal_completion_checked": False}
    snapshot = output.with_name(output.stem + "_source_snapshot")
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in report["source_sha256"]:
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("episode", "source_sha256", "checkpoint_provenance")}), flush=True)
    if not passed:
        raise ValueError("Synthetic CUDA gradient-replay verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload-window-activations", action="store_true")
    args = parser.parse_args()
    verify(args.checkpoint, args.output, args.offload_window_activations)


if __name__ == "__main__":
    main()
