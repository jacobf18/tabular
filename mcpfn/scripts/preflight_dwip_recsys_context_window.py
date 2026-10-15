"""Synthetic-input public-inference memory and parity gate; never trains weights."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from dwip_recsys_experiment import compute_context, inference_components, inference_row_scales
from dwip_recsys_provenance import require_synthetic_checkpoint
from evaluate_dwip_recsys import sha256, write_json
from evaluate_dwip_recsys_candidate import context_window_selection, impute_candidate_ratings
from tabimpute.interface import ImputePFN
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance = require_synthetic_checkpoint(args.checkpoint)
    shape, context_selection = context_window_selection(args.checkpoint, args.context_screen)
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    normalization = config["arguments"]["normalization"]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    episode = sample_rating_episode(84001, *shape, parent_rows=max(2048, shape[0]), parent_cols=max(2048, shape[1]),
                                     normalization=normalization, density_range=(.03, .03), query_limit=128,
                                     point_target="expected_rating", uniform_noise_range=(0., .15),
                                     user_noise_log_std_range=(.15, .6))
    observed = episode.source[0] * episode.row_scales[:, None] * episode.stds + episode.means + episode.row_offsets[:, None]
    observed = observed.round().double().numpy()
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["model"]
    imputer = ImputePFN(device="cuda", checkpoint_path=str(args.checkpoint), entry_wise_features=False,
                        json_config={"config": config["model"]}, model_family="new_stable", prediction_stat="mean",
                        precision_mode="amp_bfloat16", max_num_rows=shape[0], max_num_cols=shape[1], max_num_chunks=1, verbose=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prediction = impute_candidate_ratings(imputer, observed, normalization)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3
    assert peak < 128 and np.isfinite(prediction).all()
    assert np.all((prediction >= 1) & (prediction <= 5))
    support = np.isfinite(observed)
    np.testing.assert_allclose(prediction[support], observed[support], rtol=1e-12, atol=1e-12)
    means, stds, offsets = inference_components(observed, normalization)
    scales = inference_row_scales(observed, normalization, means, stds, offsets)
    normalized = (observed - offsets[:, None] - means) / stds / scales[:, None]
    query = episode.rating_query[0].numpy()
    rows, cols = np.nonzero(query)
    assert len(rows) > 0
    tensor = torch.tensor(normalized, device="cuda", dtype=torch.float32).unsqueeze(0)
    mask = torch.tensor(query, device="cuda").unsqueeze(0)
    with torch.no_grad(), compute_context():
        logits = imputer.model(tensor, decode_mask=mask)
    query_prediction = imputer.bar_distribution.mean(logits.float()).cpu().numpy()
    query_prediction = np.clip(query_prediction * scales[rows] * stds[cols] + means[cols] + offsets[rows], 1, 5)
    discrepancy = float(np.max(np.abs(query_prediction - prediction[rows, cols])))
    np.testing.assert_allclose(query_prediction, prediction[rows, cols], rtol=1e-6, atol=1e-6)
    assert all(torch.equal(tensor.cpu(), original[name]) for name, tensor in imputer.model.state_dict().items())
    assert all(parameter.grad is None for parameter in imputer.model.parameters())
    report = {"passed": True, "input_data": "Synthetic only", "neural_updates": False, "weights_bitwise_unchanged": True,
              "synthetic_provenance": provenance, "context_selection": context_selection, "block_shape": shape,
              "normalization": normalization, "peak_public_inference_allocated_gib": peak, "public_inference_seconds": seconds,
              "finite_clipped_output": True, "observed_ratings_preserved": True, "query_public_parity": True,
              "max_query_public_difference": discrepancy, "query_count": len(rows)}
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("evaluate_dwip_recsys_candidate.py").resolve(),
               Path(__file__).with_name("dwip_recsys_experiment.py").resolve(), root / "src/tabimpute/interface.py",
               root / "src/tabimpute/model/model_new_stable.py", root / "src/tabimpute/model/bar_distribution.py",
               root / "src/tabimpute/prior/recsys_rating_episode.py"]
    snapshot = args.output.with_suffix(".source_snapshot")
    snapshot.mkdir()
    report["source_sha256"] = {str(source): sha256(source) for source in sources}
    for source in sources:
        shutil.copyfile(source, snapshot / source.name)
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
