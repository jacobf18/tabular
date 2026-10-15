"""Same-step k=5 audit using the benchmark's unchanged folds and scoring helper."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from run_rowcol_controlled_experiments import REPO, WORK, BENCH, write_json
from tabimpute.interface import ImputePFN


def capture_query_flow(model, query_start):
    stats = {"population": "missing entries in held-out test rows, fold 0 MI@5", "query_row_start": query_start}
    module = model.cell_query_decoder
    original = module._chunk
    current = {}

    def chunk(*args):
        _, _, _, src, b, r, c = args
        current["selected"] = src[b, r, c].isnan() & (r >= query_start)
        current["cols"] = src.shape[-1]
        return original(*args)
    module._chunk = chunk

    def layer_hook(number):
        def hook(layer, inputs):
            key = f"layer_{number}"
            selected = current["selected"].nonzero().flatten()[:32]
            if key in stats or not selected.numel():
                return
            query, memory, allowed = (x[selected] for x in inputs)
            n, length, width = memory.shape
            q = layer.q(layer.query_norm(query)).reshape(n, layer.heads, 1, -1).float()
            k, _ = layer.kv(layer.memory_norm(memory)).chunk(2, -1)
            k = k.reshape(n, length, layer.heads, -1).transpose(1, 2).float()
            logits = (q @ k.transpose(-1, -2)) / (width // layer.heads) ** .5
            prob = logits.masked_fill(~allowed[:, None, None, :], float("-inf")).softmax(-1)
            entropy = -(prob * prob.clamp_min(1e-30).log()).sum(-1)
            count = allowed.sum(-1).float()
            cols = current["cols"]
            stats[key] = {"queries": n, "allowed_mean": float(count.mean()),
                "normalized_entropy": float((entropy.squeeze(-1) / count.log()[:, None]).mean()),
                "max_probability": float(prob.amax(-1).mean()),
                "shared_mass": float(prob[..., -2:].sum(-1).mean()),
                "uniform_shared_mass": float((2 / count).mean()),
                "row_mass": float(prob[..., :cols].sum(-1).mean()),
                "uniform_row_mass": float((allowed[:, :cols].sum(-1) / count).mean())}
        return hook

    handles = [layer.register_forward_pre_hook(layer_hook(i)) for i, layer in enumerate(module.layers)]
    def output_hook(_, inputs, output):
        if "residual" in stats:
            return
        features, _, _, src, indices = inputs
        b, r, c = indices
        chosen = (src[b, r, c].isnan() & (r >= query_start)).nonzero().flatten()[:256]
        if chosen.numel():
            x = features.reshape(-1, features.shape[-1])[chosen].float()
            y = output.reshape(-1, output.shape[-1])[chosen].float()
            stats["residual"] = {"queries": len(chosen), "base_rms": float(x.square().mean().sqrt()),
                                 "query_rms": float(y.square().mean().sqrt()),
                                 "rms_ratio": float(y.norm() / x.norm().clamp_min(1e-12))}
    handles.append(module.register_forward_hook(output_hook))
    return stats, handles


def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.4)
    # Match the existing benchmark process's default SDPA backend selection.
    torch.backends.cuda.enable_cudnn_sdp(True)
    sys.path.insert(0, str(BENCH / "code"))
    import eval_tabimpute as bench
    root = REPO / "runs/information_flow_diagnostics/cell_query_duration_audit_20260910"
    root.mkdir(parents=True, exist_ok=True)
    with np.load(bench.DATA / "mteb.matrix.npz", allow_pickle=True) as data:
        B, O, names = data["B"].astype(float), data["O"].astype(float), data["benchmark_names"]
    folds = bench.cv_folds(len(B), 10, bench.SEED)
    prepared = []
    for i in range(10):
        ti, vi = bench.cv_train_val(folds, i, .1, len(B), bench.SEED)
        R, mean, std = bench.estimate_corr(B[ti], O[ti])
        selections = {
            "mi": bench.greedy_mi(R, 15, benchmark_names=names)["selected"][:5],
            "random": bench.random_select(R, 15, seed=(bench.SEED + 7, i, 1000),
                                          benchmark_names=names)["selected"][:5]}
        prepared.append((ti, vi, mean, std, selections))
    trials = [("query2", "selected_check", "best_validation_checkpoint.pth", False, True)]
    trials += [(tag, str(step), f"checkpoint_{step}.pth", False, False)
               for step in (12000, 24000, 36000) for tag in ("query2", "control")]
    trials += [("query2", "selected_no_query", "best_validation_checkpoint.pth", True, False)]
    trials += [("query2", f"{s}_no_query", f"checkpoint_{s}.pth", True, False) for s in (12000, 36000)]
    trials += [("query2", f"{s}_testrow_flow", f"checkpoint_{s}.pth", False, True) for s in (12000, 36000)]
    trials += [("query2", "selected_testrow_flow", "best_validation_checkpoint.pth", False, True)]
    results = {}
    for tag, label, filename, disabled, diagnostic in trials:
        key = f"{tag}_{label}"
        write_json(root / "status.json", {"state": "evaluating", "active": key, "updated_unix": time.time()})
        path = root / f"{key}.json"
        if path.exists():
            results[key] = json.loads(path.read_text())
            continue
        work = WORK / f"tabimpute-cell_query_20260909_checked-{tag}"
        cfg = json.loads((work / "config.json").read_text())
        imputer = ImputePFN(device="cuda", checkpoint_path=str(work / filename), entry_wise_features=False,
            json_config={"config": cfg["model"]}, model_family="rowcol_svt_hybrid",
            precision_mode="amp_bfloat16", prediction_stat="mean")
        imputer.model.use_cell_query_decoder = not disabled and tag == "query2"
        flow, handles = capture_query_flow(imputer.model, min(len(prepared[0][0]), 3800)) if diagnostic else ({}, [])
        records = []
        for i, (ti, vi, mean, std, selections) in enumerate(prepared):
            for method, selected in selections.items():
                score = bench.impute_tabimpute(imputer, B[ti], O[ti], B[vi], O[vi], selected, mean, std, B.shape[1])
                if not np.isfinite(score["r2"]):
                    raise RuntimeError(f"Nonfinite score {key} {i} {method}")
                records.append({"fold": i, "method": method, **score})
            print(key, i, records[-2:], flush=True)
            if diagnostic:
                for row in records:
                    if filename == "best_validation_checkpoint.pth":
                        with np.load(BENCH / "logs/tabimpute_cell_query_20260909_checked_query2_mteb.npz") as reference:
                            expected = float(reference[f"{row['method']}_r2"][i, 4])
                    else:
                        step = filename.removeprefix("checkpoint_").removesuffix(".pth")
                        reference = json.loads((root / f"query2_{step}.json").read_text())
                        expected = next(r["r2"] for r in reference["folds"] if r["fold"] == i and r["method"] == row["method"])
                    if abs(row["r2"] - expected) > 1e-5:
                        raise RuntimeError(f"Scoring parity failed: {row} vs {expected}")
                break
        for handle in handles:
            handle.remove()
        result = {"checkpoint": str(work / filename), "disabled_query": disabled, "folds": records,
                  "scores": {m: float(np.mean([r["r2"] for r in records if r["method"] == m]))
                             for m in ("mi", "random")}, "flow": flow}
        write_json(path, result)
        results[key] = result
        write_json(root / "matched_mteb.json", results)
        del imputer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    write_json(root / "status.json", {"state": "complete", "updated_unix": time.time()})


if __name__ == "__main__":
    main()
