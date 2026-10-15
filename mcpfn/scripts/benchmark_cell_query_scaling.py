"""Full-model sparse-query scaling; not an all-missing-cell throughput claim."""
import argparse
import json
from pathlib import Path
import time

import torch

from run_rowcol_controlled_experiments import INIT, REPO, write_json
from tabimpute.model.model_rowcol_svt_hybrid import TabImputeModelRowColumnSVTHybrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", default="300,500,1000")
    parser.add_argument("--targets", type=int, default=1024)
    args = parser.parse_args()
    config = json.loads((INIT.parent / "config.json").read_text())["model"]
    config.update(svt_solver_precision="float32", svt_mixture_cv_reencode=True,
                  use_cell_query_decoder=True, cell_query_width=128, cell_query_layers=2,
                  cell_query_chunk_size=128)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(92)
    model = TabImputeModelRowColumnSVTHybrid(**config)
    state = torch.load(INIT, map_location="cpu", weights_only=True)["model"]
    loaded = model.load_state_dict(state, strict=False)
    assert not loaded.unexpected_keys
    assert all(k.startswith("cell_query_decoder.") for k in loaded.missing_keys)
    del state
    model.cuda().eval()
    stage = {"name": "not_started"}
    def tracked(function, name):
        def call(*positional, **keywords):
            stage["name"] = name
            return function(*positional, **keywords)
        return call
    for name in ("encode_latents", "_point_grid", "_decode_with_svt_features"):
        setattr(model, name, tracked(getattr(model, name), name))
    results = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for size in map(int, args.sizes.split(",")):
        torch.manual_seed(100 + size)
        src = torch.randn(1, size, 10, device="cuda") @ torch.randn(1, 10, size, device="cuda") / 10 ** .5
        src.masked_fill_(torch.rand_like(src) > .2, float("nan"))
        candidates = src.isnan().flatten().nonzero().flatten()
        chosen = candidates[torch.randperm(len(candidates), device="cuda")[:args.targets]]
        mask = torch.zeros_like(src, dtype=torch.bool).flatten()
        mask[chosen] = True
        mask = mask.reshape_as(src)
        for enabled in (False, True):
            model.use_cell_query_decoder = enabled
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            row = {"size": size, "query_decoder": enabled, "target_count": int(mask.sum()),
                   "chunk_size": 128, "observed_fraction": .2, "repeats": 2}
            timings = []
            try:
                for _ in range(2):
                    start = time.perf_counter()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(src, decode_mask=mask)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - start)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Nonfinite logits")
                    del logits
                row.update(seconds=timings, peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3, status="ok")
            except torch.cuda.OutOfMemoryError as error:
                row.update(status="out_of_memory", stage=stage["name"], error=str(error),
                           peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
                torch.cuda.empty_cache()
            results.append(row)
            write_json(out, {"scope": "Full encoder + solver + requested logits (target counts recorded); inference only",
                             "checkpoint": str(INIT), "config": config, "results": results})
            print(json.dumps(row), flush=True)
        del src, mask
    print(out, flush=True)


if __name__ == "__main__":
    main()
