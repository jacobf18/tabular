"""Evaluate naturally missing numerical data with the pinned upstream Energy-I-Score."""

import argparse
import concurrent.futures
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np
import pandas as pd

from evaluate_dwip_missbench_uci import CHECKPOINT
from grzesiak_benchmark import COMMIT, RSCRIPT, WORK, sha256, write_json
from grzesiak_evaluate import CHECKPOINT_SHA256, stable_seed


def model_server(work, token):
    import torch
    from tabimpute.interface import ImputePFN

    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint changed")
    torch.manual_seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.mha.set_fastpath_enabled(False)
    config = json.loads((CHECKPOINT.parent / "config.json").read_text())
    model = ImputePFN(device="cuda", checkpoint_path=str(CHECKPOINT), entry_wise_features=False,
                      json_config={"config": config.get("model", config.get("config"))},
                      model_family="new_stable", precision_mode="amp_bfloat16", prediction_stat="mean",
                      max_num_rows=None, max_num_cols=None)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Authorization") != token:
                self.send_error(403)
                return
            try:
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                missing = np.asarray(json.loads(raw)["matrix"], dtype=np.float64)
                center, scale = np.nanmean(missing, axis=0), np.nanstd(missing, axis=0)
                scale = np.where(scale == 0, 1, scale)
                prediction = model.impute(((missing - center) / scale).copy()) * scale + center
                if not np.isfinite(prediction).all():
                    raise ValueError("nonfinite model output")
                payload = json.dumps({"matrix": prediction.tolist()}, allow_nan=False).encode()
                with (work / "natural/model_queries.jsonl").open("a") as stream:
                    stream.write(json.dumps({"input_sha256": hashlib.sha256(raw).hexdigest(),
                                              "output_sha256": hashlib.sha256(payload).hexdigest(),
                                              "shape": list(missing.shape), "missing": int(np.isnan(missing).sum())}) + "\n")
                self.send_response(200)
            except Exception as error:
                payload = str(error).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *arguments):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def evaluate(work, dataset, method, rscript, timeout, url, token):
    output = work / "natural" / dataset / method
    path = output.with_suffix(".json")
    source = work / "upstream/data/datasets/incomplete_backup/numerical" / f"{dataset}.RDS"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["input_sha256"] != sha256(source):
            raise ValueError("Natural input changed")
        return prior
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"dataset": dataset, "method": method, "input_sha256": sha256(source), "seed": stable_seed(f"natural/{dataset}"),
              "upstream_commit": COMMIT, "timeout_seconds": timeout, "runner_sha256": sha256(Path(__file__)),
              "bridge_sha256": sha256(Path(__file__).with_suffix(".R")), "source_status": "fresh_natural_missingness_evaluation"}
    if method == "matrixformer":
        record["checkpoint_sha256"] = CHECKPOINT_SHA256
    with tempfile.TemporaryDirectory(prefix="grzesiak-natural-", dir="/dev/shm") as temporary:
        temporary = Path(temporary)
        request = {"input": str(source), "output": str(temporary / "prediction.csv"), "result": str(temporary / "result.json"),
                   "columns": str(output.with_suffix(".columns.csv")), "upstream": str(work / "upstream"),
                   "method": method, "seed": record["seed"], "url": url, "token": token}
        write_json(temporary / "request.json", request)
        try:
            with output.with_suffix(".log").open("w") as log:
                process = subprocess.Popen([str(rscript), "--vanilla", str(Path(__file__).with_suffix(".R")), str(temporary / "request.json")],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                           env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise TimeoutError(f"Exceeded {timeout} seconds")
            if code != 0:
                raise RuntimeError(f"R exit {code}")
            record.update(json.loads((temporary / "result.json").read_text()))
            if record["status"] == "ok":
                prediction = pd.read_csv(temporary / "prediction.csv").to_numpy()
                np.savez_compressed(output.with_suffix(".npz"), prediction=prediction)
                record["prediction_sha256"] = sha256(output.with_suffix(".npz"))
        except Exception as error:
            record.update(status="timeout" if isinstance(error, TimeoutError) else "error", error=str(error))
    write_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--rscript", type=Path, default=RSCRIPT)
    parser.add_argument("--datasets", nargs="+", default=["diabetes", "globwarm", "oceanbuoys", "popmis", "pulplignin"])
    parser.add_argument("--methods", nargs="+", default=["matrixformer", "column_mean", "mice_pmm", "mice_cart", "mice_rf", "mice_drf"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=36000)
    arguments = parser.parse_args()
    (arguments.work / "natural").mkdir(exist_ok=True)
    token = secrets.token_hex(24)
    server = model_server(arguments.work, token) if "matrixformer" in arguments.methods else None
    url = f"http://127.0.0.1:{server.server_port}" if server else ""
    manifest = {"datasets": arguments.datasets, "methods": arguments.methods, "N": 20, "deterministic_N": 1,
                "scale": False, "row_split": False, "state": "running", "timeout": arguments.timeout}
    write_json(arguments.work / "natural/run.json", manifest)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            futures = [executor.submit(evaluate, arguments.work, dataset, method, arguments.rscript, arguments.timeout, url, token)
                       for dataset in arguments.datasets for method in arguments.methods]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(result["dataset"], result["method"], result["status"], result.get("energy_iscore", result.get("error")), flush=True)
    finally:
        if server:
            server.shutdown()
            server.server_close()
    manifest["state"] = "finished_all_requested_attempts"
    write_json(arguments.work / "natural/run.json", manifest)


if __name__ == "__main__":
    main()
