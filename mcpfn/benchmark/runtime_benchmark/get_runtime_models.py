"""
Benchmark script to compare runtime of old and new models across different matrix sizes.
Tests matrices with fixed columns (10) and exponentially growing rows.
Each configuration is run 5 times to reduce noise.
Includes standard imputation and TTT (test-time training) variants for both models.
"""

import os
import time
import numpy as np
import csv
import torch
from tabimpute.interface import ImputePFN
from tabimpute.tabimpute_v2 import TabImputeV2

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
    "tabimpute",
    "workdir",
    "tabimpute-mcar_p0.4-num_cls_12-rank_1_11",
    "checkpoint_85000.pth",
)


def create_test_matrix(num_rows: int, num_cols: int, missing_rate: float = 0.1, seed: int = None):
    """Create a test matrix with missing values."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random data
    X = np.random.randn(num_rows, num_cols)
    
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-16)
    
    # Introduce missing values
    X[np.random.rand(*X.shape) < missing_rate] = np.nan
    
    return X

def benchmark_model(imputer, X: np.ndarray, model_name: str, num_runs: int = 5):
    """Run standard imputation multiple times and return all runtimes."""
    runtimes = []
    for run_num in range(num_runs):
        if run_num == 0:
            _ = imputer.impute(X.copy(), return_full=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        _ = imputer.impute(X.copy(), return_full=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        runtime = end_time - start_time
        runtimes.append(runtime)
        print(f"  {model_name} - Run {run_num + 1}/{num_runs}: {runtime:.4f} seconds")
    return runtimes


def benchmark_model_ttt(imputer, X: np.ndarray, model_name: str, k: int = 5, num_runs: int = 5):
    """Run TTT imputation multiple times and return all runtimes."""
    runtimes = []
    for run_num in range(num_runs):
        if run_num == 0:
            _ = imputer.impute_with_test_time_training(X.copy(), k=k, return_full=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        _ = imputer.impute_with_test_time_training(X.copy(), k=k, return_full=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        runtime = end_time - start_time
        runtimes.append(runtime)
        print(f"  {model_name} TTT k={k} - Run {run_num + 1}/{num_runs}: {runtime:.4f} seconds")
    return runtimes

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize models
    print("Initializing models...")
    checkpoint_path = os.environ.get("TABIMPUTE_CHECKPOINT", os.path.abspath(CHECKPOINT_PATH))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Set TABIMPUTE_CHECKPOINT env var to a valid path."
        )

    new_model = TabImputeV2(device=device, checkpoint_path=checkpoint_path)
    old_model = ImputePFN(device=device)

    print(f"New Model (TabImputeV2) size: {sum(p.numel() for p in new_model.model.parameters()):,}")
    print(f"Old Model (ImputePFN) size: {sum(p.numel() for p in old_model.model.parameters()):,}")
    print()

    num_cols = 10
    row_sizes = [10, 25, 50, 100, 250, 500]
    ttt_k = 5
    results = []

    for num_rows in row_sizes:
        print(f"\n{'='*60}")
        print(f"Testing matrix size: {num_rows} x {num_cols}")
        print(f"{'='*60}")

        X = create_test_matrix(num_rows, num_cols, missing_rate=0.1, seed=42)

        # Standard imputation
        print(f"\nNew Model ({num_rows}x{num_cols}):")
        new_runtimes = benchmark_model(new_model, X, "New Model", num_runs=5)
        print(f"\nOld Model ({num_rows}x{num_cols}):")
        old_runtimes = benchmark_model(old_model, X, "Old Model", num_runs=5)

        # TTT imputation
        print(f"\nNew Model TTT k={ttt_k} ({num_rows}x{num_cols}):")
        new_ttt_runtimes = benchmark_model_ttt(
            new_model, X, "New Model", k=ttt_k, num_runs=5
        )
        print(f"\nOld Model TTT k={ttt_k} ({num_rows}x{num_cols}):")
        old_ttt_runtimes = benchmark_model_ttt(
            old_model, X, "Old Model", k=ttt_k, num_runs=5
        )

        for run_num, runtime in enumerate(new_runtimes, 1):
            results.append({
                "model_type": "new",
                "mode": "standard",
                "num_rows": num_rows,
                "num_cols": num_cols,
                "run_number": run_num,
                "runtime": runtime,
            })
        for run_num, runtime in enumerate(old_runtimes, 1):
            results.append({
                "model_type": "old",
                "mode": "standard",
                "num_rows": num_rows,
                "num_cols": num_cols,
                "run_number": run_num,
                "runtime": runtime,
            })
        for run_num, runtime in enumerate(new_ttt_runtimes, 1):
            results.append({
                "model_type": "new",
                "mode": "ttt",
                "num_rows": num_rows,
                "num_cols": num_cols,
                "run_number": run_num,
                "runtime": runtime,
            })
        for run_num, runtime in enumerate(old_ttt_runtimes, 1):
            results.append({
                "model_type": "old",
                "mode": "ttt",
                "num_rows": num_rows,
                "num_cols": num_cols,
                "run_number": run_num,
                "runtime": runtime,
            })

        avg_new = np.mean(new_runtimes)
        avg_old = np.mean(old_runtimes)
        avg_new_ttt = np.mean(new_ttt_runtimes)
        avg_old_ttt = np.mean(old_ttt_runtimes)
        print(f"\nSummary for {num_rows}x{num_cols}:")
        print(f"  New (std):   Avg={avg_new:.4f}s")
        print(f"  Old (std):   Avg={avg_old:.4f}s")
        print(f"  New TTT:     Avg={avg_new_ttt:.4f}s")
        print(f"  Old TTT:     Avg={avg_old_ttt:.4f}s")
        if avg_new > 0:
            print(f"  Speedup std: {avg_old / avg_new:.2f}x")

    output_file = "runtime_benchmark_results.csv"
    print(f"\n{'='*60}")
    print(f"Writing results to {output_file}...")

    with open(output_file, "w", newline="") as f:
        fieldnames = ["model_type", "mode", "num_rows", "num_cols", "run_number", "runtime"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {output_file}")
    print(f"Total entries: {len(results)}")

    print(f"\n{'='*60}")
    print("Overall Summary:")
    print(f"{'='*60}")

    for num_rows in row_sizes:
        def get_times(mt, m):
            return [r["runtime"] for r in results if r["model_type"] == mt and r["mode"] == m and r["num_rows"] == num_rows]

        new_t = get_times("new", "standard")
        old_t = get_times("old", "standard")
        new_ttt_t = get_times("new", "ttt")
        old_ttt_t = get_times("old", "ttt")

        if new_t and old_t:
            print(f"{num_rows:4d}x{num_cols:2d}: New={np.mean(new_t):6.4f}s  Old={np.mean(old_t):6.4f}s  "
                  f"New_TTT={np.mean(new_ttt_t):6.4f}s  Old_TTT={np.mean(old_ttt_t):6.4f}s")

if __name__ == "__main__":
    main()
