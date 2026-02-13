"""
Benchmark script to compare runtime of old and new models across different matrix sizes.
Tests matrices with fixed columns (10) and exponentially growing rows.
Each configuration is run 5 times to reduce noise.
"""

import time
import numpy as np
import csv
import torch
from tabimpute.interface import ImputePFN

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

def benchmark_model(imputer: ImputePFN, X: np.ndarray, model_name: str, num_runs: int = 5):
    """Run imputation multiple times and return all runtimes."""
    runtimes = []
    
    for run_num in range(num_runs):
        # Warmup run (not counted)
        if run_num == 0:
            _ = imputer.impute(X.copy(), return_full=False)
            torch.cuda.synchronize() if hasattr(torch.cuda, 'synchronize') else None
        
        # Actual timed run
        torch.cuda.synchronize() if hasattr(torch.cuda, 'synchronize') else None
        start_time = time.time()
        _ = imputer.impute(X.copy(), return_full=False)
        torch.cuda.synchronize() if hasattr(torch.cuda, 'synchronize') else None
        end_time = time.time()
        
        runtime = end_time - start_time
        runtimes.append(runtime)
        print(f"  {model_name} - Run {run_num + 1}/{num_runs}: {runtime:.4f} seconds")
    
    return runtimes

def main():
    # Initialize models
    print("Initializing models...")
    new_model = ImputePFN(
        device='cuda',
        entry_wise_features=False,
        checkpoint_path='/home/jacobf18/tabular/mcpfn/src/tabimpute/workdir/tabimpute-mcar_p0.4-num_cls_8-rank_1_11/checkpoint_60000.pth'
    )
    
    old_model = ImputePFN(
        device='cuda',
        entry_wise_features=True
    )
    
    print(f"New Model size: {sum(p.numel() for p in new_model.model.parameters()):,}")
    print(f"Old Model size: {sum(p.numel() for p in old_model.model.parameters()):,}")
    print()
    
    # Define matrix sizes: fixed columns (10), exponentially growing rows
    num_cols = 10
    # Start with 10 rows, double each time up to a reasonable maximum
    row_sizes = [10, 25, 50, 100, 250, 500]
    
    # Results storage
    results = []
    
    # Run benchmarks
    for num_rows in row_sizes:
        print(f"\n{'='*60}")
        print(f"Testing matrix size: {num_rows} x {num_cols}")
        print(f"{'='*60}")
        
        # Create test matrix (use same seed for both models to ensure fair comparison)
        X = create_test_matrix(num_rows, num_cols, missing_rate=0.1, seed=42)
        
        # Benchmark new model
        print(f"\nNew Model ({num_rows}x{num_cols}):")
        new_runtimes = benchmark_model(new_model, X, "New Model", num_runs=5)
        
        # Benchmark old model
        print(f"\nOld Model ({num_rows}x{num_cols}):")
        old_runtimes = benchmark_model(old_model, X, "Old Model", num_runs=5)
        
        # Store results
        for run_num, runtime in enumerate(new_runtimes, 1):
            results.append({
                'model_type': 'new',
                'num_rows': num_rows,
                'num_cols': num_cols,
                'run_number': run_num,
                'runtime': runtime
            })
        
        for run_num, runtime in enumerate(old_runtimes, 1):
            results.append({
                'model_type': 'old',
                'num_rows': num_rows,
                'num_cols': num_cols,
                'run_number': run_num,
                'runtime': runtime
            })
        
        # Print summary for this size
        avg_new = np.mean(new_runtimes)
        avg_old = np.mean(old_runtimes)
        speedup = avg_old / avg_new if avg_new > 0 else 0
        print(f"\nSummary for {num_rows}x{num_cols}:")
        print(f"  New Model - Avg: {avg_new:.4f}s, Std: {np.std(new_runtimes):.4f}s")
        print(f"  Old Model - Avg: {avg_old:.4f}s, Std: {np.std(old_runtimes):.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Write results to CSV
    output_file = 'runtime_benchmark_results.csv'
    print(f"\n{'='*60}")
    print(f"Writing results to {output_file}...")
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['model_type', 'num_rows', 'num_cols', 'run_number', 'runtime']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to {output_file}")
    print(f"Total entries: {len(results)}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("Overall Summary:")
    print(f"{'='*60}")
    
    for num_rows in row_sizes:
        new_times = [r['runtime'] for r in results if r['model_type'] == 'new' and r['num_rows'] == num_rows]
        old_times = [r['runtime'] for r in results if r['model_type'] == 'old' and r['num_rows'] == num_rows]
        
        if new_times and old_times:
            avg_new = np.mean(new_times)
            avg_old = np.mean(old_times)
            speedup = avg_old / avg_new if avg_new > 0 else 0
            print(f"{num_rows:4d}x{num_cols:2d}: New={avg_new:6.4f}s, Old={avg_old:6.4f}s, Speedup={speedup:5.2f}x")

if __name__ == "__main__":
    main()
