import time
import numpy as np
import pandas as pd

# Libraries to benchmark
try:
    from fancyimpute import SoftImpute
    from hyperimpute.plugins.imputers import Imputers
except ImportError as e:
    print(f"Missing library: {e}")
    print("Please install via: pip install fancyimpute hyperimpute")
    exit()

def generate_data(rows=1000, cols=50, missing_rate=0.4):
    """Generates a random matrix with missing values."""
    np.random.seed(42)
    X = np.random.randn(rows, cols)
    mask = np.random.rand(rows, cols) < missing_rate
    X_miss = X.copy()
    X_miss[mask] = np.nan
    return X_miss

def benchmark():
    # 1. Setup Data
    ROWS, COLS = 2000, 50
    print(f"Generating data ({ROWS}x{COLS}) with 40% missing values...")
    X_miss = generate_data(rows=ROWS, cols=COLS)
    
    # ---------------------------------------------------------
    # Benchmark FancyImpute
    # ---------------------------------------------------------
    print("\nRunning FancyImpute (SoftImpute)...")
    start_time = time.time()
    
    # FancyImpute returns a numpy array
    solver_fancy = SoftImpute(verbose=False)
    X_filled_fancy = solver_fancy.fit_transform(X_miss)
    
    fancy_duration = time.time() - start_time
    print(f"FancyImpute Time: {fancy_duration:.4f} seconds")

    # ---------------------------------------------------------
    # Benchmark HyperImpute
    # ---------------------------------------------------------
    print("\nRunning HyperImpute (SoftImpute Plugin)...")
    start_time = time.time()
    
    # HyperImpute initialization
    # We specifically call the 'softimpute' plugin to make it a fair comparison
    # (The default HyperImpute learner uses a different ensemble approach)
    solver_hyper = Imputers().get("softimpute")
    
    # HyperImpute can handle numpy, but often prefers DataFrames internally
    X_filled_hyper = solver_hyper.fit_transform(X_miss.copy())
    
    hyper_duration = time.time() - start_time
    print(f"HyperImpute Time: {hyper_duration:.4f} seconds")

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print("-" * 30)
    if fancy_duration < hyper_duration:
        print(f"Winner: FancyImpute is {hyper_duration / fancy_duration:.2f}x faster.")
    else:
        print(f"Winner: HyperImpute is {fancy_duration / hyper_duration:.2f}x faster.")

if __name__ == "__main__":
    benchmark()