# Runtime Benchmark

This folder contains scripts and results for benchmarking the runtime performance of the old and new models across different matrix sizes.

## Files

- **`get_runtime_models.py`**: Script to run benchmarks comparing old and new models
- **`runtime_benchmark_results.csv`**: CSV file containing all runtime measurements
- **`plot_runtime_benchmark.py`**: Script to generate visualization plots from the CSV results

## Usage

### Running the Benchmark

```bash
cd runtime_benchmark
python get_runtime_benchmark.py
```

This will:
- Test matrices with fixed columns (10) and exponentially growing rows (10, 20, 40, 80, 160, 320, 640)
- Run each configuration 5 times to reduce noise
- Output results to `runtime_benchmark_results.csv`

### Generating Plots

```bash
python plot_runtime_benchmark.py
```

This will:
- Read the CSV file
- Generate a plot comparing old vs new model runtimes
- Show error bars based on standard deviation across runs
- Display speedup annotations
- Save the plot as `runtime_benchmark.png`

## CSV Format

The CSV file contains the following columns:
- `model_type`: "new" or "old"
- `num_rows`: Number of rows in the test matrix
- `num_cols`: Number of columns (always 10)
- `run_number`: Run number (1-5)
- `runtime`: Runtime in seconds

All individual runtimes are included so that standard deviations and other statistics can be calculated later.
