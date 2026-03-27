# TabImpute

[![arXiv](https://img.shields.io/badge/arXiv-2510.02625-b31b1b.svg)](https://arxiv.org/abs/2510.02625)
[![PyPI version](https://img.shields.io/pypi/v/tabimpute.svg)](https://pypi.org/project/tabimpute/)
[![Python versions](https://img.shields.io/pypi/pyversions/tabimpute.svg)](https://pypi.org/project/tabimpute/)

This is the official repository of TabImpute, a transformer-based architecture for missing data imputation on tabular data.

TabImpute is a pre-trained foundation model for filling in missing values in numerical and mixed-type tables with little or no task-specific tuning. The repository contains the published Python package, pretrained model loading from Hugging Face, benchmarking code, training utilities, and optional extensions for preprocessing, categorical handling, and test-time adaptation.

At a glance, this repo provides:

- `tabimpute`, a pip-installable library for zero-shot tabular imputation
- pretrained checkpoints and interfaces for both the original model and `TabImputeV2`
- benchmarking and plotting code for comparing imputation quality and runtime
- training scripts and utilities for reproducing or extending the models

TabImpute is designed to work as a practical imputation tool out of the box while still exposing lower-level hooks for advanced use cases such as custom preprocessing, postprocessing constraints, chunked inference on larger tables, and test-time training.

## Installation

Install from PyPI:

```bash
pip install tabimpute
```

Install from source (editable):

```bash
cd mcpfn
pip install -e .
```

Optional TabPFN extensions support:

```bash
pip install "tabimpute[tabpfn_extensions]"
# or from source:
pip install -e ".[tabpfn_extensions]"
```

Optional extras:

```bash
# Benchmark/plotting stack
pip install -e ".[benchmark]"

# Data generation/training stack
pip install -e ".[training]"

# Preprocessing and categorical helper utilities
pip install -e ".[preprocessing,categorical]"
```

## Usage

```python
from tabimpute.interface import ImputePFN
import numpy as np

imputer = ImputePFN(device='cpu') # cuda if available

X = np.random.rand(5, 5)
print("Original X:")
print(X)
X[np.random.rand(*X.shape) < 0.1] = np.nan
print('X with NaNs:')
print(X)

out1 = imputer.impute(X.copy())
print(out1)
```

## Citation

If you use TabImpute, please consider citing our paper:

```bibtex
@article{feitelberg2025tabimpute,
  title={TabImpute: Accurate and Fast Zero-Shot Missing-Data Imputation with a Pre-Trained Transformer},
  author={Feitelberg, Jacob and Saha, Dwaipayan and Choi, Kyuseong and Ahmad, Zaid and Agarwal, Anish and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2510.02625},
  year={2025}
}
```
