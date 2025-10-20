# TabImpute

This is the official repository of TabImpute, a transformer-based architecture for missing data imputation on tabular data.

## Installation

```bash
cd mcpfn
pip install -e .
```

## Usage

```python
from tabimpute.interface import ImputePFN, MCTabPFNEnsemble
import numpy as np

imputer = ImputePFN(device='cpu') # cuda if available
ensemble_imputer = MCTabPFNEnsemble(device='cpu') # cuda if available

X = np.random.rand(5, 5)
print("Original X:")
print(X)
X[np.random.rand(*X.shape) < 0.1] = np.nan
print('X with NaNs:')
print(X)

out1 = imputer.impute(X.copy())
out2 = ensemble_imputer.impute(X.copy())
print(out1)
print(out2)
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