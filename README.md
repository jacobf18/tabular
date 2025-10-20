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
