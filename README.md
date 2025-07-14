# MCPFN (Matrix Completion Prior-Fitted Network)

This is a PyTorch implementation of MCPFN, a transformer-based architecture for matrix completion on tabular data.

## Installation

```bash
pip install -e .
```

## Usage

```python
from mcpfn.interface import ImputePFN
import numpy as np

imputer = ImputePFN(device='cpu', # 'cuda' if you have a GPU
                    encoder_path='./src/mcpfn/model/encoder.pth', # Path to the encoder model
                    borders_path='./borders.pt', # Path to the borders tensor
                    checkpoint_path='./stage1/checkpoint/test.ckpt') # Path to the checkpoint file

X = np.random.rand(5, 5)
print(X)
X[np.random.rand(*X.shape) < 0.1] = np.nan
print(X)

out = imputer.impute(X)
print(out)
```