# TabImpute Pretraining: Decaying Spectral Structure

## Motivation

The current pretraining data generator produces matrices from a hard low-rank + noise model:

$$X = UV^\top + \varepsilon, \quad U \in \mathbb{R}^{m \times r},\ V \in \mathbb{R}^{n \times r}, \quad U_{ij}, V_{ij} \overset{iid}{\sim} \mathcal{N}(0,1)$$

When $U$ and $V$ have i.i.d. Gaussian entries, the singular values of $UV^\top$ follow the Marchenko-Pastur distribution — all singular values are roughly the same order of magnitude, producing a **hard spectral cliff**: $r$ large singular values followed by a sudden drop to the noise floor $\sigma_\varepsilon$.

This is a useful mathematical idealization but has two concrete problems:

1. **It is unrealistic.** Real-world matrices — genomics, recommendation systems, MRI k-space, financial returns — have smoothly decaying spectra with no clean signal/noise separation. Training on hard low-rank matrices creates a distribution shift relative to real data.

2. **It makes high-rank training ill-posed.** For a 50×50 matrix at $p_{\text{obs}} = 0.5$, the observed entries (1250) are insufficient to determine completions for rank > ~13 (degrees of freedom exceed observations). Training on underdetermined problems causes the model to learn that high-rank missing entries are unpredictable, converging to near-mean imputation rather than exploiting structure.

The fix is to replace the hard low-rank + noise DGP with a **smoothly decaying spectral model** that more closely resembles real data and allows the model to learn top-$k$ reconstruction even when the true underlying rank is higher.

---

## New Data Generating Process

### Structure

Generate matrices as:

$$X = U \cdot \text{diag}(\sigma) \cdot V^\top + \varepsilon$$

where:
- $U \in \mathbb{R}^{m \times r}$ has orthonormal columns (columns drawn from the Haar measure via QR decomposition of a random Gaussian matrix)
- $V \in \mathbb{R}^{n \times r}$ has orthonormal columns, drawn independently
- $\sigma \in \mathbb{R}^r$ is a **randomly sampled decaying spectrum** (see below)
- $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$ is i.i.d. noise

Using QR decomposition to generate $U, V$ (rather than raw Gaussian matrices) ensures the singular vectors are uniformly distributed on the Stiefel manifold, cleanly separating the geometry of the subspace from the magnitude of each component.

### Spectrum generation

Sample a new spectrum independently for each training example:

```python
def sample_spectrum(rank, rng):
    # Random power-law decay exponent
    alpha = rng.uniform(0.5, 2.0)

    # Base power-law decay
    spectrum = np.array([(i + 1) ** (-alpha) for i in range(rank)], dtype=np.float32)

    # Normalize so leading singular value = 1, then apply random overall scale
    spectrum /= spectrum[0]
    scale = rng.uniform(1.0, 5.0)
    spectrum *= scale

    return spectrum
```

This produces spectra ranging from **slowly decaying** (α ≈ 0.5, many components roughly equally important) to **rapidly decaying** (α ≈ 2.0, effectively low-rank despite high generative rank). The model is trained across this full range.

### Full matrix generator

```python
def make_matrix(m, n, rank, noise_std, rng):
    # Orthonormal factor matrices
    U = np.linalg.qr(rng.standard_normal((m, rank)))[0]  # (m, rank)
    V = np.linalg.qr(rng.standard_normal((n, rank)))[0]  # (n, rank)

    # Random decaying spectrum
    spectrum = sample_spectrum(rank, rng)

    # Low-rank signal
    X = (U * spectrum) @ V.T   # (m, n)

    # Additive noise
    X += noise_std * rng.standard_normal((m, n))

    return X
```

### Observation mask

Keep $p_{\text{obs}}$ fixed at 0.5 (unchanged). The spectral decay means the effective reconstruction target is the **top-$k$ approximation** rather than the full rank-$r$ matrix, making the task well-posed even when $r$ is large — the tail components are below the noise floor and should not be reconstructed.

### Hyperparameter ranges for pretraining mix

Sample the following per example:

| Parameter | Distribution |
|-----------|-------------|
| `rank` | `Uniform({5, 6, ..., 30})` |
| `alpha` (decay exponent) | `Uniform(0.5, 2.0)` |
| `scale` (leading singular value) | `Uniform(1.0, 5.0)` |
| `noise_std` | `Uniform(0.05, 0.5)` |
| `m, n` | From existing matrix size distribution |

---

## What the Model Should Learn

With this DGP, the NLL objective naturally encourages the model to:

1. **Identify the top-$k$ singular components** that are above the noise floor
2. **Reconstruct those components** accurately for missing entries
3. **Output high-entropy predictions** for directions that are below the noise floor — correctly treating them as unrecoverable

This is the behavior you want at test time: given a real matrix with a smoothly decaying spectrum, the model finds the recoverable structure and is appropriately uncertain about the rest. The model does not need to know $k$ explicitly — it emerges naturally from the ratio of singular values to noise level in the observed entries.

---

## Evaluation

### Benchmark change

The existing rank sweep benchmark (RMSE vs. true rank) should be supplemented with an **effective rank sweep**: instead of varying the true generative rank $r$, vary the **noise-to-signal ratio** at a fixed generative rank, controlling how many components are above the noise floor. This better reflects what the model is actually learning.

Concretely, fix $r = 20$ and sweep `noise_std` from 0.01 to 2.0. At low noise, all 20 components are recoverable. At high noise, only the top few are. Plot RMSE vs. number of components above noise floor for TabImpute vs. SoftImpute.

### Baselines

- `softimpute` (existing)
- `tabimpute` trained on old hard low-rank DGP (existing checkpoints)
- `tabimpute` trained on new decaying spectrum DGP (new)

### Expected behavior

The new model should match or exceed SoftImpute at all effective ranks, with the largest gains in the intermediate regime (effective rank 5–15) where the decaying spectrum makes the problem well-posed but non-trivial.

---

## Implementation Notes

- The change is **entirely in the data generator**. No changes to model architecture, loss function, or training loop are required.
- The old hard low-rank DGP can be recovered by setting `alpha` to a very large value (e.g. 100), making all but the first singular value negligible. This is useful as a sanity check.
- Ensure the spectrum is sampled **per example**, not per batch, to maximize diversity of training signal.
- Log the sampled `alpha` and effective rank (number of singular values above `noise_std`) as training metrics to verify the distribution is as expected.
