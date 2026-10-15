"""Generic spectral matrices with diverse orthonormal singular-vector geometry."""
import numpy as np

import tabimpute.prior.spectrum_training_set_generation as spectrum


class GeometrySpectrumGenerator(spectrum.DecayingSpectrumGenerator):
    def _sample_orthonormal_matrix(self, n_rows, rank):
        kind = self.config.get("spectrum_basis_kind", "haar")
        if kind == "mixed":
            kind = str(np.random.choice(("haar", "coherent", "smooth", "offset")))
        if kind == "haar":
            return super()._sample_orthonormal_matrix(n_rows, rank)
        if kind == "coherent":
            strength = float(np.random.uniform(.25, 2.))
            row_scale = np.exp(np.random.normal(0., strength, (n_rows, 1)))
            features = row_scale * np.random.standard_normal((n_rows, rank))
        elif kind == "smooth":
            dimensions = int(np.random.randint(1, 6))
            positions = np.random.standard_normal((n_rows, dimensions))
            bandwidth = float(np.exp(np.random.uniform(np.log(.3), np.log(3.))))
            frequencies = np.random.standard_normal((dimensions, rank)) / bandwidth
            phases = np.random.uniform(0., 2 * np.pi, (1, rank))
            features = np.cos(positions @ frequencies + phases)
            # Keep the sampled basis full rank even with very smooth latent functions.
            features += .03 * np.random.standard_normal((n_rows, rank))
        elif kind == "offset":
            features = np.random.standard_normal((n_rows, rank))
            features[:, 0] = 1. + .05 * features[:, 0]
        else:
            raise ValueError(f"Unknown spectral basis geometry: {kind}")
        q, r = np.linalg.qr(features, mode="reduced")
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.
        return q * signs


def install_geometry_generator():
    """Opt in within an isolated process; default configurations retain Haar bases."""
    spectrum.DecayingSpectrumGenerator = GeometrySpectrumGenerator
