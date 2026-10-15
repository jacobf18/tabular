"""Unchanged DWIP trainer with explicit per-component latent rehearsal shapes."""
import train_direct_pure_rabel_curriculum as base
import train_dwip_spectral_geometry as dwip


class ShapeAwareMixedMatrixPrior(base.MixedMatrixPrior):
    def __init__(self, phase, device, missingness_type, mixed_config):
        super().__init__(phase=phase, device=device, missingness_type=missingness_type, mixed_config=mixed_config)
        specifications = [spec for spec in mixed_config["mixed_priors"] if float(spec.get("weight", 1.)) > 0]
        for child, specification in zip(self.children, specifications):
            dimensions = specification.get("shape_override")
            if dimensions is None:
                continue
            expected = {"num_rows_low", "num_rows_high", "num_cols_low", "num_cols_high"}
            if specification["kind"] != "latent_factor" or set(dimensions) != expected:
                raise ValueError("Explicit shape overrides are only supported for latent components")
            if any(type(value) is not int or value < 2 for value in dimensions.values()):
                raise ValueError("Shape bounds must be integer dimensions of at least two")
            if dimensions["num_rows_low"] > dimensions["num_rows_high"] or dimensions["num_cols_low"] > dimensions["num_cols_high"]:
                raise ValueError("Shape bounds must be ordered")
            child.config.update(dimensions)


def main():
    base.MixedMatrixPrior = ShapeAwareMixedMatrixPrior
    dwip.main()


if __name__ == "__main__":
    main()
