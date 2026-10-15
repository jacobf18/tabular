"""Train generic spectral geometry with observed-only support shrinkage."""
from general_spectral_geometry import install_geometry_generator
from train_spectral_support_shrinkage import main


if __name__ == "__main__":
    install_geometry_generator()
    main()
