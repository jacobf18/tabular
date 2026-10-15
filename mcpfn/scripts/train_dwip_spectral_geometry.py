"""Fine-tune unchanged DWIP with general spectral geometry and support augmentation."""
import argparse
from functools import partial
import sys

import train_direct_pure_rabel_curriculum as base
from general_spectral_geometry import install_geometry_generator
from spectral_support_normalization import apply_shrunk_support_mask
from tabimpute.model.model_new_stable import TabImputeModelStable


class DwipTrainingAdapter(TabImputeModelStable):
    def forward(self, src, return_embeddings=False, decode_mask=None, prior_index=None):
        # The generic trainer supplies prior labels; DWIP never receives them.
        return super().forward(src, return_embeddings=return_embeddings, decode_mask=decode_mask)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--support-variance-df", type=float, required=True)
    args, rest = parser.parse_known_args()
    if args.support_variance_df <= 0:
        raise ValueError("support-variance-df must be positive")
    install_geometry_generator()
    base.TabImputeModelPureRaBEL = DwipTrainingAdapter
    base.MODEL_CONFIG = {}
    base.MODEL_FAMILY = "new_stable"
    base.apply_support_mask = partial(apply_shrunk_support_mask, variance_df=args.support_variance_df)
    sys.argv = [sys.argv[0], *rest]
    base.main()


if __name__ == "__main__":
    main()
