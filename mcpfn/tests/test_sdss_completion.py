import numpy as np
import pytest

import evaluate_sdss_completion as experiment


@pytest.mark.parametrize('name', experiment.TRANSFORMS)
def test_transforms_keep_signed_flux_and_missingness(name):
    matrix = np.arange(-10., 20.).reshape(6, 5)
    matrix[0, 1] = np.nan
    values, inverse = experiment.transform_observed(matrix, name)
    np.testing.assert_allclose(inverse(values), matrix, atol=1e-12, equal_nan=True)


def test_development_and_confirmation_are_disjoint_and_keep_mask():
    data = dict(spectra=np.arange(12000.).reshape(12, 1000), mask=np.zeros((12, 1000), dtype=bool))
    data['mask'][0, 100] = True
    development = experiment.make_split(data, 'development', rows=5)
    confirmation = experiment.make_split(data, 'confirmation', rows=5)
    assert not set(development[2]) & set(confirmation[2])
    for truth, valid, rows, bands in (development, confirmation):
        np.testing.assert_array_equal(truth, data['spectra'][rows][:, bands])
        np.testing.assert_array_equal(valid, ~data['mask'][rows][:, bands])
