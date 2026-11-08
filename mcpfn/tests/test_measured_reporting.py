import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from report_measured_physics import seed_summary


def test_intervals_average_conditions_before_seeds():
    frame = pd.DataFrame([dict(dataset='scene', method='method', rate=rate, seed=seed, nrmse=seed + rate)
                          for seed in (1, 2, 3) for rate in (.5, .8, .95)])
    summary = seed_summary(frame, ['dataset', 'method']).iloc[0]
    assert summary.seeds == 3
    np.testing.assert_allclose(summary['mean'], 2 + np.mean([.5, .8, .95]))
    np.testing.assert_allclose(summary.ci95_half_width, student_t.ppf(.975, 2) / np.sqrt(3))
