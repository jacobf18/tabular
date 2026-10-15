import numpy as np
import pytest
import torch

from diagnose_dwip_latent_retention import masked_case, prepare_cases, score_prediction, summarize
from run_dwip_spectral_geometry import build_commands


def test_retention_cases_share_shapes_masks_and_preserve_rng(tmp_path):
    _, _, prior, _ = build_commands(tmp_path)
    torch.manual_seed(92)
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    cases = prepare_cases(prior, 1, 123)
    assert len(cases) == 8
    assert torch.equal(torch.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    repeated = prepare_cases(prior, 1, 123)
    for first, second in zip(cases, repeated):
        assert first['id'] == second['id']
        for key in ('source', 'target', 'mask', 'score_scale'):
            torch.testing.assert_close(first[key], second[key], equal_nan=True)
        assert not (first['mask'] & torch.isfinite(first['source'])).any()
    for offset in (0, 4):
        latent, latent_support, spectral, spectral_support = cases[offset:offset + 4]
        assert latent['source'].shape == spectral['source'].shape
        assert torch.equal(torch.isfinite(latent['source']), torch.isfinite(spectral['source']))
        assert torch.equal(latent_support['mask'], spectral_support['mask'])
        assert not (torch.isfinite(latent_support['source']) & ~torch.isfinite(latent['source'])).any()


def test_retention_scales_and_support_do_not_depend_on_hidden_targets():
    observation = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    keep = observation.remainder(3) != 0
    before = masked_case(observation, observation, keep, 'broad_support', 57)
    changed = masked_case(observation, observation + 100000, keep, 'broad_support', 57)
    torch.testing.assert_close(before['source'], changed['source'], equal_nan=True)
    torch.testing.assert_close(before['score_scale'], changed['score_scale'])
    assert torch.equal(before['mask'], changed['mask'])


def test_retention_scores_are_unclipped_and_only_use_missing_targets():
    observation = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    keep = observation.remainder(2) == 0
    case = {**masked_case(observation, observation, keep, 'mcar', 1), 'id': 'case', 'group': 'group'}
    exact = score_prediction(case, observation.numpy())
    assert exact['sse'] == 0
    prediction = observation.numpy().copy()
    prediction[keep.numpy()] += 100000
    assert score_prediction(case, prediction)['sse'] == 0
    prediction[~keep.numpy()] += 1000
    assert score_prediction(case, prediction)['sse'] > 10000
    prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        score_prediction(case, prediction)


def test_retention_summary_distinguishes_case_macro_and_cell_weighted_error():
    result = summarize([{'group': 'group', 'n': 1, 'sse': 1., 'raw_sse': 1.},
                        {'group': 'group', 'n': 3, 'sse': 27., 'raw_sse': 27.}])['group']
    assert result['macro_case_rmse'] == 2
    assert result['pooled_rmse'] == pytest.approx(np.sqrt(7))
    assert result['case_count'] == 2
