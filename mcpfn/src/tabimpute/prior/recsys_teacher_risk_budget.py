import math

import torch

from tabimpute.prior.recsys_rating_episode import expected_macro_rmse


def teacher_risk_budget_anchor(prediction, episode, teacher_prediction, mae_weight=.1,
                               macro_query_scope='parent'):
    if episode.metadata.get('gradient_data') != 'synthetic':
        raise ValueError('Risk-budget anchoring requires a synthetic episode')
    if not math.isfinite(mae_weight) or mae_weight <= 0:
        raise ValueError('Both rating risks require a positive finite MAE weight')
    if macro_query_scope not in ('parent', 'decoded'):
        raise ValueError('Unknown risk-budget macro query scope')
    if prediction.ndim != 1 or prediction.numel() != int(episode.query.sum()):
        raise ValueError('Student predictions must align with all synthetic queries')
    selected = episode.rating_query[episode.query]
    if not selected.any():
        raise ValueError('Require exposed synthetic rating queries')
    student = prediction[selected]
    if teacher_prediction is None or teacher_prediction.shape != student.shape:
        raise ValueError('Require aligned synthetic-teacher rating predictions')
    if teacher_prediction.requires_grad or teacher_prediction.device != student.device:
        raise ValueError('Synthetic-teacher predictions must be detached on the student device')
    if teacher_prediction.dtype != student.dtype:
        raise ValueError('Teacher and student rating predictions must share dtype')
    if not torch.isfinite(prediction).all() or not torch.isfinite(teacher_prediction).all():
        raise ValueError('Require finite student and synthetic-teacher predictions')
    if episode.ordinal_probabilities is None:
        raise ValueError('Require synthetic ordinal probabilities')
    probabilities = episode.ordinal_probabilities[episode.query][selected]
    if probabilities.shape != (student.numel(), 5) or probabilities.requires_grad:
        raise ValueError('Require detached five-rating synthetic probabilities')
    if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
            or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=1e-5, rtol=1e-5)):
        raise ValueError('Invalid synthetic ordinal probabilities')
    users = episode.query.nonzero()[:, 1][selected]
    num_users = episode.source.shape[1]
    counts = torch.bincount(users, minlength=num_users)
    weights = 1 / counts[users].to(student.dtype)
    effective_counts = episode.global_query_counts if macro_query_scope == 'parent' else None
    if macro_query_scope == 'parent' and effective_counts is None:
        raise ValueError('Parent risk scope requires synthetic global query counts')
    seed = episode.metadata['seed'] + 7919
    student_rmse = expected_macro_rmse(student, probabilities, users, num_users, seed,
                                      effective_counts=effective_counts)
    teacher_rmse = expected_macro_rmse(teacher_prediction, probabilities, users, num_users, seed,
                                      effective_counts=effective_counts)
    values = torch.arange(1, 6, device=student.device, dtype=student.dtype)
    student_mae = (((student[:, None] - values).abs() * probabilities).sum(-1) * weights).sum() / (counts > 0).sum()
    teacher_mae = (((teacher_prediction[:, None] - values).abs() * probabilities).sum(-1) * weights).sum() / (counts > 0).sum()
    excess_rmse = torch.nn.functional.relu(student_rmse - teacher_rmse)
    excess_mae = torch.nn.functional.relu(student_mae - teacher_mae)
    loss = excess_rmse + mae_weight * excess_mae
    return loss, {'teacher_anchor_mode': 'macro_risk_budget',
                  'student_expected_macro_rmse': student_rmse.item(),
                  'teacher_expected_macro_rmse': teacher_rmse.item(),
                  'student_expected_macro_mae': student_mae.item(),
                  'teacher_expected_macro_mae': teacher_mae.item(),
                  'positive_excess_rmse': excess_rmse.item(), 'positive_excess_mae': excess_mae.item(),
                  'mae_weight': mae_weight, 'macro_query_scope': macro_query_scope,
                  'shared_rmse_sampling_seed': seed}
