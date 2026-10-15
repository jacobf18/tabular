import json

import pytest
import torch

from dwip_teacher_mean_horizon import HORIZON_STEPS, horizon_recipe, verify_horizon_only_change
from dwip_teacher_mean_training import base, build_training_prior, control_recipe, sample_training_batch


def test_one_horizon_factor_preserves_the_completed_mean_huber_recipe():
    control, candidate = control_recipe(), horizon_recipe()
    assert {name for name in vars(control) if getattr(control, name) != getattr(candidate, name)} == {"phases_json", "decay_steps"}
    assert base.parse_phases(candidate)[0].end_step == HORIZON_STEPS == 128000
    assert HORIZON_STEPS < 200000
    assert candidate.point_loss_weight == 2 and candidate.cdf_loss_weight == .05
    assert candidate.nll_loss_weight == 1 and candidate.point_loss_kind == "huber"
    assert candidate.seed == 1201 and candidate.data_seed == 1150924


@pytest.mark.parametrize("field,value", (("lr", 2e-5), ("seed", 1202), ("data_seed", 7),
                                        ("point_loss_weight", 4.), ("support_variance_df", 2.),
                                        ("warmup_steps", 1200), ("decay_steps", 63400)))
def test_unrelated_learning_changes_are_rejected(field, value):
    control, candidate = control_recipe(), horizon_recipe()
    setattr(candidate, field, value)
    with pytest.raises(ValueError):
        verify_horizon_only_change(control, candidate)


def test_new_shapes_are_not_hidden_in_the_horizon_change():
    control, candidate = control_recipe(), horizon_recipe()
    phases = json.loads(candidate.phases_json)
    phases[0]["cols_high"] = 200
    candidate.phases_json = json.dumps(phases)
    with pytest.raises(ValueError, match="shapes"):
        verify_horizon_only_change(control, candidate)


def test_cosine_schedule_keeps_warmup_and_reaches_same_floor_at128k():
    args = horizon_recipe()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=args.lr)
    scheduler = base.make_scheduler(optimizer, warmup_steps=args.warmup_steps, stable_steps=args.stable_steps,
                                   decay_steps=args.decay_steps, min_lr_ratio=args.min_lr_ratio)
    scale = scheduler.lr_lambdas[0]
    assert scheduler.total_steps == HORIZON_STEPS
    assert scale(0) == 1 / 600 and scale(599) == 1.
    assert scale(127999) == scale(128000) == .05
    assert .05 < scale(64000) < 1.


def test_output_paths_can_change_without_changing_learning():
    control, candidate = control_recipe(), horizon_recipe()
    candidate.run_name = "new-owned-output"
    candidate.work_dir = "new-owned-work"
    candidate.fixed_validation_cache = "unchanged-shared-case-content"
    verify_horizon_only_change(control, candidate)


def test_sampled_training_batches_do_not_depend_on_phase_end_step():
    control, candidate = control_recipe(), horizon_recipe()
    device = torch.device("cpu")
    before, after = (build_training_prior(args, device=device) for args in (control, candidate))
    for step in range(1, 5):
        original = sample_training_batch(before, control, step, device)
        changed = sample_training_batch(after, candidate, step, device)
        for expected, actual in zip(original[:3], changed[:3]):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0., equal_nan=True)
        assert original[3] == changed[3]
