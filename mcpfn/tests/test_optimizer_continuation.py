from copy import deepcopy

import pytest
import torch

from optimizer_continuation import retain_adamw_state
from train_direct_pure_rabel_curriculum import make_scheduler


def setup():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.02)
    for _ in range(4):
        optimizer.zero_grad()
        model(torch.ones(2, 3)).square().sum().backward()
        optimizer.step()
    checkpoint = deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=4))
    new = torch.nn.Linear(3, 2)
    new.load_state_dict(checkpoint["model"])
    fresh = torch.optim.AdamW(new.parameters(), lr=6e-6, weight_decay=.02)
    scheduler = make_scheduler(fresh, warmup_steps=600, stable_steps=0,
                               decay_steps=23400, min_lr_ratio=.05)
    return model, optimizer, checkpoint, new, fresh, scheduler


def test_retains_history_and_new_schedule_with_exact_next_update():
    old, optimizer, checkpoint, new, fresh, scheduler = setup()
    lr = fresh.param_groups[0]["lr"]
    schedule = deepcopy(scheduler.state_dict())
    report = retain_adamw_state(fresh, checkpoint, new)
    assert report["source_epoch"] == 4
    assert fresh.param_groups[0]["lr"] == lr
    assert fresh.param_groups[0]["initial_lr"] == 6e-6
    assert scheduler.state_dict() == schedule
    optimizer.param_groups[0]["lr"] = lr
    for model, opt in ((old, optimizer), (new, fresh)):
        opt.zero_grad()
        model(torch.ones(2, 3)).square().sum().backward()
        opt.step()
    for a, b in zip(old.parameters(), new.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert fresh.state[b]["step"] == 5


@pytest.mark.parametrize("failure", ["shape", "order", "count", "hyperparameter", "empty"])
def test_rejects_incompatible_state(failure):
    _, _, checkpoint, new, fresh, _ = setup()
    if failure == "shape":
        checkpoint["optimizer"]["state"][0]["exp_avg"] = torch.zeros(5)
    elif failure == "order":
        checkpoint["model"] = dict(reversed(list(checkpoint["model"].items())))
    elif failure == "count":
        checkpoint["optimizer"]["param_groups"][0]["params"].pop()
    elif failure == "hyperparameter":
        checkpoint["optimizer"]["param_groups"][0]["betas"] = (.5, .9)
    else:
        checkpoint["optimizer"]["state"] = {}
    with pytest.raises(ValueError):
        retain_adamw_state(fresh, checkpoint, new)


def test_queue_changes_only_optimizer_initialization():
    from pathlib import Path
    from run_spectral_optimizer_continuation import build_commands
    from run_spectral_geometry_continuations import build_commands as control
    root = Path("/tmp/optimizer_continuation_test")
    _, original, prior, _ = control(root)
    _, new, new_prior, _ = build_commands(root)
    assert prior == new_prior
    command = list(new["retained1e5"])
    command.remove("--retain-init-optimizer-state")
    for flag in ("--run-name", "--init-model-checkpoint"):
        command[command.index(flag) + 1] = original["continue1e5"][original["continue1e5"].index(flag) + 1]
    assert command == original["continue1e5"]
