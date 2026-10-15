import pytest

import replay_single_model_point_inclusive_teachers as replay
import train_single_matrixformer_point_inclusive as training
from test_single_model_training_teacher_replay import row_fixture


def test_new_replay_is_bound_to_the_new_trial_and_original_policy(tmp_path):
    assert replay.training is training
    assert replay.POLICY == replay.baseline.POLICY
    row, arguments = row_fixture(tmp_path)
    replay.check_row(row, *arguments)
    arguments[4]["teacher"][0, 0, 0] = 2
    with pytest.raises(ValueError):
        replay.check_row(row, *arguments)


@pytest.mark.parametrize("state,step,pid,count", [
    ("training", 64, 999999999, 1792), ("training_complete", 63, 999999999, 1792),
    ("training_complete", 64, 1, 1792), ("training_complete", 64, 999999999, 1791)])
def test_no_live_or_partial_teacher_replay(monkeypatch, state, step, pid, count):
    monkeypatch.setattr(replay, "read", lambda path: {"state": state, "step": step, "pid": pid})
    with pytest.raises(ValueError):
        replay.require_finished_training({"batches": [{"tasks": [None] * count}]})
