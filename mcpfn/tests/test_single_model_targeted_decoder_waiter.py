from pathlib import Path

import pytest

from recover_single_model_targeted_decoder_waiter import validate_command


def test_live_binding_requires_the_exact_existing_training_invocation():
    validate_command(["python", "-u", "/model/train.py", "--stage", "train"], Path("/model/train.py"))


@pytest.mark.parametrize("arguments", [
    ["python", "/model/other.py", "--stage", "train"],
    ["python", "/model/train.py", "--stage", "register"],
    ["python", "/model/train.py", "--stage", "train", "--output", "/elsewhere"],
    ["python", "/model/train.py"],
    ["python", "/model/train.py", "--stage"],
])
def test_rejects_other_processes_or_outputs(arguments):
    with pytest.raises(ValueError):
        validate_command(arguments, Path("/model/train.py"))
