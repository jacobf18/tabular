import test_single_model_full_network_cap90_trial as predecessor


SOURCE = predecessor.SOURCE
for old, new in {
    "import train_single_matrixformer_full_network_cap90 as training": "import train_single_matrixformer_full_network_cap90_v2 as training",
    "831100901, 832100901": "851100901, 852100901",
    "841100901, 842100901": "861100901, 862100901",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_actual_trainer_uses_strict_allocator_only_verifier_and_binds_its_test():
    assert "V36 is a separately registered fresh-seed scientific trial" in training.POLICY
    assert "Preserve V35's original failed registration-entry validation" in training.POLICY
    assert training.verify_registration.__globals__["cap90_recipe"] is training.cap90_recipe
    assert 'config["base"] != cap90_recipe.expected_base(' in training.ADAPTED_SOURCE
    assert 'cap90_recipe.validate_base(config["base"]' in training.ADAPTED_SOURCE
    assert '"tests/test_single_model_full_network_cap90_recipe.py"' in training.REGISTER_SOURCE
    assert training.require_evidence.__globals__["failed_entry_evidence"] is training.failed_entry_evidence
    assert training.REGISTER_SOURCE.count('base["memory_fraction"] = .9') == 1
    assert training.PARENT == training.predecessor.PARENT
    assert training.TRIAL_QUALIFICATION == training.predecessor.TRIAL_QUALIFICATION
    assert training.BOUNDARIES == training.predecessor.BOUNDARIES
    before = training.original.alignment.fixed.seeds_in([training.predecessor.batches(), training.predecessor.confirmation_cases()])
    after = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    assert len(after) == 3868 and not before.intersection(after)


def entry_fixture(tmp_path, monkeypatch):
    directory = tmp_path / "previous"
    directory.mkdir()
    producer = tmp_path / "producer"
    producer.mkdir()
    base = {"memory_fraction": .8, "objective": "preserved"}
    training.write_json(producer / "config.json", base)
    training.write_json(directory / "config.json", {"base": training.cap90_recipe.expected_base(base), "input_sha256": {}, "source_sha256": {}})
    digest = training.sha256(directory / "config.json")
    training.write_json(directory / "registration.json", {"config_sha256": digest})
    training.write_json(directory / "status.json", {"state": "failed", "pid": 2147483000,
        "error": "ValueError('Never change the registered decoder continuation or its original V8 qualification baseline')"})
    (directory / "train_and_audit.log").write_text("preserved failure")
    for boundary in training.predecessor.BOUNDARIES:
        destination = tmp_path / f"validation_{boundary:03d}"
        destination.mkdir()
        (directory / destination.name).symlink_to(destination, target_is_directory=True)
    monkeypatch.setattr(training.predecessor, "OUTPUT", directory)
    monkeypatch.setattr(training.predecessor.original.alignment.fixed, "PRODUCER", producer)
    return directory


def test_failure_evidence_binds_only_the_stopped_unstarted_trial(tmp_path, monkeypatch):
    directory = entry_fixture(tmp_path, monkeypatch)
    evidence = training.failed_entry_evidence()
    assert set(evidence) == {str(directory / name) for name in ("config.json", "registration.json", "status.json", "train_and_audit.log")}


@pytest.mark.parametrize("artifact", ["training.jsonl", "validation_0.json", "parameter_inventory.json", "training_batches", "latest_checkpoint.pth", "selected_checkpoint.pth"])
def test_any_existing_scientific_artifact_rejects_entry_failure_interpretation(tmp_path, monkeypatch, artifact):
    directory = entry_fixture(tmp_path, monkeypatch)
    (directory / artifact).touch()
    with pytest.raises(ValueError, match="before model setup"):
        training.failed_entry_evidence()


def test_even_one_prediction_in_a_reserved_directory_rejects_entry_only_failure(tmp_path, monkeypatch):
    directory = entry_fixture(tmp_path, monkeypatch)
    (directory / "validation_000/prediction.npz").touch()
    with pytest.raises(ValueError, match="remain empty"):
        training.failed_entry_evidence()
