import test_single_model_all_imputation_retention_training as previous_tests
from single_model_registered_adaptation import replace_once


SOURCE = previous_tests.SOURCE
for old, new in {
    "single_model_all_imputation_retention_training": "single_model_cohort_anchor_retention_training",
    "preflight_single_model_all_imputation_retention": "preflight_single_model_cohort_anchor_retention",
    "test_single_model_all_imputation_retention_preflight as base_fixtures":
        "test_single_model_cohort_anchor_retention_preflight as base_fixtures",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, 'input_sha256={str(probe.PARENT): probe.PARENT_SHA256})',
    'input_sha256={identity["checkpoint"]: identity["sha256"] for identity in probe.reference_identities().values()},\n'
    '        retention_references=probe.reference_identities(), base={"parent_checkpoint": probe.reference_identities()["v8"]["checkpoint"],\n'
    '        "input_sha256": {identity["checkpoint"]: identity["sha256"] for identity in probe.reference_identities().values()}})')
SOURCE = replace_once(SOURCE, 'reference = torch.nn.Linear(2, 2).eval().requires_grad_(False)',
    'reference = {role: torch.nn.Linear(2, 2).eval().requires_grad_(False) for role in ("v8", "v29")}')
SOURCE = replace_once(SOURCE, 'assert model is reference and not torch.is_grad_enabled()',
    'assert any(model is current for current in reference.values()) and not torch.is_grad_enabled()')
SOURCE = replace_once(SOURCE, 'calls.append(values.clone())',
    'calls.append((next(role for role, current in reference.items() if model is current), values.clone()))')
exec(compile(SOURCE, __file__, "exec"), globals())


def test_full_trial_records_the_exact_cohort_assignments(tmp_path, monkeypatch):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    replay_all(state)
    hashes = state.finish()
    report = integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
    assert report["reference_role_counts"] == {"v8": 4, "v29": 22}
    assert [role for role, source in calls].count("v8") == 4
    assert [role for role, source in calls].count("v29") == 22
    assert report["reference"] == {role: {**identity, "state_tensors": 254, "bitwise_equal": True}
        for role, identity in probe.reference_identities().items()}
    assert integration.retention.BACKWARD_SOURCE_SHA256 == integration.previous.retention.BACKWARD_SOURCE_SHA256


@pytest.mark.parametrize("mutation", ["missing", "swapped", "digest", "unbound", "parent"])
def test_both_original_references_must_be_bound_in_full_trial_registration(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        config["retention_references"].pop("v8")
    elif mutation == "swapped":
        config["retention_references"]["v8"], config["retention_references"]["v29"] = (
            config["retention_references"]["v29"], config["retention_references"]["v8"])
    elif mutation == "digest":
        config["base"]["input_sha256"][config["base"]["parent_checkpoint"]] = "changed"
    elif mutation == "unbound":
        config["input_sha256"].pop(config["base"]["parent_checkpoint"])
    else:
        config["parent_checkpoint"] = "/not-the-registered-parent.pth"
    with pytest.raises(ValueError):
        integration.identity(config)
    assert not calls


@pytest.mark.parametrize("mutation", ["role", "payload_role", "payload_identity", "aliased_models", "changed_reference"])
def test_full_replay_never_swaps_or_mutates_references(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    if mutation == "aliased_models":
        references["v8"] = references["v29"]
        with pytest.raises(ValueError):
            integration.RetentionReplay(training, config, counts, references, None, backend)
        return
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    path = Path(next(iter(state.tasks)))
    payload = torch.load(path, weights_only=False)
    task, decision, folder = state.tasks[str(path)]
    if mutation == "role":
        decision["reference_role"] = "v29"
    elif mutation.startswith("payload_"):
        reference_path = Path(decision["artifact"])
        cached = torch.load(reference_path, weights_only=False)
        if mutation == "payload_role":
            cached["reference_role"] = "v29"
        else:
            cached["identity"] = probe.reference_identities()["v29"]
        torch.save(cached, reference_path)
        decision["sha256"] = probe.sha256(reference_path)
    else:
        replay_all(state)
        original = integration.retention.verify_model
        monkeypatch.setattr(integration.retention, "verify_model", lambda model, identity:
            {**original(model, identity), "bitwise_equal": model is not references["v29"]})
        with pytest.raises(ValueError):
            state.finish()
        return
    with pytest.raises(ValueError):
        state.check(path, payload)
    assert not state.summary_path.exists()


@pytest.mark.parametrize("mutation", ["role", "count", "missing_reference"])
def test_qualification_requires_cohort_bindings_as_well_as_full_grid_hashes(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    replay_all(state)
    hashes = state.finish()
    report = probe.read(state.summary_path)
    if mutation == "role":
        rows = [json.loads(line) for line in state.rows_path.read_text().splitlines()]
        rows[0]["reference_role"] = "v29"
        state.rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes[str(state.rows_path)] = probe.sha256(state.rows_path)
        report["artifact_sha256"][str(state.rows_path)] = probe.sha256(state.rows_path)
    elif mutation == "count":
        report["reference_role_counts"]["v8"] = 3
    else:
        report["reference"].pop("v29")
    probe.write_json(state.summary_path, report)
    hashes[str(state.summary_path)] = probe.sha256(state.summary_path)
    with pytest.raises(ValueError):
        integration.require_report(training, {"artifact_sha256": hashes, "admission_attempts": counts})
