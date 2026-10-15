from copy import deepcopy

import pytest
import torch

import preflight_single_model_general_imputation_teacher as preflight


def test_full_batch_retains_completed_v26_shapes_masses_and_teachers_with_fresh_seeds():
    actual = preflight.batch()
    previous = preflight.preceding.training.batches()[0]
    assert len(actual["tasks"]) == len(previous["tasks"]) == 28
    for task, original in zip(actual["tasks"], previous["tasks"]):
        normalized = deepcopy(task)
        normalized["gradient_seed"] = original["gradient_seed"]
        normalized["specification"].get("structured", normalized["specification"])["seed"] = (
            original["specification"].get("structured", original["specification"])["seed"])
        assert normalized == original
    seeds = preflight.training.alignment.fixed.seeds_in(actual)
    assert len(seeds) == 56
    assert not seeds.intersection(preflight.training.alignment.fixed.seeds_in(previous))
    assert not seeds.intersection(preflight.training.alignment.fixed.seeds_in(preflight.preceding.training.confirmation_cases()))
    roles = [preflight.expected_role(task) for task in actual["tasks"]]
    assert roles.count("general") == 26
    assert roles.count("initialization") == 0 and roles.count("specialist") == 2
    assert preflight.PARENT == preflight.preceding.training.OUTPUT / "selected_checkpoint.pth"
    assert preflight.RATE == 1e-5


def test_execution_and_audit_bind_existing_top_two_blocks_without_mutating_registered_sources():
    source = preflight.ADAPTED_SOURCE
    assert 'view = top_two.TopTwoBlockParameterView(model)' in source
    assert 'with top_two.capture_coordinates(model, view):' in source
    assert source.count('row[0].startswith(PREFIXES)') == 2
    assert source.count('frozen = frozen_inventory(') == 2
    assert 'paired.protected_update(' in source and 'paired.check_endpoint(' in source
    assert 'hashes.update(audit_raw(payload, raw_path))' in source
    assert 'teacher_evidence = audit_teacher_replay(output, config)' in source
    assert '"selection_qualified": False' in source
    assert preflight.PREFIXES == ('transformer_blocks.6.', 'transformer_blocks.7.', 'decoder.')
    assert preflight.template.PARENT == preflight.training.alignment.fixed.PARENT
    assert preflight.inspect.unwrap(preflight.replay_targets).__globals__ is preflight.__dict__
    assert preflight.audit_teacher_replay.__globals__ is preflight.__dict__


def inventory():
    current = {f'transformer_blocks.7.weight_{index:02d}': torch.ones(2) for index in range(60)}
    current.update({'decoder.weight': torch.ones(3), 'transformer_blocks.6.weight': torch.ones(2), 'transformer_blocks.5.weight': torch.zeros(2),
                    'other_buffer': torch.zeros(2),
                    'transformer_blocks.7.self_attention_between_features.rope.inv_freq': torch.zeros(2)})
    mutable = {name for name in current if name.startswith(preflight.PREFIXES) and not name.endswith('inv_freq')}
    return current, mutable


@pytest.mark.parametrize('buffer_name', ['other_buffer', 'transformer_blocks.7.self_attention_between_features.rope.inv_freq'])
def test_frozen_inventory_tracks_both_mutable_modules_and_preserves_signed_zero(monkeypatch, buffer_name):
    initial, mutable = inventory()
    monkeypatch.setattr(preflight, 'read', lambda path: [[name] for name in mutable])
    current = deepcopy(initial)
    current['transformer_blocks.7.weight_00'].add_(1)
    current['decoder.weight'].add_(1)
    result = preflight.frozen_inventory(initial, current, mutable)
    assert result['frozen_tensors'] == 3
    assert result['changed_trainable_tensors'] == ['transformer_blocks.7.weight_00', 'decoder.weight']
    current[buffer_name][0] = -0.
    with pytest.raises(ValueError, match='Frozen parameter or buffer'):
        preflight.frozen_inventory(initial, current, mutable)


@pytest.mark.parametrize('mutation', ['omitted', 'wrong_scope', 'shape', 'nonfinite'])
def test_invalid_state_inventory_cannot_pass(monkeypatch, mutation):
    initial, mutable = inventory()
    registered = [[name] for name in mutable]
    monkeypatch.setattr(preflight, 'read', lambda path: registered)
    current = deepcopy(initial)
    if mutation == 'omitted':
        current.pop('decoder.weight')
    elif mutation == 'wrong_scope':
        mutable.remove('decoder.weight')
        mutable.add('other_buffer')
    elif mutation == 'shape':
        current['decoder.weight'] = torch.ones(4)
    else:
        current['decoder.weight'][0] = float('nan')
    with pytest.raises(ValueError):
        preflight.frozen_inventory(initial, current, mutable)


def test_raw_scratch_evidence_preserves_every_gradient_and_exact_proposal_alias(tmp_path, monkeypatch):
    output, scratch = tmp_path / 'canonical', tmp_path / 'scratch'
    output.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(preflight, 'OUTPUT', output)
    monkeypatch.setattr(preflight, 'SCRATCH', scratch)
    gradients = torch.arange(84 * 8, dtype=torch.float32).reshape(84, 8)
    gradients[0, 0] = -0.
    payload = {'gradients': gradients, 'snapshot': {'task_gradients': gradients[:28].clone()},
               'record': {'composite_tasks': 28, 'protected_tasks': 84}, 'batch': {'tasks': [None] * 28}}
    path = output / 'raw_update.pt'
    digest = preflight.save_raw(payload, path)
    restored = torch.load(path, weights_only=True, mmap=True)
    assert digest == preflight.sha256(path)
    assert preflight.composite.byte_equal(restored['gradients'], gradients)
    assert path.is_symlink() and path.resolve() == scratch / path.name
    evidence = preflight.audit_raw(restored, path)
    preflight.verify_hashes(evidence)
    with pytest.raises(FileExistsError):
        preflight.save_raw(payload, path)
    restored['snapshot']['task_gradients'] = restored['snapshot']['task_gradients'].clone()
    with pytest.raises(ValueError):
        preflight.audit_raw(restored, path)


@pytest.mark.parametrize('filesystem,reserve,budget', [('checkpoint', 18, 1), ('root', 6, 0), ('scratch', 50, 16)])
def test_capacity_checks_all_reserves_before_any_work(monkeypatch, filesystem, reserve, budget):
    paths = {'checkpoint': preflight.ROOT, 'root': preflight.Path('/'), 'scratch': preflight.SCRATCH.parent}
    free = {path: 100 * 1024**3 for path in paths.values()}
    free[paths[filesystem]] = (reserve + budget) * 1024**3 - 1
    monkeypatch.setattr(preflight.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': free[path]})())
    with pytest.raises(OSError, match=f'{reserve} GiB reserve'):
        preflight.require_space()
    free[paths[filesystem]] += 1
    preflight.require_space()


def predecessor():
    result = {"complete": True, "gate": {"passed": False, "confirmation_evaluated": True},
              "scientific_stages_repeated": False, "campaign": None, "review": None}
    audit = {"complete": True, "complete_training_audit": True, "through_step": 64, "accepted_updates": 64,
             "best": {"step": 64}, "training_episodes": 1792, "prediction_grids": 1216, "top_two_block_training": True,
             "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
             "artifact_sha256": {str(preflight.PARENT): preflight.PARENT_SHA256}}
    replay = {"complete": True, "endpoint_grids": 56, "selected_grids": 152}
    confirmation = {"complete": True, "cases": 284, "prediction_grids": 1136}
    return [result, audit, replay, confirmation]


@pytest.mark.parametrize("record,path,value", [
    (0, ("complete",), False), (0, ("gate", "passed"), True),
    (0, ("gate", "confirmation_evaluated"), False), (0, ("scientific_stages_repeated",), True),
    (0, ("campaign",), "unexpected"), (0, ("review",), "unexpected"),
    (1, ("through_step",), 50), (1, ("accepted_updates",), 63), (1, ("best", "step"), 50),
    (1, ("training_episodes",), 1791), (1, ("prediction_grids",), 1064), (1, ("top_two_block_training",), False),
    (1, ("independently_regenerated_training_tasks",), 1791),
    (1, ("independently_regenerated_rating_contexts",), 127),
    (1, ("artifact_sha256", str(preflight.PARENT)), "substituted"),
    (2, ("endpoint_grids",), 55), (2, ("selected_grids",), 151),
    (3, ("cases",), 283), (3, ("prediction_grids",), 1135),
])
def test_only_the_complete_audited_rejected_v26_lifecycle_can_initialize_the_probe(record, path, value):
    reports = predecessor()
    preflight.validate_predecessor(*reports, preflight.PARENT_SHA256)
    current = reports[record]
    for field in path[:-1]:
        current = current[field]
    current[path[-1]] = value
    with pytest.raises(ValueError, match="audited but unqualified V26"):
        preflight.validate_predecessor(*reports, preflight.PARENT_SHA256)


def test_parent_hash_is_not_an_unqualified_teacher_or_replacement_reference():
    with pytest.raises(ValueError):
        preflight.validate_predecessor(*predecessor(), "different-checkpoint")
    source = preflight.ADAPTED_SOURCE
    assert 'teacher_policy.identities(config["base"], training.alignment.fixed.PARENT)' in source
    assert 'config["parent_selection_qualified"] is not False' in source
    assert 'config["retention_baseline_for_future_training"] != str(training.alignment.fixed.PARENT)' in source
    assert 'config["input_sha256"][str(PARENT)] != PARENT_SHA256' in source
    assert preflight.PARENT != preflight.training.alignment.fixed.PARENT


@pytest.mark.parametrize("domain,structured,role", [("imputation", False, "general"),
    ("imputation", True, "general"), ("recsys", False, "specialist")])
def test_independent_audit_checks_actual_new_teacher_policy_and_original_identity(domain, structured, role):
    specification = {"domain": domain}
    if structured:
        specification["structured"] = {"family": "mcar"}
    task = {"specification": specification}
    config = {"teacher_identities": {name: {"checkpoint": name, "sha256": name + "-hash"}
                                    for name in ("initialization", "general", "specialist")}}
    routing = {"specialist_weight": float(role == "specialist"), "target": role,
               "policy": preflight.teacher_policy.POLICY, "teacher_identity": config["teacher_identities"][role]}
    assert preflight.check_routing({"routing": routing}, task, config) == role
    for field, value in {"target": "initialization", "teacher_identity": {"checkpoint": str(preflight.PARENT)},
                         "policy": preflight.template.teacher_policy.POLICY}.items():
        with pytest.raises(ValueError, match="original role and checkpoint"):
            preflight.check_routing({"routing": {**routing, field: value}}, task, config)


def test_only_teacher_policy_initializer_and_fresh_streams_change_not_optimizer_or_audit_math():
    import ast

    actual_tree = ast.parse(preflight.ADAPTED_SOURCE)
    expected_source = preflight.template.ADAPTED_SOURCE.replace(
        '# Final-two-block plus decoder optimizer preflight',
        '# General-imputation teacher optimizer preflight',
    )
    original_tree = ast.parse(expected_source)
    for name in ("run", "audit", "predictions", "batch"):
        actual = next(node for node in actual_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        original = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual) == ast.dump(original)
    assert preflight.DATA_SEED == 581100901 and preflight.GRADIENT_SEED == 582100901
    original_policy = {"legacy": "initialization", "structured": "general", "recsys": "specialist"}
    assert preflight.template.teacher_policy.POLICY == original_policy
    assert preflight.teacher_policy.POLICY == {**original_policy, "legacy": "general"}
