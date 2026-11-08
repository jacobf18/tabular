"""Freeze previously unevaluated time windows without changing methods or masks' geometry."""

import argparse
import json
from pathlib import Path

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as evaluation


def plans():
    italian_targets = (0, 2352, 3024, 3360, 4032, 5040, 5712, 6384, 7056, 7728, 8400, 8736)
    italian_donors = (336, 672, 1008, 1344, 1680, 4368) * 2
    beijing_targets = (26304, 27648, 28992, 30336, 31680, 33024)
    beijing_donors = (2016, 2352, 2688, 3024, 3360, 3696)
    return {
        'air_quality': {'confirmation': [(target, donor, 3696) for target, donor in zip(italian_targets, italian_donors)]},
        'beijing': {'confirmation': [(target, donor, 7728) for target, donor in zip(beijing_targets, beijing_donors)]},
    }


def register(args):
    original_plans = evaluation.PLANS
    replication_plans = plans()
    excluded = []
    for dataset, splits in replication_plans.items():
        prior_targets = [target for cases in original_plans[dataset].values() for target, _, _ in cases]
        for target, _, _ in splits['confirmation']:
            assert all(abs(target - previous) >= evaluation.WINDOW for previous in prior_targets)
        frame, targets, _ = evaluation.load_data(dataset, args.data)
        qualified = []
        for starts in splits['confirmation']:
            try:
                _, details = evaluation.make_case(frame, targets, starts)
                if min(details['target_cells'], details['validation_target_cells']) < 30:
                    raise ValueError('Fewer than 30 measured target or validation cells')
            except ValueError as error:
                excluded.append(dict(dataset=dataset, starts=starts, reason=str(error)))
                continue
            qualified.append(starts)
        splits['confirmation'] = qualified
    evaluation.PLANS = replication_plans
    evaluation.register(args)
    protocol_path = args.output / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['source_sha256'][str(Path(__file__))] = shared.sha256(__file__)
    protocol['phase'] = 'Replication registered after initial confirmation; original models, grids, and endpoints unchanged'
    protocol['mask_timing'] = 'Independent recorded donor dates; some Italian mask donors postdate the reference window, but only Boolean masks are transferred'
    protocol['initial_run'] = str(evaluation.ROOT / 'outage_replay_v2')
    protocol['pre_scoring_exclusions'] = excluded
    shared.write_json(protocol_path, protocol)
    (args.output / 'replication_source.py').write_bytes(Path(__file__).read_bytes())
    print('Pre-scoring coverage exclusions:', excluded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=evaluation.ROOT / 'data')
    parser.add_argument('--datasets', nargs='+', default=list(evaluation.PLANS), choices=list(evaluation.PLANS))
    register(parser.parse_args())
