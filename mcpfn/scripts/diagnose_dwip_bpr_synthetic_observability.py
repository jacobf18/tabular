"""Summarize count references computed on the exact shared synthetic model inputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from evaluate_dwip_recsys import sha256, write_json


def run(args, reference):
    if args.output.exists():
        raise ValueError("Preserve existing synthetic diagnostic reports")
    if not reference["synthetic_only"] or reference["real_data_loaded"] or reference["neural_training"] or reference["gradients_enabled"] or not reference["model_states_unchanged"]:
        raise ValueError("Require a completed read-only synthetic model comparison")
    groups = defaultdict(list)
    for record in reference["cases"]:
        if "context_references" not in record:
            raise ValueError("References must be measured on the shared episodes, not regenerated afterward")
        groups[(record["seed"], record["mode"])].append(record)
    records = []
    for (seed, mode), group in groups.items():
        if len(group) != len(reference["averages"]) or any(record["shared_episode_sha256"] != group[0]["shared_episode_sha256"] or record["context_references"] != group[0]["context_references"] for record in group):
            raise ValueError("Count references did not share every model's exact synthetic episode")
        records.append({"seed": seed, "mode": mode, "shared_episode_sha256": group[0]["shared_episode_sha256"], **group[0]["context_references"]})
    if len(records) != 2 * len(reference["seeds"]):
        raise ValueError("Require both context modes for every registered synthetic seed")
    snapshot = args.output.parent / f"{args.output.stem}_source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), snapshot / Path(__file__).name)
    report = {"synthetic_only": True, "real_data_loaded": False, "neural_models_loaded": False,
              "reference": str(args.reference.resolve()), "reference_sha256": sha256(args.reference),
              "source_sha256": {str(Path(__file__).resolve()): sha256(Path(__file__))}, "cases": records,
              "interpretation": "Visible counts use only the synthetic input mask; global parent counts are an information-rich reference. Neither is a neural teacher, an inference blend, a Cornac metric, or a goal result."}
    write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    run(args, reference)


if __name__ == "__main__":
    main()
