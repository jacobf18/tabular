"""Describe paired synthetic confirmation errors without selecting a real-benchmark model."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from recsys_evaluation_common import sha256, write_csv, write_json


def summarize(cases):
    groups = defaultdict(list)
    for case in cases:
        if case.get("gradient_data") != "synthetic":
            raise ValueError("Only synthetic confirmation cases may enter this diagnostic")
        domain = case["domain"]
        family = case["prior"] if domain == "imputation" else case["preference_family"]
        groups[domain, family].append(case)
        groups[domain, "ALL"].append(case)
    rows = []
    for (domain, family), selected in sorted(groups.items()):
        row = {"domain": domain, "family": family, "cases": len(selected)}
        for role in ("parent", "student", "teacher"):
            for metric in ("rmse", "mae"):
                row[f"{role}_{metric}"] = sum(case[role][metric] for case in selected) / len(selected)
                row[f"{role}_{metric}_wins_vs_parent"] = sum(case[role][metric] < case["parent"][metric] for case in selected)
        row["mean_specialist_weight"] = sum(case["routing"]["specialist_weight"] for case in selected) / len(selected)
        row["student_rmse_change_percent"] = 100 * (row["student_rmse"] / row["parent_rmse"] - 1)
        row["teacher_rmse_change_percent"] = 100 * (row["teacher_rmse"] / row["parent_rmse"] - 1)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source = arguments.confirmation.resolve() / "cases.jsonl"
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in source.read_text().splitlines()]
    rows = summarize(cases)
    write_csv(output / "by_family.csv", rows)
    write_json(output / "audit.json", {"input": str(source), "input_sha256": sha256(source),
                                       "source_sha256": sha256(Path(__file__)), "cases": len(cases),
                                       "real_data_used": False, "checkpoint_selection_performed": False,
                                       "caveat": "Descriptive paired errors on the existing confirmation bank; not a new independent confirmation or statistical significance test",
                                       "summary": [row for row in rows if row["family"] == "ALL"]})
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
