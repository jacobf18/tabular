"""Pinned acquisition and orchestration for arXiv:2511.04833."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request


COMMIT = "d8eae280b19baaa36fa56c5f539033bfdb7ae701"
REPOSITORY = "https://github.com/ChristopheMuller/benchmark"
SHEET = ("https://docs.google.com/spreadsheets/d/"
         "1rFnJkfpF-YfK04uGa-IzjzZYy-czLQZEiiLOF4hr3_w/gviz/tq"
         "?tqx=out:csv&sheet=Cleaned%20Methods%20-%20ALL")
REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "runs/benchmark_evaluations/grzesiak_20260925"
RSCRIPT = REPO / ".benchmark-envs/grzesiak-r/bin/Rscript"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(data)
        temporary.replace(path)
    return {"url": url, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def acquire(work):
    tree_path = work / "upstream_tree.json"
    records = [download(f"https://api.github.com/repos/ChristopheMuller/benchmark/git/trees/{COMMIT}?recursive=1", tree_path)]
    tree = json.loads(tree_path.read_text())
    if tree["sha"] != COMMIT or tree.get("truncated"):
        raise ValueError("Incomplete or wrong upstream tree")
    selected = [entry for entry in tree["tree"] if entry["type"] == "blob" and (
        entry["path"].startswith(("R/", "data/", "python/", "latex/", "results/", "supplementary/"))
        or entry["path"] in ("README.md", "renv.lock", "_targets.R", "env.def", "python_requirements.txt", "requirements.txt"))]

    def fetch_entry(entry):
        record = download(f"https://raw.githubusercontent.com/ChristopheMuller/benchmark/{COMMIT}/{urllib.parse.quote(entry['path'])}",
                          work / "upstream" / entry["path"])
        data = Path(record["path"]).read_bytes()
        git_digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if git_digest != entry["sha"]:
            raise ValueError(f"Upstream blob hash mismatch: {entry['path']}")
        record["git_blob_sha1"] = git_digest
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        records.extend(executor.map(fetch_entry, selected))
    records.append(download(SHEET, work / "method_metadata.csv"))
    records.append(download("https://arxiv.org/html/2511.04833v1", work / "paper.html"))
    helper = download("https://raw.githubusercontent.com/BioGenies/imputomics/6943055e14277596fb2a3030ef94dfed46beaefd/R/utils.R",
                      work / "dependency_sources/imputomics/utils.R")
    records.append(helper)
    write_json(work / "dependency_sources/provenance.json", helper | {
        "commit": "6943055e14277596fb2a3030ef94dfed46beaefd", "use": "unchanged helpers for upstream aregImpute wrappers"})
    write_json(work / "acquisition.json", {"repository": REPOSITORY, "commit": COMMIT,
               "paper": "https://arxiv.org/abs/2511.04833", "files": records,
               "metadata_caveat": "Google Sheet is a retrieval-time snapshot, not versioned with the upstream commit.",
               "license_caveat": "No repository license found; upstream files remain local, not vendored into our paper repository."})
    print(f"Verified {len(selected)} pinned upstream files", flush=True)


def recover(work, rscript):
    subprocess.run([str(rscript), "--vanilla", str(Path(__file__).with_name("grzesiak_recover.R")), str(work)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "recover"))
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--rscript", type=Path, default=RSCRIPT)
    arguments = parser.parse_args()
    if arguments.command == "acquire":
        acquire(arguments.work)
    else:
        recover(arguments.work, arguments.rscript)


if __name__ == "__main__":
    main()
