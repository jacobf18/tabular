# MatrixFormer research branch snapshot

## Relationship to main

`main` commit `2cf54ac` contains the published MatrixFormer and MatrixFormer+
runtime, synthetic training utilities, and evaluation entry points. The
`train_kc` branch retains the broader research history and experimental code.
Its additional single-model experiments are research snapshots, not a replacement
for the frozen experts used in the paper.

An isolated checkout of `main` was checked on 2026-10-15 using both original
checkpoints on CPU. The observed-error blend completed a synthetic 12-by-9 matrix
with 30 hidden entries, returned finite predictions, and preserved observed
values. Both checkpoint hashes matched their registered values. This is an
inference smoke test, not a new benchmark score. It required no source files from
the working `train_kc` checkout and performed no training. The H200 was left to
its existing workload.

The published main snapshot also passed 259 selected unit tests. On this
research branch, the focused runtime, prior, and evaluation suite passed 257
tests on CPU, with one CUDA BF16 parity test skipped. The additional research
archive contains experiment-specific tests that can require local checkpoints,
generated caches, and registered run artifacts; it should not be interpreted as
an independently validated model release.

## Files that explain the final models

- `AUTORESEARCH_TIMELINE.md`: chronology, rejected alternatives, synthetic priors,
  objective changes, and selection of the final general model.
- `mcpfn/DWIP_NEW_MODEL_SUMMARY.md`: main-stage results and provenance.
- `mcpfn/STREAMED_TRAINING.md`: large-matrix continuation and memory handling.
- `mcpfn/DWIP_RECSYS_RESULT.md`: the synthetic-trained ratings specialist.
- `mcpfn/OBSERVED_ROUTER.md` and `mcpfn/MATRIXFORMER_PLUS_RESULTS.md`: the frozen
  blend, observed-error fitting, and comparisons.
- `mcpfn/MATCHED_LLM_EVALUATION.md`, `mcpfn/SYNTHETIC_EVALUATIONS.md`, and
  `mcpfn/ORIENTATION_EVALUATION.md`: evaluation protocols and limitations.
- `mcpfn/SINGLE_MODEL_GOAL.md`: the separate, ongoing single-checkpoint research
  journal. Its experimental candidates do not replace the frozen two-expert blend.
- `benchmark-selection/logs/results_summary_r2_at_5.md`: historical baseline
  scores, retained alongside the original local evaluation queue scripts. These
  are earlier experiments rather than the current paper's matched LLM results.

## External artifacts still required

The general and specialist checkpoint tensors remain outside Git. Their paths,
hashes, and shared architecture are recorded in
`mcpfn/evaluation_protocols/matrixformer_frozen_artifacts.json`. The `model` object
can be supplied as `json_config={"config": model_config}` when constructing an
`ImputePFN` directly. Historical evaluation scripts also read the adjacent full
`config.json` files, so preserve those with the checkpoint files.

Benchmark datasets, archived predictions, validation caches, and training logs
are also external artifacts. A clean source checkout does not by itself
reproduce a full historical evaluation or training campaign. Registered trainers
check source and artifact hashes; copied historical paths require the matching
inputs, while changed protocols require a new registration.

The small command recipes already published on main are retained explicitly in
this branch. No checkpoint tensors, private Jellyfish panel, conversation dumps,
dataset ZIPs, or generated prediction archives are included in this snapshot.

Only selected benchmark-selection helpers and local queue recipes are vendored.
The legacy queues can require additional upstream scripts and benchmark data
from `https://github.com/smolix/benchmark-selection.git`, revision
`f339e73fead83337dba21ab78518df3e1fd40a05`. Obtain that checkout separately if
reproducing those historical queues; it is not required for frozen MatrixFormer
inference. Queue paths and old process IDs describe the original environment
and should be reviewed before execution.

## Baseline compatibility

The local ReMasker and DiffPuter fixes are preserved as patches under
`mcpfn/benchmark/patches/`, with exact upstream commits and application commands.
The third-party submodule pointers are unchanged. Their already-applied local
patches remain visible as dirty submodules; no commits are pushed to third-party
remotes.

## Branch environment

This research branch retains its earlier package dependencies and experiment
layout. From `mcpfn/`, install the package and optional evaluation tools as needed:

```bash
pip install -e . tabpfn_extensions cornac pytest
export PYTHONPATH="src:scripts:../benchmark-selection/code"
python -m pytest tests/test_observed_router.py tests/test_streamed_inference.py \
  tests/test_synthetic_low_rank_benchmark.py tests/test_matched_llm.py
```

The latest main branch's leaner package imports and reorganized benchmark CLI
remain on main. This snapshot preserves the research working tree rather than
merging those layout changes into active experiments.
