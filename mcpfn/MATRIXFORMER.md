# MatrixFormer and MatrixFormer+

This directory contains the frozen-expert MatrixFormer inference, synthetic
training utilities, and evaluation code used by the MatrixFormer paper.
MatrixFormer+ is the observed-error convex blend of the general imputation
expert and the recommender specialist. It uses two frozen checkpoints rather
than a separately trained blend checkpoint.

## Entry points

- `src/tabimpute/interface.py`: `ImputePFN`, including the unchanged `new_stable`
  architecture, mean predictions, precision controls, and checkpoint loading.
- `src/tabimpute/routing.py`: `TwoExpertRouter(policy="observed_blend")`.
- `src/tabimpute/streamed_inference.py`: memory-bounded inference.
- `scripts/evaluate_two_expert_router.py`: tabular, ratings, and LLM comparisons.
- `scripts/evaluate_matched_llm.py`: source-paper-matched LLM evaluation.
- `scripts/evaluate_matrixformer_plus_synthetic.py`: synthetic low-rank evaluation.
- `scripts/evaluate_matrixformer_plus_large_recsys.py`: larger ratings panels.
- `scripts/evaluate_causal_matrixformer_plus.py`: causal completion comparisons.
- `scripts/evaluate_orientation_ensemble.py`: orientation controls.
- `scripts/train_dwip_teacher_mean_horizon.py`,
  `scripts/train_streamed_large_matrices.py`, and
  `scripts/dwip_recsys_experiment.py`: the synthetic training procedures.

The detailed protocols and existing results are in `MATRIXFORMER_PLUS_RESULTS.md`,
`OBSERVED_ROUTER.md`, `MATCHED_LLM_EVALUATION.md`, `SYNTHETIC_EVALUATIONS.md`,
`ORIENTATION_EVALUATION.md`, and `STREAMED_TRAINING.md`.

## Installation and checks

From this directory, install the research evaluation dependencies as needed:

```bash
pip install -e '.[training,benchmark,tabpfn_extensions]' cornac pytest
export PYTHONPATH="src:scripts:../benchmark-selection/code"
python -m pytest tests/test_observed_router.py tests/test_streamed_inference.py \
  tests/test_synthetic_low_rank_benchmark.py tests/test_matched_llm.py \
  tests/test_tabimpute_v2_interface_compatibility.py
python scripts/evaluate_matrixformer_plus_synthetic.py --help
python scripts/evaluate_matched_llm.py --help
```

The original TabImpute entry-wise interface and `tabimpute.tabimpute_v2` remain
available. Optional baseline packages are imported when those baselines are used.

## Artifacts and provenance

Checkpoint tensors, datasets, generated predictions, and private experiment
archives are excluded from this source publication. Evaluation requires the
checkpoints and input panels referenced by each protocol; installation alone
does not download the MatrixFormer experts. Historical scripts retain their
registered paths and hashes, so reproducing an archived run requires supplying
those artifacts or explicitly registering a new evaluation.

Small historical command recipes are retained under `runs/objective_queues/`
because the synthetic samplers verify their contents. These are configuration
files, not weights or training data. The legacy causal aggregation helper under
`69d7e55fd826e0325c3115d0/scripts/` supports the original saved-panel evaluator;
the latest benchmark CLI remains under `benchmark/run_benchmark.py`.

`../benchmark-selection/code/` contains the LLM selection helpers and their
upstream license. MatrixFormer results are separate from the Jellyfish paper:
`evaluation_protocols/jellyfish_counterfactual_evaluation.md` documents the
unavailable panel and the proposed counterfactual protocol, not completed results.
