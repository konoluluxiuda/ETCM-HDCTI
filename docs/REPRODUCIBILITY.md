# Reproducibility Guide

## 1. Environment

Create the tested Python 3.10/TensorFlow 2.21 environment:

```bash
conda env create -f environment.yml
conda activate HDCTI_tfnew
```

The final runs used an NVIDIA RTX 5060 Ti under WSL. GPU execution is optional;
set `HDCTI_FORCE_CPU=1` for CPU execution. The final model disables dense
all-node attention with `attention.max.nodes=0`.

## 2. Data Layout

Prepare the datasets as described in `docs/DATA_PREPARATION.md`. Dataset and
split directories are intentionally excluded from Git.

The frozen compound-cold manifests expect:

```text
dataset/<dataset>/splits/strict_compound_cold_start_seed_52026_k5/
```

If a split manifest is absent, the strict loader can generate it from the
configured positive and negative samples. A published result should reuse the
same manifest rather than regenerate it silently.

## 3. Final Model

List the eight frozen four-dataset jobs and verify config hashes:

```bash
./run_schpt_full.sh --dry-run
```

Run or resume them:

```bash
./run_schpt_full.sh
./run_schpt_full.sh --resume results/batch_runs/schpt_full_<timestamp>
```

Each dataset runs a matched Hctx-P+SDIS baseline and the final
Hctx-P+SDIS+SCHPT candidate. The summarizer writes `results.tsv`, `summary.md`,
and `summary.json` under the run directory.

## 4. External Baselines

Validate the 16 frozen jobs:

```bash
./run_cold_start_external_baselines_full.sh --dry-run
```

Run Dual-HGNN-CTI, LightGCN-CTI, R-GCN-CTI, and sparse HGT-CTI on the same
compound-cold folds:

```bash
./run_cold_start_external_baselines_full.sh
```

No database-specific hyperparameter search is performed.

## 5. Frozen Paper Tables

The repository contains only the minimal machine-readable result bundle, not
raw logs or checkpoints. Verify and regenerate the Markdown tables with:

```bash
python -m tools.export_paper_result_bundle --verify-only
python tools/build_paper_results_tables.py
```

Both commands check SHA-256 values before accepting result files.

## 6. Ranking Limitation Audit

The full-candidate evaluation is retained because sampled-pair AUPR does not
establish target-library retrieval performance:

```bash
./run_non_neural_cold_start_baselines.sh
./run_inner_full_candidate_ranking_audit.sh
./run_full_candidate_ranking_gate.sh
```

This audit is a limitation analysis, not a source of positive headline claims.

## 7. Tests

Run the retained paper-pipeline suite:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

GPU-dependent model smoke tests may be skipped or slower on CPU-only systems.
