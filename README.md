# ETCM-HDCTI

Research code for a support-conditioned extension of HDCTI for
side-information-assisted compound cold-start prediction in traditional
Chinese medicine (TCM) networks.

The final model combines three components in one fixed configuration:

- **Hctx-P**: pair-level interaction between herb context and a candidate target;
- **SDIS**: deterministic suppression of an unsupported compound-ID score;
- **SCHPT**: fold-local, leave-one-compound-out herb-target prototype transfer.

The implementation uses strict fold-local C-P statistics, inner-validation
early stopping, fixed split manifests, and five-fold outer evaluation. Dense
all-node self-attention is disabled in the final experiments.

## Upstream Attribution

This repository is a derivative of the original
[HDCTI implementation](https://github.com/tong87-bio/HDCTI) accompanying:

> Qiao Y, Hu P, Zhang J, et al. Identifying novel therapeutic targets of
> natural compounds in traditional Chinese medicine herbs with hypergraph
> representation learning. *Briefings in Bioinformatics*. 2025;26(4):bbaf399.
> [doi:10.1093/bib/bbaf399](https://doi.org/10.1093/bib/bbaf399)

The original MIT copyright and license are retained in [LICENSE](LICENSE).
Inherited and newly implemented components are documented in
[UPSTREAM_PROVENANCE.md](UPSTREAM_PROVENANCE.md). Do not cite this repository
as a from-scratch implementation of the HDCTI backbone.

## Repository Layout

```text
HDCTI.py, HDR.py, rating.py    model and strict evaluation pipeline
base/, util/                   shared training and graph utilities
configs/                       frozen paper configurations and manifests
tools/                         data preparation and result verification
tests/                         tests for the retained paper pipeline
paper_artifacts/results/       compact, hash-verified paper result bundle
figures/etcm_case_study/       reproducible representative-case figure
docs/                          method, protocol, result, and limitation records
```

Raw datasets, checkpoints, logs, manuscript drafts, and generated run
directories are intentionally excluded from Git.

## Environment

The final experiments were run with Python 3.10 and TensorFlow 2.21 in the
`HDCTI_tfnew` environment.

```bash
conda env create -f environment.yml
conda activate HDCTI_tfnew
```

The GPU launcher supplies the WSL/NVIDIA library path used on the development
machine:

```bash
./run_hdcti.sh configs/HDCTI_tcmsp_schpt_full.conf
```

Use `HDCTI_FORCE_CPU=1` for CPU-only execution.

## Data

Dataset files are not redistributed by this repository. Prepare the following
directories under `dataset/`:

```text
dataset/TCMsuite/
dataset/TCMSP/
dataset/Symmap/
dataset/ETCM2.0_core_mention10/
```

Each training directory must provide `H_C.txt`, `C_P.txt`, `P_D.txt`, the
positive sample file referenced by its config, and either a fixed negative
file or a reusable strict split manifest. See
[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for provenance and layout.

## Reproduce the Paper Experiments

Inspect all frozen jobs without training:

```bash
./run_schpt_full.sh --dry-run
./run_cold_start_external_baselines_full.sh --dry-run
```

Run the final SCHPT ablation and candidate model on all four datasets:

```bash
./run_schpt_full.sh
```

Run the same-input external baselines:

```bash
./run_cold_start_external_baselines_full.sh
```

Regenerate and verify the compact paper tables:

```bash
python -m tools.export_paper_result_bundle --verify-only
python tools/build_paper_results_tables.py
```

Detailed commands and expected outputs are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Results and Scope

The frozen tables are available in
[docs/FINAL_RESULTS_TABLES.md](docs/FINAL_RESULTS_TABLES.md). The supported
task is **compound C-P cold-start with H-C side information available**. It is
not target-cold, double-cold, or structure-only de novo prediction.

The historical 1:1 sampled-pair metrics are retained for comparison, but they
must not be interpreted as full target-library retrieval performance. The
full-candidate audit and its negative result are reported in
[docs/FULL_CANDIDATE_RANKING_GATE.md](docs/FULL_CANDIDATE_RANKING_GATE.md).

## Tests

Core tests do not require retraining:

```bash
python -m unittest \
  tests.test_strict_protocol \
  tests.test_herb_prototype_transfer \
  tests.test_cold_start_hctx_ablation \
  tests.test_cold_start_external_baselines \
  tests.test_paper_results_tables
```

## License

MIT. See [LICENSE](LICENSE) and [UPSTREAM_PROVENANCE.md](UPSTREAM_PROVENANCE.md).
