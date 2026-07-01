# ETCM2.0 Core Dataset Integration Notes

## What Changed

This update adds an ETCM2.0 dataset preparation workflow and switches the active
training configuration to the generated core dataset:

- `tools/build_etcm2_entity_mappings.py`
  - Parses ETCM2.0 raw JSON pages.
  - Builds entity mapping tables for herbs, compounds, proteins/targets, and diseases.
  - Writes audit statistics for entity counts and sources.
- `tools/build_etcm2_relations.py`
  - Builds positive relation tables: `H_C.txt`, `C_P.txt`, `P_D.txt`, `H_D.txt`.
  - Writes `ONE_indices.txt` from positive compound-protein edges.
  - Writes relation coverage and intersection audit statistics.
- `tools/create_etcm2_core.py`
  - Creates `dataset/ETCM2.0_core` from `dataset/ETCM2.0_processed`.
  - Keeps only compound-protein positives whose compound appears in `H_C` and protein appears in `P_D`.
  - Generates 1:1 negative samples in `ZERO_indices.txt` with seed `2026`.

The generated dataset directories are intentionally ignored by git through
`.gitignore`:

```text
dataset/
```

The scripts and this note are tracked; the large dataset files are local-only.

## Active Dataset

`HDCTI.conf` now points to:

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
```

Current local core dataset statistics:

```text
H_C.txt          36,216
C_P.txt         109,747
P_D.txt       1,991,225
H_D.txt          41,076
ONE_indices     109,747
ZERO_indices    109,747
```

Core connectivity checks:

```text
C_P compound with H_C support: 100%
C_P protein with P_D support: 100%
P_D protein with C_P support: 100%
H_D herb with H_C support: 100%
P_D disease with H_D support: 51.07%
```

The remaining `P_D disease with H_D support` ratio is lower because ETCM2.0
contains many disease-target records for diseases that do not have related-herb
records. The core dataset is optimized for the current HDCTI compound-protein
training target, not for complete disease-herb closure.

## Runtime Path Changes

The model no longer hard-codes `dataset/TCMsuite` for auxiliary files:

- `rating.py`
  - Reads `H_C.txt`, `C_P.txt`, `P_D.txt`, and `H_D.txt` from the directory that
    contains `datapath`.
- `util/io.py`
  - Reads `ZERO_indices.txt` from the same directory as `datapath`.
  - Samples negatives 1:1 with the loaded positives instead of using a fixed
    TCMsuite count.
- `HDR.py`
  - Writes cross-validation `test_fold_*.txt` files into the active dataset
    directory.

This keeps TCMsuite, Symmap, TCMSP-style, and ETCM2.0-style directories easier
to switch by changing `datapath`.

## GPU/OOM Change

ETCM2.0_core has 19,242 compounds. The original full self-attention code builds
an `N x N` attention matrix for compounds:

```text
19242 x 19242 ~= 1.38 GiB per attention logits matrix
```

On the RTX 5060 Ti this caused GPU OOM at `Softmax_6`.

`HDCTI.py` now supports:

```text
attention.max.nodes=2000
```

If a node type exceeds this threshold, the expensive full self-attention block
for that node type is skipped. For ETCM2.0_core this skips compound full
self-attention but keeps protein full self-attention because the protein count
is only 548.

## Expected Impact

Positive effects:

- ETCM2.0_core can be loaded by the existing training pipeline.
- Compound-protein positives are strongly connected to the auxiliary graphs.
- GPU training avoids the large compound self-attention OOM.
- Dataset switching is now controlled by `datapath` instead of hard-coded paths.

Trade-offs:

- Skipping compound full self-attention changes the model architecture for large
  datasets. Graph propagation, gating, protein full self-attention, and the
  later feature-wise attention remain active.
- Results on ETCM2.0_core are therefore not directly identical to runs that use
  full compound self-attention on smaller datasets.
- Lowering `attention.max.nodes` further reduces memory use; raising it may
  improve expressiveness but can reintroduce OOM.

Recommended default for ETCM2.0_core:

```text
attention.max.nodes=2000
```

If memory is still tight:

```text
attention.max.nodes=0
```

This disables full self-attention for both compound and protein nodes.

## Rebuild Commands

From the repository root:

```bash
python tools/build_etcm2_entity_mappings.py \
  --input dataset/ETCM2.0 \
  --output dataset/ETCM2.0_processed \
  --progress-every 500

python tools/build_etcm2_relations.py \
  --input dataset/ETCM2.0 \
  --output dataset/ETCM2.0_processed \
  --progress-every 500

python tools/create_etcm2_core.py \
  --input dataset/ETCM2.0_processed \
  --output dataset/ETCM2.0_core \
  --seed 2026 \
  --negative-ratio 1.0
```

Run training:

```bash
conda activate HDCTI
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python main.py
```
