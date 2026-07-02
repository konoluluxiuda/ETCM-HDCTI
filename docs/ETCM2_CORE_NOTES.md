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

## Experiment Configuration History

### Baseline Project Configuration Before ETCM2.0

The project originally used TCMsuite with hard-coded auxiliary paths:

```text
datapath=./dataset/TCMsuite/ONE_indices.txt
batch_size=2000
learnRate=-init 0.005 -max 1
num.max.epoch=50
evaluation.setup=-cv 5
```

Auxiliary files were read directly from `./dataset/TCMsuite/` in `rating.py` and
`util/io.py`.

### First ETCM2.0_core Configuration

After creating `ETCM2.0_core`, the first ETCM2.0 training attempt used:

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
batch_size=2000
learnRate=-init 0.005 -max 1
attention.max.nodes=2000
gpu.multiprocessing=False
```

This configuration kept protein full self-attention but skipped compound full
self-attention, because ETCM2.0_core has 19,242 compounds. It still hit CUDA
instability on the local RTX 5060 Ti setup.

### Final Stable ETCM2.0_core Configuration

The configuration that completed training was:

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
batch_size=1000
learnRate=-init 0.001 -max 1
attention.max.nodes=0
gpu.multiprocessing=False
num.max.epoch=50
evaluation.setup=-cv 5
```

This is now the recommended local RTX 5060 Ti configuration for ETCM2.0_core.

## GPU/OOM and CUDA Stability Changes

ETCM2.0_core has 19,242 compounds. The original full self-attention code builds
an `N x N` attention matrix for compounds:

```text
19242 x 19242 ~= 1.38 GiB per attention logits matrix
```

On the RTX 5060 Ti this caused GPU OOM at `Softmax_6`.

`HDCTI.py` now supports:

```text
attention.max.nodes=<integer>
```

If a node type exceeds this threshold, the expensive full self-attention block
for that node type is skipped.

Additional stability changes were added after CUDA illegal-address crashes:

- Replaced manual BCE:

  ```python
  log(sigmoid(x)) + log(1 - sigmoid(x))
  ```

  with numerically stable:

  ```python
  tf.nn.sigmoid_cross_entropy_with_logits(...)
  ```

- Replaced deprecated dense/sparse matmul path:

  ```python
  tf.sparse_tensor_to_dense(...)
  tf.matmul(..., a_is_sparse=True)
  ```

  with:

  ```python
  tf.sparse_tensor_dense_matmul(...)
  ```

- Cast PageRank weighting arrays to `float32`.
- Set `NVIDIA_TF32_OVERRIDE=0` by default in `util/gpu.py`.
- Reduced `batch_size` from `2000` to `1000`.
- Reduced learning rate from `0.005` to `0.001`.
- Set `attention.max.nodes=0`, disabling full self-attention for both compound
  and protein nodes.

## Model Differences From The Original HDCTI Run

Relative to the original HDCTI code path, the stable ETCM2.0_core run changes
the model/training behavior in these ways:

- Full `N x N` self-attention is disabled for both compounds and proteins.
- Hypergraph-style sparse propagation over `H_C` and `P_D` remains active.
- Self-gating remains active.
- Feature-wise attention after propagation remains active.
- The loss is mathematically the same binary cross entropy objective, but now
  computed by TensorFlow's stable logits-based implementation.
- Sparse graph propagation uses TensorFlow's standard sparse-dense matmul op
  instead of converting sparse matrices to dense tensors first.

This means the stable ETCM2.0_core result should be interpreted as an
ETCM2.0_core-compatible HDCTI variant, not as a bitwise-identical reproduction
of the small-dataset full-attention architecture.

## Expected Impact

Positive effects:

- ETCM2.0_core can be loaded by the existing training pipeline.
- Compound-protein positives are strongly connected to the auxiliary graphs.
- GPU training avoids the large compound self-attention OOM and the observed
  CUDA illegal-address crashes on the local RTX 5060 Ti setup.
- Dataset switching is now controlled by `datapath` instead of hard-coded paths.

Trade-offs:

- Skipping full self-attention changes the model architecture for large
  datasets. Graph propagation, gating, and later feature-wise attention remain
  active.
- Results on ETCM2.0_core are therefore not directly identical to runs that use
  full compound/protein self-attention on smaller datasets.
- Lowering `attention.max.nodes` further reduces memory use; raising it may
  improve expressiveness but can reintroduce OOM.

Recommended default for ETCM2.0_core:

```text
attention.max.nodes=0
batch_size=1000
learnRate=-init 0.001 -max 1
```

This disables full self-attention for both compound and protein nodes and uses a
more conservative optimizer setup. It was selected after RTX 5060 Ti runs hit
`CUDA_ERROR_ILLEGAL_ADDRESS` with the less conservative GPU configuration.

If later testing uses a newer TensorFlow/CUDA stack with stable RTX 50-series
support, this can be relaxed for comparison:

```text
attention.max.nodes=2000
```

That keeps protein full self-attention while skipping compound full
self-attention. It did not remain stable on the local RTX 5060 Ti setup with
TensorFlow 2.6/CUDA 11.2, so it is not the current default.

## Training Run Log

### 2026-07-02 Model/Config Restore Note

The model implementation and active configuration were restored to the
pre-stability-downgrade ETCM2.0_core state after the stable run was recorded.
The current restored runtime configuration is:

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
batch_size=2000
learnRate=-init 0.005 -max 1
attention.max.nodes=2000
gpu.multiprocessing=False
```

The stable-run notes below are retained as historical experiment records. They
describe the conservative configuration that completed training on the local RTX
5060 Ti setup, but they are no longer the active model/config state after this
restore.

### 2026-07-01 ETCM2.0_core Stable GPU Run

Configuration:

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
evaluation.setup=-cv 5
num.max.epoch=50
batch_size=1000
attention.max.nodes=0
learnRate=-init 0.001 -max 1
gpu.multiprocessing=False
```

Observed result:

```text
training: 50 batch 175 loss: 232.37389
model checkpoint: ./saved_model/2026-07-01 20-42-32/hdcti_model.ckpt
fold [5] auc: 0.9692204571558858
running time: 1579.260556 s
```

Notes:

- The run completed through epoch 50 and saved model weights.
- The result used the final stable configuration above, including disabled full
  self-attention, stable BCE, standard sparse-dense graph propagation, lower
  learning rate, smaller batch size, and TF32 disabled.
- The console output did not include a populated aggregate 5-fold summary after
  `The result of 5-fold cross validation:`; the visible recorded metric from
  the supplied log is the fold `[5]` AUC above.
- This run confirms the conservative sparse propagation plus disabled full
  self-attention configuration avoids the earlier GPU OOM and illegal-address
  crashes on the local RTX 5060 Ti setup.

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
