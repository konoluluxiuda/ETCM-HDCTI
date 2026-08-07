# Data Preparation

## 1. Data Are Not Redistributed

The four training datasets and raw ETCM2.0 pages are excluded from Git because
of their size and source-specific redistribution conditions. Users must obtain
the source data from the corresponding database or the original HDCTI release.

The original HDCTI datasets and code were published at:

- https://github.com/tong87-bio/HDCTI
- https://doi.org/10.1093/bib/bbaf399

ETCM2.0 source provenance should cite the ETCM2.0 database article and record
the retrieval date. Do not treat unobserved compound-target pairs as confirmed
biological negatives.

## 2. Required Relation Files

Each processed dataset directory uses the following normalized relations:

| File | Relation | Role |
|---|---|---|
| `H_C.txt` | herb-compound | compound-side hypergraph and side information |
| `C_P.txt` | compound-protein | fold-local supervised graph statistics |
| `P_D.txt` | protein-disease | protein-side hypergraph |
| positive sample file | compound-protein-label | training/evaluation records |
| negative sample file | sampled unobserved pairs | fixed comparison records |

`H_D.txt` may exist in processed data but is disabled in the final model because
its independence and provenance are not uniformly established across datasets.

## 3. ETCM2.0 Processing

The retained construction sequence is:

```bash
python tools/build_etcm2_entity_mappings.py --help
python tools/build_etcm2_relations.py --help
python tools/create_etcm2_core.py --help
python tools/create_etcm2_pruned_core.py --help
```

The paper uses `ETCM2.0_core_mention10`, which filters compounds by
`mention_count >= 10` after constructing the connected core. The script keeps
source entity IDs and lets the runtime loader create dense internal indices.

## 4. Strict Splits

The final protocol is compound cold-start:

- all C-P pairs for a held-out compound belong to one outer fold;
- H-C side information remains available;
- all C-P-derived PageRank, degree, prototype, and prior statistics are rebuilt
  from the current fold's training positives;
- inner validation is created inside the outer training partition;
- outer-test labels are never used for model or epoch selection.

The exact seed and split directory are frozen in each `*_schpt_full.conf` file.

## 5. Audits

- Dataset statistics: `docs/DATASET_STATISTICS.md`
- H-D provenance and leakage audit: `docs/H_D_SOURCE_AUDIT.md`
- ETCM ingredient evidence processing: `docs/ETCM2_INGREDIENT_VALIDATION.md`
- Cold-start protocol: `docs/COMPOUND_COLD_START_PROTOCOL.md`

