# Dataset Statistics

Generated from local files under `dataset/` on 2026-07-02.
Large raw ETCM2.0 JSON files are summarized by file count and size only.

## Overview

| Dataset | Size | H_C | C_P | P_D | H_D | ONE | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TCMsuite` | 42.3MB | 6,496 | 43,669 | 44,170 | 2,354,225 | 43,669 | 1,048,576 |
| `TCMSP` | 3.4MB | 33,756 | 56,102 | 365 | 39,934 | 56,102 | 0 |
| `Symmap` | 161.8MB | 85,172 | 37,991 | 194,558 | 382,930 | 37,991 | 6,477,695 |
| `ETCM2.0_processed` | 34.0MB | 67,335 | 180,589 | 2,033,920 | 41,076 | 180,589 | 0 |
| `ETCM2.0_core` | 32.7MB | 36,216 | 109,747 | 1,991,225 | 41,076 | 109,747 | 109,747 |
| `ETCM2.0 raw JSON` | 40.2GB | - | - | - | - | - | - |

## ETCM2.0 Raw JSON

- Path: `dataset/ETCM2.0`
- Size: 40.2GB
- `etcm_herbs` JSON files: 2,076
- `etcm_targets` JSON files: 260
- `etcm_diseases` JSON files: 7,705
- Raw zip: `dataset/ETCM2.0_raw.zip` (2.4GB)

## TCMsuite

- Path: `dataset/TCMsuite`
- Size: 42.3MB

### Relation Files

| Relation | Unique edges | Raw rows | Malformed rows |
|---|---:|---:|---:|
| H_C (herb-compound) | 6,496 | 6,496 | 0 |
| C_P (compound-protein) | 43,669 | 43,669 | 0 |
| P_D (protein-disease) | 44,170 | 44,170 | 0 |
| H_D (herb-disease) | 2,354,225 | 2,354,225 | 0 |
| ONE (positive samples) | 43,669 | 43,669 | 0 |
| ZERO (negative samples) | 1,048,576 | 1,048,576 | 0 |

### Entity Usage From Edges

| Entity view | Count |
|---|---:|
| herbs_in_H_C | 1,009 |
| compounds_in_H_C | 1,193 |
| compounds_in_C_P | 1,187 |
| proteins_in_C_P | 7,258 |
| proteins_in_P_D | 2,045 |
| diseases_in_P_D | 11,071 |
| herbs_in_H_D | 1,008 |
| diseases_in_H_D | 11,071 |

### Connectivity / Intersection Review

| Check | Count | Total | Coverage |
|---|---:|---:|---:|
| C_P compounds with H_C support | 1,187 | 1,187 | 100.00% |
| C_P proteins with P_D support | 2,045 | 7,258 | 28.18% |
| P_D proteins with C_P support | 2,045 | 2,045 | 100.00% |
| P_D diseases with H_D support | 11,071 | 11,071 | 100.00% |
| H_D herbs with H_C support | 1,008 | 1,008 | 100.00% |
| C_P edges with both H_C and P_D support | 25,482 | 43,669 | 58.35% |
| ONE/ZERO overlap | 0 | 43,669 | 0.00% |

### Cross-validation Fold Files

| File | Rows |
|---|---:|
| `test_fold_0.txt` | 17,468 |
| `test_fold_1.txt` | 17,468 |
| `test_fold_2.txt` | 17,468 |
| `test_fold_3.txt` | 17,467 |
| `test_fold_4.txt` | 17,467 |

## TCMSP

- Path: `dataset/TCMSP`
- Size: 3.4MB

### Relation Files

| Relation | Unique edges | Raw rows | Malformed rows |
|---|---:|---:|---:|
| H_C (herb-compound) | 33,756 | 33,756 | 0 |
| C_P (compound-protein) | 56,102 | 56,102 | 0 |
| P_D (protein-disease) | 365 | 173,476 | 0 |
| H_D (herb-disease) | 39,934 | 40,019 | 0 |
| ONE (positive samples) | 56,102 | 56,102 | 0 |
| ZERO (negative samples) | 0 | 0 | 0 |

### Entity Usage From Edges

| Entity view | Count |
|---|---:|
| herbs_in_H_C | 501 |
| compounds_in_H_C | 13,655 |
| compounds_in_C_P | 6,929 |
| proteins_in_C_P | 1,748 |
| proteins_in_P_D | 364 |
| diseases_in_P_D | 321 |
| herbs_in_H_D | 447 |
| diseases_in_H_D | 322 |

### Mapping Files

| Entity | Rows |
|---|---:|
| herbs | 502 |
| compounds | 13,729 |
| proteins | 1,753 |
| diseases | 322 |

### Connectivity / Intersection Review

| Check | Count | Total | Coverage |
|---|---:|---:|---:|
| C_P compounds with H_C support | 6,907 | 6,929 | 99.68% |
| C_P proteins with P_D support | 364 | 1,748 | 20.82% |
| P_D proteins with C_P support | 364 | 364 | 100.00% |
| P_D diseases with H_D support | 321 | 321 | 100.00% |
| H_D herbs with H_C support | 446 | 447 | 99.78% |
| C_P edges with both H_C and P_D support | 41,762 | 56,102 | 74.44% |
| ONE/ZERO overlap | 0 | 56,102 | 0.00% |

## Symmap

- Path: `dataset/Symmap`
- Size: 161.8MB

### Relation Files

| Relation | Unique edges | Raw rows | Malformed rows |
|---|---:|---:|---:|
| H_C (herb-compound) | 85,172 | 85,172 | 0 |
| C_P (compound-protein) | 37,991 | 38,043 | 0 |
| P_D (protein-disease) | 194,558 | 196,110 | 0 |
| H_D (herb-disease) | 382,930 | 382,988 | 0 |
| ONE (positive samples) | 37,991 | 38,043 | 0 |
| ZERO (negative samples) | 6,477,695 | 6,477,695 | 0 |

### Entity Usage From Edges

| Entity view | Count |
|---|---:|
| herbs_in_H_C | 686 |
| compounds_in_H_C | 25,659 |
| compounds_in_C_P | 1,618 |
| proteins_in_C_P | 4,027 |
| proteins_in_P_D | 17,898 |
| diseases_in_P_D | 6,155 |
| herbs_in_H_D | 685 |
| diseases_in_H_D | 10,900 |

### Mapping Files

| Entity | Rows |
|---|---:|
| herbs | 697 |
| compounds | 27,277 |
| proteins | 18,192 |
| diseases | 12,690 |

### Connectivity / Intersection Review

| Check | Count | Total | Coverage |
|---|---:|---:|---:|
| C_P compounds with H_C support | 1,596 | 1,618 | 98.64% |
| C_P proteins with P_D support | 3,733 | 4,027 | 92.70% |
| P_D proteins with C_P support | 3,733 | 17,898 | 20.86% |
| P_D diseases with H_D support | 4,365 | 6,155 | 70.92% |
| H_D herbs with H_C support | 674 | 685 | 98.39% |
| C_P edges with both H_C and P_D support | 35,960 | 37,991 | 94.65% |
| ONE/ZERO overlap | 0 | 37,991 | 0.00% |

## ETCM2.0_processed

- Path: `dataset/ETCM2.0_processed`
- Size: 34.0MB

### Relation Files

| Relation | Unique edges | Raw rows | Malformed rows |
|---|---:|---:|---:|
| H_C (herb-compound) | 67,335 | 67,335 | 0 |
| C_P (compound-protein) | 180,589 | 180,589 | 0 |
| P_D (protein-disease) | 2,033,920 | 2,033,920 | 0 |
| H_D (herb-disease) | 41,076 | 41,076 | 0 |
| ONE (positive samples) | 180,589 | 180,589 | 0 |
| ZERO (negative samples) | 0 | 0 | 0 |

### Entity Usage From Edges

| Entity view | Count |
|---|---:|
| herbs_in_H_C | 1,898 |
| compounds_in_H_C | 38,255 |
| compounds_in_C_P | 22,541 |
| proteins_in_C_P | 973 |
| proteins_in_P_D | 562 |
| diseases_in_P_D | 7,693 |
| herbs_in_H_D | 457 |
| diseases_in_H_D | 3,929 |

### Mapping Files

| Entity | Rows |
|---|---:|
| herbs | 2,075 |
| compounds | 38,255 |
| proteins | 987 |
| diseases | 7,751 |

### Connectivity / Intersection Review

| Check | Count | Total | Coverage |
|---|---:|---:|---:|
| C_P compounds with H_C support | 22,541 | 22,541 | 100.00% |
| C_P proteins with P_D support | 548 | 973 | 56.32% |
| P_D proteins with C_P support | 548 | 562 | 97.51% |
| P_D diseases with H_D support | 3,929 | 7,693 | 51.07% |
| H_D herbs with H_C support | 457 | 457 | 100.00% |
| C_P edges with both H_C and P_D support | 109,747 | 180,589 | 60.77% |
| ONE/ZERO overlap | 0 | 180,589 | 0.00% |

## ETCM2.0_core

- Path: `dataset/ETCM2.0_core`
- Size: 32.7MB

### Relation Files

| Relation | Unique edges | Raw rows | Malformed rows |
|---|---:|---:|---:|
| H_C (herb-compound) | 36,216 | 36,216 | 0 |
| C_P (compound-protein) | 109,747 | 109,747 | 0 |
| P_D (protein-disease) | 1,991,225 | 1,991,225 | 0 |
| H_D (herb-disease) | 41,076 | 41,076 | 0 |
| ONE (positive samples) | 109,747 | 109,747 | 0 |
| ZERO (negative samples) | 109,747 | 109,747 | 0 |

### Entity Usage From Edges

| Entity view | Count |
|---|---:|
| herbs_in_H_C | 1,832 |
| compounds_in_H_C | 19,242 |
| compounds_in_C_P | 19,242 |
| proteins_in_C_P | 548 |
| proteins_in_P_D | 548 |
| diseases_in_P_D | 7,693 |
| herbs_in_H_D | 457 |
| diseases_in_H_D | 3,929 |

### Mapping Files

| Entity | Rows |
|---|---:|
| herbs | 1,832 |
| compounds | 19,242 |
| proteins | 548 |
| diseases | 7,693 |

### Connectivity / Intersection Review

| Check | Count | Total | Coverage |
|---|---:|---:|---:|
| C_P compounds with H_C support | 19,242 | 19,242 | 100.00% |
| C_P proteins with P_D support | 548 | 548 | 100.00% |
| P_D proteins with C_P support | 548 | 548 | 100.00% |
| P_D diseases with H_D support | 3,929 | 7,693 | 51.07% |
| H_D herbs with H_C support | 457 | 457 | 100.00% |
| C_P edges with both H_C and P_D support | 109,747 | 109,747 | 100.00% |
| ONE/ZERO overlap | 0 | 109,747 | 0.00% |

### Cross-validation Fold Files

| File | Rows |
|---|---:|
| `test_fold_0.txt` | 43,899 |
| `test_fold_1.txt` | 43,899 |
| `test_fold_2.txt` | 43,899 |
| `test_fold_3.txt` | 43,899 |
| `test_fold_4.txt` | 43,898 |

## Notes

- `Unique edges` are deduplicated by the first two columns.
- For sample files, labels in the third column are ignored for edge uniqueness.
- TCMSP/Symmap relation names are normalized to the same H_C/C_P/P_D/H_D concepts for comparison.
- `ETCM2.0_core` is the active training dataset in `HDCTI.conf` at the time this report was generated.
