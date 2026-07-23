# ETCM2.0 Representative Case Figure

## Figure contract

- Core conclusion: Hctx-P/CHCR moved two independently supported targets toward
  the top of the frozen candidate list, while a retained conflict case shows
  that high rank is not equivalent to validated activity.
- Archetype: asymmetric mixed-modality figure.
- Final size: 183 mm wide and 105 mm high.
- Vector exports: SVG and PDF with editable text.
- Preview export: 400 dpi PNG.
- Source data: `source_data.tsv`, generated from the frozen representative-case
  JSON.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/hdcti-mpl \
  /home/zry/.conda/envs/HDCTI/bin/python \
  figures/etcm_case_study/figure_etcm_cases.py
```

The script reads:

```text
configs/etcm_topk_representative_cases.json
```

It does not import TensorFlow, restore a checkpoint, or change model rankings.

## Evidence boundary

- Solid arrows denote direct experimental evidence.
- Herb context is model-side information, not independent binding evidence.
- Dashed database paths are post-hoc hypotheses only.
- The three frozen cases are illustrative and do not estimate overall model
  precision.
