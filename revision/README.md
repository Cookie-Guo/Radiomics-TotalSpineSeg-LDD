# Patient-level analysis

This folder is the **authoritative** code and table set for the current manuscript.

The first-submission disc-level scripts are archived in [`../original_submission/`](../original_submission/).

Identifiers are anonymous (`P001`…`P210` internally; `S001`… for SPIDER).
Row alignment: `data/extracted_data.xlsx` has the same row order as `data/labels.csv`. The `MASK` column is the anonymous `disc_id` (not a name).

## Layout

```text
revision/
├── data/
│   ├── extracted_data.xlsx      # 630 × 1,762 3D features; MASK = P###_L#-L#
│   ├── labels.csv               # disc_id, patient_id, level, pfirrmann
│   ├── original_split.csv       # original disc-level train/test (leakage audit)
│   └── extraction_params.yaml
├── splits/
│   └── assignments.csv          # ONLY allowed split (seed 4321)
├── scripts/                     # 00_common, 02_primary … 11_figures
├── results/                     # numbers reported in the paper (02 … 11)
└── figures/                     # Figure 1–5 (png/pdf) + Figure 1 draw.io
```

`results/` numbering starts at **02** because 00 (shared helpers) lives only under `scripts/`; the same integers are used on both sides so each `scripts/` stage folder maps onto the `results/` folder with the same number.

## How to rerun modelling (no images required)

From the repository root, after installing `revision/requirements.txt`:

| Step | Script | Writes |
|---|---|---|
| 3D + 2D primary CatBoost | `scripts/02_primary/unified_primary.py` | `results/02_primary/primary_performance.csv` |
| Ten models on the 517 3D features | `scripts/02_primary/ten_models_unified.py` | `ten_models_test_performance.csv` |
| Simple-feature models | `scripts/04_compare_simple/model_simple.py` | `results/04_compare_simple/` |
| Frozen ResNet50 models | `scripts/05_compare_cnn/model_vision.py` | `results/05_compare_cnn/` |
| Official RadImageNet ResNet50 | `scripts/05_compare_cnn/extract_radimagenet_official.py` | `results/05_compare_cnn/` |
| Volume confounding | `scripts/08_volume_confound/volume_confounding.py` | `results/08_volume_confound/` |
| Interpretability / SHAP | `scripts/09_interpretability/interpretability.py` | `results/09_interpretability/` |
| Apply frozen 517 to SPIDER | `scripts/10_external_spider/apply_primary.py` | `results/10_external_spider/performance.csv` |
| Figures 3–5 | `scripts/11_figures/fig3_ten_models.R` etc. | `figures/Figure3_*` … `Figure5_*` |

Extraction scripts (`extract_2d_features.py`, `extract_resnet50.py`, `extract_perturbed.py`, `extract.py`) need local NIfTI files and are provided for methods transparency only.

**Do not** call `train_test_split` / `createDataPartition`. Always read `splits/assignments.csv`.

## Reported point estimates

These match `results/02_primary/` and `results/10_external_spider/performance.csv`.

- CatBoost 3D (primary): macro AUC **0.936** (0.913–0.958); accuracy 0.754; quadratic κ 0.871
- CatBoost 2D: **0.916** (0.884–0.945)
- Frozen official RadImageNet ResNet50: **0.874** (0.834–0.910); vs frozen ImageNet 0.864, p=0.67 (ns); both below 3D 0.936. Audit: `results/05_compare_cnn/audit_weights.md`
- SPIDER, frozen 517, expert masks: **0.838** (0.816–0.860); n = 597 discs / 199 patients
- SPIDER is **not** an end-to-end TotalSpineSeg external test (masks are the public expert labels)

## What was intentionally left out

| Left out | Reason |
|---|---|
| Raw MRI / masks | Ethics; available on request |
| `features_{original,erode1,dilate1}.csv` | ~56 MB intermediates; ICC summaries are included |
| `B1_2d_slice_meta.csv` | Local paths |
| Name–ID map | Never published |
| Acquisition/QC audit scripts | Require raw images not included here |
| Legacy R implementation | Superseded by the Python pipeline |
