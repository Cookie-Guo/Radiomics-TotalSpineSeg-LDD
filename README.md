# Automated Radiomics Pipeline with TotalSpineSeg for Lumbar Disc Degeneration Classification and SHAP Interpretability Analysis

Source code and de-identified tables for:

*Automated Radiomics Pipeline with TotalSpineSeg for Classification of Lumbar Disc Degeneration on T2-Weighted MRI: Development and Interpretability Analysis*

Frontiers in Medicine (manuscript 1929252).

## Which folder to use

| Folder | Role |
|---|---|
| **[`revision/`](revision/)** | **Current manuscript.** Patient-level split. Primary CatBoost macro AUC **0.936**. |
| [`original_submission/`](original_submission/) | First submission only. Disc-level split. Historical AUC 0.932. Do not use these scripts for the revised numbers. |

```text
.
├── revision/                 # current paper
│   ├── data/                 # anonymous features + labels
│   ├── splits/assignments.csv
│   ├── scripts/              # 00_common … 11_figures (same numbers as results/)
│   ├── results/
│   └── figures/
└── original_submission/      # first submission (archived)
    ├── config/
    ├── data/
    ├── scripts_python/
    ├── scripts_R/
    └── output/models/
```

## Revision (patient-level; August 2026)

| Item | Value |
|---|---|
| Split | Patient-level 80:20 (`revision/splits/assignments.csv`, seed 4321) |
| Train / test | 168 / 42 patients (504 / 126 discs) |
| Primary model | **CatBoost**, pre-specified (depth 2, lr 0.05, l2 1, 223 iterations) |
| 3D features after reduction | 1,762 → 1,548 → 1,548 → **517** |
| **Primary 3D macro AUC** | **0.936 (0.913–0.958)** |
| Same-pipeline 2D mid-sagittal | 0.916 (0.884–0.945) |
| Simple baseline (disc/CSF SI) | 0.760 |
| Frozen ResNet50 (ImageNet) | 0.864 (0.828–0.899) |
| Frozen official RadImageNet ResNet50 | **0.874 (0.834–0.910)**; vs ImageNet Δ=+0.010, p=0.67 (ns); both below 3D 0.936 |
| Volume-only | 0.552 |
| Mask ±1-voxel ICC > 0.75 | 45.3% of features (median ICC 0.713) |
| SPIDER external (frozen 517; expert masks) | **0.838 (0.816–0.860)** — not a new primary AUC |

```text
python revision/scripts/02_primary/unified_primary.py
python revision/scripts/02_primary/ten_models_unified.py
```

Both scripts read only `revision/splits/assignments.csv` plus the de-identified feature tables. They do not re-split the data.

Details: [`revision/README.md`](revision/README.md).

## What is not in this repository

- Raw MRI and TotalSpineSeg masks (institutional ethics)
- The name-to-ID map (never published)
- SPIDER images (use the public SPIDER release on Zenodo, record 10159290)

Imaging data may be shared by the corresponding author upon reasonable academic request and ethics approval (Wangjing Hospital, China Academy of Chinese Medical Sciences).

## Requirements

**Python 3.10+** (PyRadiomics feature extraction needs 3.10): `pandas`, `numpy`, `scipy`, `scikit-learn`, `catboost`, `openpyxl`.  
Optional extractors: `pyradiomics`, `SimpleITK`, `torch`, `torchvision`, `pingouin`.

**R**: `caret`, `pROC`, `ggplot2`, `shapviz`, `catboost`.

See `revision/requirements.txt` for the revision modelling stack.

## License

MIT — see [LICENSE](LICENSE).
