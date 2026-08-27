# scripts/

Script folders share integer prefixes with `results/`. Stage 00 has scripts only (no public results folder).

```
scripts/
├── 00_common/           split, labels, de-identification
│   ├── splits.py
│   ├── make_labels.py
│   └── anonymize.py
├── 02_primary/          3D + 2D CatBoost primary and ten-model table
│   ├── unified_primary.py      → results/02_primary/
│   ├── ten_models_unified.py
│   ├── ten_models_pairwise.py
│   └── class_auc_ci.py
├── 03_compare_2d/       mid-sagittal 2D radiomics
│   ├── extract_2d_features.py
│   └── model_2d_3d.py
├── 04_compare_simple/   simple-measurement baseline
│   ├── extract_simple_features.py
│   ├── model_simple.py
│   └── single_feature_models.py
├── 05_compare_cnn/      frozen ImageNet / RadImageNet encoders
│   ├── extract_resnet50.py
│   ├── model_vision.py
│   ├── extract_radimagenet_official.py
│   ├── audit_radimagenet_weights.py
│   └── fetch_official_radimagenet_h5.py
├── 06_pairwise/         paired bootstrap across comparator models
│   └── comparator_pairwise.py
├── 07_perturbation_icc/ mask ±1-voxel ICC
│   ├── extract_perturbed.py
│   └── compute_icc.py
├── 08_volume_confound/  volume confounding
│   └── volume_confounding.py
├── 09_interpretability/ SHAP / grade trends / Top-k ablation
│   └── interpretability.py
├── 10_external_spider/  SPIDER external tables
│   ├── download.py             → local SPIDER download directory
│   ├── inventory.py
│   ├── extract.py
│   └── apply_primary.py
└── 11_figures/          Figures 3–5
    ├── fig3_ten_models.R
    ├── fig4_roc_cm.R
    └── fig5_shap.R
```

**Split:** every modelling script must read `splits/assignments.csv`. Do not create a new split.
**Meta:** important runs write a sibling `.meta.json`.
