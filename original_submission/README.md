# Original submission (disc-level; archived)

This folder is the **first-submission** pipeline. It is kept so the original GitHub deposit remains inspectable.

It is **not** the source of the revised manuscript numbers.

| | Original submission (this folder) | Current paper (`../revision/`) |
|---|---|---|
| Split | Disc-level 80:20 (same patient can appear in train and test) | Patient-level (`assignments.csv`, seed 4321) |
| Primary AUC | 0.932 | **0.936 (0.913–0.958)** |
| Features after reduction | 313 | 517 |
| Main code | `scripts_R/` (caret) + `scripts_python/` notebooks | `revision/scripts/` (Python CatBoost primary) |

## Layout

```text
original_submission/
├── config/extraction_params.yaml
├── data/radiomic_feature_data.xlsx   # de-identified 3D features + disc-level train/test label
├── data/images/                      # placeholder; NIfTI not public
├── data/masks/
├── scripts_python/                   # feature extraction + original ICC notebook
├── scripts_R/                        # 10-model training, comparison, SHAP
└── output/models/                    # models fitted for the original 0.932 analysis
```

Relative paths inside `scripts_R/` and `scripts_python/` (`../data`, `../output`, `../config`) still resolve after this move.

The public feature table is `data/radiomic_feature_data.xlsx` (columns `Pfirrmann_Grading`, `Dataset`, then features). Some R scripts still name the file `extracted_data.xlsx`; rename or edit the path if you rerun them.

## Do not use this folder to reproduce the revised paper

For the patient-level primary analysis, go to [`../revision/`](../revision/).
