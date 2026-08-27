# 03_compare_2d

## Reported estimates

Same patient-level split and CatBoost pipeline as `02_primary/`.

| Config | macro AUC (95% CI) | n_feat |
|---|---|---|
| **3D_primary** | **0.936 (0.913–0.958)** | 517 |
| **2D_midsagittal** | **0.916 (0.884–0.945)** | 244 |

3D: 1762→1548→1548→**517**. 2D: 1209→941→941→**244** (Kruskal–Wallis + Bonferroni → NZV → \|r\|>0.9).

`features_2d.csv`: mid-sagittal largest-area slice, force2D, no 3 mm resampling. Authoritative table: `../02_primary/primary_performance.csv`.
