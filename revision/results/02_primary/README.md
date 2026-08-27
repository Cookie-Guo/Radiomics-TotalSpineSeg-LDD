# 02_primary

Patient-level CatBoost primary and the ten-model table on the same 517 3D features.

| Item | Value |
|---|---|
| Split | `revision/splits/assignments.csv` (seed 4321) |
| Reduction (train only) | Kruskal–Wallis + Bonferroni → near-zero variance → \|r\|>0.9 |
| **3D macro AUC** | **0.936 (0.913–0.958)**, 517 features |
| **2D macro AUC** | **0.916 (0.884–0.945)**, 244 features |

Main table: `primary_performance.csv`. Ten models: `ten_models_test_performance.csv`.
