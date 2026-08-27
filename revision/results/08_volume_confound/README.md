# 08_volume_confound

Volume defined as `original_shape_MeshVolume` (mm³). Spearman vs all features and vs the primary 517; volume-only CatBoost on the same split and hyperparameters (depth=2, lr=0.05, l2=1, iter=223, seed=4321).

- All features \|ρ\|>0.5: 0.172 (302/1760), median \|ρ\|=0.223
- Primary 517 \|ρ\|>0.5: 0.203 (105/517), median \|ρ\|=0.234
- Volume-only macro AUC: **0.552 (0.491–0.622)** vs 3D primary 0.936
- MeshVolume not among the 517

Tables: `feature_volume_spearman.csv`, `volume_only_performance.csv`, `volume_vs_primary.csv`, `summary.json`.
