# 09_interpretability

On the primary 517-feature CatBoost (same split and hyperparameters): grade-wise Spearman, correlation vs simple measurements, native TreeSHAP, Top-5/10 ablation.

- 517 vs grade: median \|ρ\|=0.351; \|ρ\|>0.5 = 0.263 (136/517)
- vs DHI / peak SI: almost no \|ρ\|>0.5 overlap (0.008 / 0.000)
- Ablation macro AUC: Top-5 = 0.915; Top-10 = **0.922**; primary = 0.936

Tables: `concordance_table.csv`, `shap_mean_abs.csv`, `ablation_performance.csv`, `summary.json`.
