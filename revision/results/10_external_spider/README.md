# 10_external_spider

## Scope of this validation

Frozen 517-feature CatBoost applied to public SPIDER (Zenodo 10159290). Expert reference masks, conventional sagittal T2, L3–S1. Not an end-to-end TotalSpineSeg test. Images live under `<spider_root>`; this folder holds inventory, features and performance only.

| Item | Value |
|---|---|
| External macro AUC | **0.838 (0.816–0.860)** |
| vs internal primary | Δ = **−0.098** (internal 0.936 unchanged) |
| Accuracy / κ_quad | 0.536 / 0.705 |
| n | 597 discs / 199 patients |
| Grades I–V | 52 / 93 / 191 / 146 / 115 |
| Feature alignment | **517/517** |

Extractor: 3×3×3 mm, binWidth=5, normalizeScale=200. Main table: `performance.csv`.
