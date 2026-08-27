# 05_compare_cnn

Frozen ResNet50 embeddings (no fine-tuning, no flip) with the same CatBoost head and split.

| Arm | macro AUC (95% CI) | Main table |
|---|---|---|
| Frozen ImageNet ResNet50 | **0.864 (0.828–0.899)** | `imagenet_vs_radiomics.csv` |
| Frozen official RadImageNet ResNet50 | **0.874 (0.834–0.910)** | `radimagenet_vs_radiomics.csv` |

Delta = +0.010, p = 0.67 (ns). Both below 3D radiomics **0.936 (0.913–0.958)** and 2D **0.916**. Weight provenance: `audit_weights.md`. Pairwise bootstrap: `../06_pairwise/`.
