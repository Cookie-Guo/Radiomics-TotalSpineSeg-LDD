# Automated Radiomics Pipeline with TotalSpineSeg for Lumbar Disc Degeneration Classification and SHAP Interpretability Analysis

This repository contains the complete source code and workflow for the study: 
*"Automated Radiomics Pipeline with TotalSpineSeg for Classification of Lumbar Disc Degeneration on T2-Weighted MRI: Development and Interpretability Analysis"*

## Research Highlights

- **Efficiency**: Automated segmentation via **TotalSpineSeg** processed 210 cases in **24 minutes (6.9s/case)**.
- **Reproducibility**: **99.3%** of extracted features showed excellent reproducibility (**ICC > 0.75**).
- **Optimal Performance**: The **CatBoost** model achieved a macro-averaged **AUC of 0.932** (95% CI: 0.906–0.955).
- **Explainability**: SHAP analysis identified **10th Percentile**, **Sphericity**, and **Difference Entropy** as key imaging biomarkers.

##  Repository Structure
- `config/`: PyRadiomics extraction parameters.
- `data/`: Placeholders for imaging and features (Protected due to patient privacy).
- `scripts_python/`: Notebooks for feature extraction and ICC selection.
- `scripts_R/`: Scripts for machine learning training, model comparison, and SHAP analysis.
- - `output/`: Organized directory placeholders for generated models and publication-ready visualizations (results are withheld until formal publication).

##  Requirements
- **Python**: pyradiomics, pingouin, pandas, numpy.
- **R**: caret, pROC, ggplot2, shapviz, catboost.

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
