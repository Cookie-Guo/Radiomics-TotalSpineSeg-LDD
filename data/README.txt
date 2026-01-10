Data Availability and Instructions

1. Data Privacy Statement
Due to patient privacy regulations and institutional ethical requirements, the raw MRI datasets used in this study are not hosted in this public repository. However, to ensure full reproducibility of the machine learning results, the extracted radiomic feature data with dataset split labels (radiomic_feature_data.xlsx) is publicly available in this directory.

2. Directory Structure
data/
├── images/                       # Raw T2-weighted MRI files (.nii or .nii.gz) [not publicly available]
├── masks/                        # Segmentation masks from TotalSpineSeg (.nii or .nii.gz) [not publicly available]
└── radiomic_feature_data.xlsx    # Extracted radiomic features with dataset split labels [publicly available]

3. Public Data: radiomic_feature_data.xlsx
This file contains all information needed to reproduce the reported model performance:
- Total samples: 630 intervertebral discs (L3-L4, L4-L5, L5-S1 from 210 patients)
- Columns:
  - Sample identifier
  - Dataset: Training (n=504, 80%) or Test (n=126, 20%), stratified by Pfirrmann grade with random seed 4321
  - Pfirrmann_Grading: Clinical Pfirrmann Grade (I, II, III, IV, or V)
  - 1,762 radiomic features (after ICC filtering, ICC > 0.75)
- Users can directly use this file with scripts in scripts_R/ to reproduce all machine learning results.

4. File Format and Metadata (for full pipeline reproduction)
For users who obtain access to the original imaging data:
- Imaging Data: All MRI and mask files must be in NIfTI format.
- Alignment: The Subject IDs in the Excel files must match the corresponding file names in the images/ and masks/ folders.
- Feature Extraction: Use config/extraction_params.yaml for PyRadiomics configuration.

5. Academic Data Access
The original MRI datasets used in this research may be provided by the corresponding author upon reasonable academic request, subject to approval by the Ethics Committee of Wangjing Hospital, China Academy of Chinese Medical Sciences.
