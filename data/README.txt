Data Availability and Instructions
1. Data Privacy Statement
Due to patient privacy regulations, institutional ethical requirements, and the pending status of the associated publication, the raw MRI datasets and processed radiomics feature files used in this study are not hosted in this public repository. These datasets were utilized to develop and validate the automated radiomics pipeline for lumbar disc degeneration (LDD) classification.

2. Directory Structure Requirements
To ensure the provided scripts execute correctly, please organize your local data directory as follows:

Plaintext

data/
├── images/            # Raw T2-weighted MRI files (.nii or .nii.gz)
├── masks/             # Segmentation masks from TotalSpineSeg (.nii or .nii.gz)
├── features_run1.xlsx # Features from the first extraction (for ICC analysis)
├── features_run2.xlsx # Features from the second extraction (for ICC analysis)
└── extracted_data.xlsx# Consolidated feature set for machine learning training
3. File Format and Metadata
Imaging Data: All MRI and mask files must be in NIfTI format.

Feature Files: Excel files must include the following essential columns:

MASK: Unique Subject/Sample ID.

quality: Clinical Pfirrmann Grade (I, II, III, IV, or V).

Alignment: The Subject IDs in the Excel files must match the corresponding file names in the images/ and masks/ folders.

4. Academic Data Access
The datasets used in this research may be provided by the corresponding author upon reasonable academic request, subject to institutional approval and data sharing agreements.