# Intelligence_BM
Code to publish article on hybrid loss function
## Dataset
This study uses the Alzheimer's MRI 4 Classes Dataset available on IEEE data port
https://ieee-dataport.org/documents/alzheimer-disease-classification-data
Due to dataset size considerations, the dataset is not included in this repository. Users can download the dataset directly from Kaggle and place it in the appropriate data directory before running the experiments.
Show more lines
## Loss-Apr26.py
Loss function and their shapes used in this paper 
## NLINEX-structure.py
NLINEX loss profiles for six parameter pairs, demonstrating the eect of k and c
on the combined exponential, linear, and quadratic contributions, including the optimised
pair (2.02, 0.206).
## PCA-ALL-1.py
Start
│
├── Import libraries
├── Set hyperparameters
├── Define loss functions
├── Build models
├── Load images
├── PCA variance plot
├── Bayesian optimization
├── Create optimized loss functions
├── 5-fold Cross Validation
│      ├── CNN Baseline
│      ├── NeuroNet + NLINEX
│      ├── NeuroNet + Focal
│      ├── NeuroNet + Lovasz
│      ├── NeuroNet + BCE
│      ├── NeuroNet + MSE
│      ├── NeuroNet + MAE
│      └── NeuroNet + Dice
├── Overall performance metrics
├── Wilcoxon significance test
├── Confusion matrices
├── Noise robustness
├── Class imbalance experiment
└── End
# PCA-ALL-2.py
Start
│
├── Import libraries
├── Set hyperparameters
├── Define loss functions
├── Build models
├── Load images
├── PCA variance plot
├── Bayesian optimization
├── Create optimized loss functions
├── 5-fold Cross Validation
│      ├── CNN Baseline
│      ├── NeuroNet + NLINEX
│      ├── NeuroNet + Focal
│      ├── NeuroNet + Lovasz
│      ├── NeuroNet + BCE
│      ├── NeuroNet + MSE
│      ├── NeuroNet + MAE
│      └── NeuroNet + Dice
├── Overall performance metrics
├── Wilcoxon significance test
├── Confusion matrices
├── Noise robustness
├── Class imbalance experiment
├── ★ Class-wise metrics from confusion matrices
│      ├── NonDemented
│      │      ├── Sensitivity
│      │      ├── Specificity
│      │      ├── Precision
│      │      └── F1-score
│      └── MildDemented
│             ├── Sensitivity
│             ├── Specificity
│             ├── Precision
│             └── F1-score
└── End
## Commons and difference between two pipeline (PCA-ALL-1 vs PCA-ALL-2)
# Common
Functionally, the training pipeline is identical in both files. They both:
Load the same data.
Use the same preprocessing (StandardScaler + PCA).
Use the same models (NeuroNet and CNN).
Use the same seven loss functions.
Use the same Bayesian optimization.
Use the same 5-fold × 5-run evaluation.
Compute the same overall metrics (Accuracy, Precision, Recall, F1, Specificity, ROC-AUC).
Perform the same statistical test, noise robustness experiment, and class imbalance experiment.
# Differenec
The only substantive addition in PCA-ALL-2.py is after all experiments have finished:
It takes the existing confusion matrices.
Computes per-class metrics (Sensitivity, Specificity, Precision, and F1-score) for NonDemented and MildDemented.
Prints these additional results.
No model training, optimization, preprocessing, or evaluation methodology changes between the two pipelines. The difference is an extra reporting stage that provides more detailed analysis of the confusion matrices.
