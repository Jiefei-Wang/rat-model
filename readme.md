# Detection of Frustration-Related Operant Behavior in Rats via Machine Learning

This repository implements data processing, feature extraction, Gradient Boosting model training/sweeps, and manuscript evaluation for detecting frustration-related operant behavior in rats from lever press force/time profiles.

## Project Structure

```
rat-model/
├── data/
│   ├── cohort1/
│   ├── cohort2/
│   ├── cohort3/
│   ├── cohort3_application_Gen 2 Sucrose PR/
│   └── Profiling-Weights_age_sex.xlsx
├── modules/
│   ├── data_management.py
│   ├── feature_extraction.py
│   ├── manuscript_utils.py
│   └── read_data.py
├── output/
│   ├── data/
│   ├── figures/
│   └── plots/
├── scripts/
│   ├── __init__.py
│   ├── data/
│   │   ├── 01_application_data.py
│   │   └── 01_raw_data.py
│   ├── manuscript/
│   │   ├── descriptive.ipynb
│   │   └── models.ipynb
│   ├── sweep_models/
│   │   ├── __init__.py
│   │   ├── sweep_GB.py
│   │   └── sweep_peak.py
│   └── sweep_params/
│       ├── parallel_sweep.py
│       ├── sweep_params_gb.py
│       └── sweep_params_peak.py
├── .gitignore
└── readme.md
```

## Workflow

1. **Data Processing & Feature Extraction**
   - Extract features from Cohorts 1, 2, and 3, saving processed datasets (`df_raw.pkl`, `df_ML.pkl`, `df_ML_train.pkl`, `df_ML_test.pkl`) to `output/data/`:
     ```bash
     python scripts/data/01_raw_data.py
     ```
   - Extract features from Cohort 3 progressive ratio (PR) application data (`df_app_ML.pkl`):
     ```bash
     python scripts/data/01_application_data.py
     ```

2. **Hyperparameter Tuning (W&B)**
   - Initialize the W&B grid search for Gradient Boosting hyperparameters:
     ```bash
     python scripts/sweep_params/sweep_params_gb.py
     ```
   - Launch parallel W&B agents using the generated sweep path:
     ```bash
     python scripts/sweep_params/parallel_sweep.py --agent-path <entity>/<project>/<sweep_id>
     ```
   - (Optional) Evaluate peak detection hyperparameters:
     ```bash
     python scripts/sweep_params/sweep_params_peak.py
     ```

3. **Manuscript Analysis & Results**
   - Open and run cohort demographics, Table 1, and Figure 3A:
     ```bash
     jupyter notebook scripts/manuscript/descriptive.ipynb
     ```
   - In `scripts/manuscript/models.ipynb`, manually update the Gradient Boosting hyperparameters in the model training cell to match the optimal values identified from the W&B sweep in Step 2.
   - Run the modeling notebook for univariate feature performance (Figure 3B), Gradient Boosting training, SHAP values (Figure 4), AUC degradation & chunking curves (Figure 5), test set boxplots (Figure 6), and progressive ratio inference (Figure 7):
     ```bash
     jupyter notebook scripts/manuscript/models.ipynb
     ```
