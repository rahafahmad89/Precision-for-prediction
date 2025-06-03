# 🧬 Precision in Prediction: Tailoring Machine Learning for Breast Cancer Missense Variants

This repository implements a complete machine learning pipeline for predicting the pathogenicity of **missense variants** in **breast cancer genes**, using disease-specific datasets. The framework includes data preprocessing, feature selection, classifier benchmarking, model evaluation, interpretability (LIME, permutation importance), calibration analysis, and statistical testing.

📝 **Associated Manuscript**:  
*“Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variants Pathogenicity Prediction”*  
📎 [Preprint or GitHub DOI Placeholder]

---

## 📂 Project Structure

```
.
├── data/               # Input and processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/          # Optional EDA or visualizations
├── outputs/            # Model results, figures, metrics
│   ├── models/
│   └── figures/
├── src/                # Source code for execution
│   └── main.py
├── tests/              # Unit tests (planned)
├── requirements.txt    # Pip dependencies
├── environment.yml     # Conda environment
├── LICENSE             # License (MIT)
└── README.md           # Project documentation
```

---

## 🚀 How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/variant-prediction-ML.git
cd variant-prediction-ML
```

2. **Install dependencies**:
   - Using pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Using conda:
     ```bash
     conda env create -f environment.yml
     conda activate bc_variant_predictor
     ```

3. **Prepare your dataset**:
Place your `dataset.csv` file inside `data/raw/`.

4. **Run the pipeline**:
```bash
python src/main.py
```

---

## 🔍 Pipeline Workflow

- Data cleaning, label encoding
- Pearson correlation filter (threshold = 0.9)
- Feature selection using RFE
- Classifier training and evaluation
- Performance metrics: AUC, precision, recall, F1-score, MCC, Cohen's kappa
- ROC and PR curve generation
- LIME interpretability per model
- Permutation Feature Importance
- Statistical tests: z-test, ANOVA, Shapiro-Wilk, Levene
- Output saving as .xlsx and .png files

---

## 🤖 Machine Learning Models Evaluated

- Extra Trees Classifier
- Random Forest Classifier
- XGBoost Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Naive Bayes
- K-Nearest Neighbors (KNN)
- AdaBoost

---

## 📊 Output Artifacts

- `results1.xlsx`, `results2.xlsx` – Metric scores
- `statistical.xlsx` – p-values and test statistics
- `roc_curve.png`, `precision_recall_curve.png`, `calibration_curve.png`
- `correlation_heatmap.png`
- LIME and permutation feature plots

---

## 📖 Citation

Please cite the following if you use this repository:

> Ahmad, R.M., Mohamad, M.S., Ali, B.R.  
> "Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variants Pathogenicity Prediction" (2025).  
> [GitHub Repository / Preprint Link Placeholder]

---

## 🔐 License

Distributed under the MIT License. See `LICENSE` for more information.