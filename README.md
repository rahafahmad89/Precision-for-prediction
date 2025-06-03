# 🧬 Precision in Prediction: Tailoring Machine Learning for Breast Cancer Missense Variants

This repository implements a complete machine learning pipeline for predicting the pathogenicity of **missense variants** in **breast cancer genes**, using disease-specific datasets. The framework includes data preprocessing, feature selection, classifier benchmarking, model evaluation, interpretability (LIME, permutation importance), calibration analysis, and statistical testing.

📝 **Associated Manuscript**:  
*“Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variants Pathogenicity Prediction”*  
📎 [Preprint or GitHub DOI Placeholder] (will be updated once published)

---

## 📂 Project Structure

```
.
├── data/              
├── outputs/            # Model results, figures, metrics
│   ├── models/
│   └── figures/
├── main.py
├── requirements.txt    # Pip dependencies
├── LICENSE             # License (MIT)
└── README.md           # Project documentation
```

---

## 🚀 How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/rahafahmad89/Precision-for-prediction.git
cd Precision-for-prediction

```

2. **Install dependencies**:
   - Using pip:
     ```bash
     pip install -r requirements.txt
     ```
   - you can also use Conda if preffered

3. **Prepare your dataset**:
Place your `dataset.csv` file inside `data/raw/` (Dataset-1 sample is provided as a template for the data structure and to test the workflow).

4. **Run the pipeline**:
```bash
python main.py
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

## 📚 Citation

If you use this repository, please cite as (will be updated once published):

```bibtex

  authors = {Rahaf M. Ahmad, Mohd Saberi Mohamad, Bassam R. Ali*},
  title = {Precision in Prediction: Tailoring Machine Learning for Breast Cancer Missense Variants },
  year = {2025},
  Link to GiHub repository annd publication doi = {\\url{https://github.com/rahafahmad89/Precision-for-prediction}},
  note = {MIT License}
}
```

---

## 🔐 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👩‍💻 Author

**Rahaf M. Ahmad**  
Ph.D. Candidate | Genetics & Machine Learning  
United Arab Emirates University  
ORCID: [0000-0002-7531-5264](https://orcid.org/0000-0002-7531-5264)

---

## 🤝 Acknowledgements

- Inspired by the need for robust and interpretable predictions in precision oncology.
