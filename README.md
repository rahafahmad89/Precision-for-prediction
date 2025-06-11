# 🧬 Precision in Prediction: Tailoring Machine Learning for Breast Cancer Missense Variants

This repository implements a complete machine learning pipeline for predicting the pathogenicity of **missense variants** in **breast cancer genes**, using disease-specific datasets. The framework includes preprocessing, feature selection, classifier benchmarking, multi-seed model evaluation, interpretability (LIME, permutation importance), calibration analysis, and statistical testing.

📝 **Associated Manuscript**:  
*“Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variant Pathogenicity Prediction”*  
📄 Manuscript Submitted  
📎 [GitHub Repository](https://github.com/rahafahmad89/Precision-for-prediction)

---

## 📂 Project Structure

```
.
├── data/                          # Input datasets and feature descriptions
├── scripts/                       # Phase 1, Phase 2, interpretability, plotting
│   ├── phase1_seed_evaluation.py
│   ├── phase2_best_seed_selection.py
│   ├── interpretability_lime.py
│   └── permutation_importance.py
├── results/                       # Metrics, plots, and visualizations
│   ├── combined_ROC_CI.png
│   ├── metrics_with_CI.xlsx
│   ├── best_seed_per_model.csv
│   ├── LIME_TP_TN_FP_FN.png
│   └── Permutation_Importance_ET.png
├── main.py                        # Unified pipeline execution
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📌 Highlights

- Disease-specific modeling for breast cancer variant pathogenicity
- Two-phase ML pipeline with best seed evaluation
- Interpretability with LIME on TP, TN, FP, FN samples
- Feature ranking via Permutation Importance
- Bootstrapped confidence intervals and statistical model validation
- Clinically oriented performance metric prioritization

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/rahafahmad89/Precision-for-prediction.git
cd Precision-for-prediction
```

### 2. Install dependencies
Using pip:
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
Place your input CSV file in `data/raw/`. A sample `Dataset-1.csv` is provided for structure reference.

### 4. Run the pipeline
```bash
python main.py
```

---

## 🔁 Two-Phase Evaluation Strategy

The workflow is structured in two phases for reproducibility and robustness:

- **Phase 1**: Initial training with a fixed seed (42) for baseline comparisons and interpretability setup.
- **Phase 2**: Evaluation across multiple seeds (42, 101, 202, 303, 404) to select the best-performing seed per classifier.

Interpretability (LIME and PMI), ROC with confidence intervals, and metric tables are generated using the best seed identified in Phase 2 for the best-performing model (Extra Trees).

---

## 🤖 Machine Learning Models Evaluated

- Extra Trees Classifier *(Best Performing)*
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

- `combined_ROC_CI.png` – ROC curves with 95% CI
- `metrics_with_CI.xlsx` – AUC, precision, recall, F1-score with 95% CI
- `best_seed_per_model.csv` – Best-performing random seed per classifier
- `LIME_TP_TN_FP_FN.png` – Local interpretability of TP, TN, FP, FN
- `Permutation_Importance_ET.png` – PMI for Extra Trees using optimal seed
- `statistical.xlsx` – z-tests, Shapiro-Wilk, Levene’s test, ANOVA results
- Calibration, PR, and feature heatmap plots

---

## 📚 Citation

If you use this repository, please cite:

```bibtex
@article{ahmad2025precision,
  author = {Rahaf M. Ahmad, Noura Al Dhaheri, Mohd Saberi Mohamad, Bassam R. Ali},
  title = {Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variant Pathogenicity Prediction},
  year = {2025},
  journal = {To be updated upon acceptance},
  url = {https://github.com/rahafahmad89/Precision-for-prediction}
}
```

---

## 🔐 License

Distributed under the MIT License. See the `LICENSE` file for details.

---

## 👩‍💻 Author

**Rahaf M. Ahmad**  
Ph.D. Candidate – Genetics & Machine Learning  
United Arab Emirates University  
ORCID: [0000-0002-7531-5264](https://orcid.org/0000-0002-7531-5264)

---

## 🤝 Acknowledgements

- This work is part of an ongoing effort to integrate interpretable AI into genomic variant classification for precision oncology.
