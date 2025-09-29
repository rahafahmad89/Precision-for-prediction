# 🧬 Precision in Prediction: Tailoring ML for Breast Cancer Missense Variants

This repository implements a **reproducible machine learning pipeline** for predicting the pathogenicity of **missense variants** in **breast cancer–associated genes**.  

The pipeline accompanies the manuscript:  
*“Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variant Pathogenicity Prediction”*  

📎 [GitHub Repository](https://github.com/rahafahmad89/Precision-for-prediction)  

---

## 📂 Repository Structure

```text
.
├── Test.csv                          # Provided standardized test set
├── model_artifacts/                  # Pre-trained model and supporting files
│   ├── Extra_Trees_model.joblib             # Best-performing trained model
│   ├── preprocessor.joblib           # Preprocessing pipeline
│   ├── meta.json                     # Model metadata
│   └── thresholds.json               # Thresholds for classification
├── phase-1-main.py                   # Phase 1 training (baseline seed = 42)
├── Phase-2-main.py                   # Phase 2 training (multi-seed, selects best)
├── testing.py                        # Testing pipeline (inference + evaluation)
├── requirements.txt                  # Full training dependencies
├── requirments_for_testing.txt       # Minimal dependencies for testing only (typo kept for consistency)
├── LICENSE
└── README.md
```

> **Note**: A `results/` directory is created automatically on first run to store metrics, plots, and predictions.

---

## 🚀 Quick Start (Testing Only)

You can reproduce the main results **without retraining** by using the provided model and dataset.

```bash
# 1. Install minimal dependencies
pip install -r requirments_for_testing.txt

# 2. Run testing with provided model + dataset
python testing.py   --model model_artifacts/best_model.joblib   --data  Test.csv   --out   results/test_predictions.csv
```

### Generated Outputs
- `results/test_predictions.csv` — predictions for each variant  
- `results/test_metrics.json` — AUC, F1, precision, recall, MCC, accuracy  
- `results/summary.txt` — human-readable performance summary  

---

## 🏋️ Full Training Workflow (Optional)

For full reproducibility, training can be repeated in **two phases**:

- **Phase 1**: Initial training with a fixed seed (42) for baseline comparisons and interpretability setup.  
- **Phase 2**: Evaluation across multiple seeds (42, 101, 202, 303, 404) to select the best-performing seed per classifier.  

Interpretability (LIME and PMI), ROC with confidence intervals, and metric tables are generated using the best seed identified in Phase 2 for the best-performing model.

### Phase 1 — Baseline Seed Evaluation
```bash
pip install -r requirements.txt
python phase-1-main.py --seed 42 --out_dir results/phase1
```

### Phase 2 — Multi-Seed Evaluation
```bash
python Phase-2-main.py --seeds 42 101 202 303 404 --out_dir results/phase2
```

This step identifies the best seed and saves the final model to:

```text
model_artifacts/best_model.joblib
```

---

## 🤖 Models Implemented

- Extra Trees (best-performing)  
- Random Forest  
- XGBoost  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- Naive Bayes  
- K-Nearest Neighbors (KNN)  
- AdaBoost  

---

## 📊 Outputs

The pipeline produces the following artifacts:

- ROC curves with confidence intervals → `combined_ROC_CI.png`  
- Bootstrapped metrics → `metrics_with_CI.xlsx`  
- Best seed summary → `best_seed_per_model.csv`  
- Interpretability plots → `LIME_TP_TN_FP_FN.png`, `Permutation_Importance.png`  
- Statistical tests → `statistical.xlsx`  
- Calibration and PR curves  

📌 **Reproducibility**  
- A trained model (`Extra_Trees_model.joblib`) and standardized test dataset (`Test.csv`) are included.  
- Minimal dependencies for testing are specified in `requirments_for_testing.txt`.  
- Full training and evaluation can be reproduced with `requirements.txt`.  

---

## 📚 Citation

If you use this repository, please cite (Will be updated once published):

```bibtex
@article{ahmad2025precision,
  author = {Ahmad, Rahaf M., Al Dhaheri, Noura, Mohamad, Mohd Saberi and Ali, Bassam R.*},
  title = {Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variant Pathogenicity Prediction},
  year = {2025},
  journal = {Briefings in Bioinformatics-in review},
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

## Acknowledgements
This work was supported by the United Arab Emirates University through Strategic Research Program (#12R111) and Research Start-up Program (#12M109). Rahaf M. Ahmad is supported by a PhD fellowship from the United Arab Emirates University.
This work is part of an ongoing effort to integrate interpretable AI into genomic variant classification for precision oncology.  
