# 🧬 Precision in Prediction: Tailoring ML for Breast Cancer Missense Variants

This repository implements a **reproducible machine learning pipeline** for predicting the pathogenicity of **missense variants** in **breast cancer–associated genes**.  

The pipeline accompanies the manuscript:  
*“Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variant Pathogenicity Prediction”*  

📎 [GitHub Repository](https://github.com/rahafahmad89/Precision-for-prediction)  

---

## 📂 Repository Structure

```text
.
├── data/
│   ├── test_data.csv                 # Provided standardized test set
│   └── README_DATA.md                # Data schema and field descriptions
├── models/
│   └── ExtraTrees_best_seed.pkl      # Pre-trained Extra Trees model (best seed)
├── results/                          # Results generated on first run
│   └── (predictions, metrics, plots)
├── phase1_main.py                    # Phase 1 training (baseline seed = 42)
├── phase2_main.py                    # Phase 2 training (multi-seed, selects best)
├── testing.py                        # Testing pipeline (inference + evaluation)
├── requirements.txt                  # Full training dependencies
├── requirements_for_testing.txt      # Minimal dependencies for testing only
├── LICENSE
└── README.md
```

## 🚀 Quick Start (Testing Only)
You can reproduce the main results without retraining by using the provided model and dataset.

bash
Copy code
# 1. Install minimal dependencies
pip install -r requirements_for_testing.txt

# 2. Run testing with provided model + dataset
python testing.py \
  --model models/ExtraTrees_best_seed.pkl \
  --data  data/test_data.csv \
  --out   results/test_predictions.csv



## Generated outputs:

results/test_predictions.csv — predictions for each variant

results/test_metrics.json — AUC, F1, precision, recall, MCC, accuracy

results/summary.txt — human-readable performance summary

## 🏋️ Full Training Workflow (Optional)
For full reproducibility, the training can be repeated in two phases.
The workflow is structured in two phases for reproducibility and robustness:

- **Phase 1**: Initial training with a fixed seed (42) for baseline comparisons and interpretability setup.
- **Phase 2**: Evaluation across multiple seeds (42, 101, 202, 303, 404) to select the best-performing seed per classifier.

Interpretability (LIME and PMI), ROC with confidence intervals, and metric tables are generated using the best seed identified in Phase 2 for the best-performing model (Extra Trees).

## Phase 1 — Baseline seed evaluation
bash
Copy code
pip install -r requirements.txt
python phase1_main.py --seed 42 --out_dir results/phase1
Phase 2 — Multi-seed evaluation
bash
Copy code
python phase2_main.py --seeds 42 101 202 303 404 --out_dir results/phase2

This step identifies the best seed and saves the final model to:

text
Copy code
models/ExtraTrees_best_seed.pkl


## 🤖 Models Implemented
Extra Trees (best-performing)

Random Forest

XGBoost

Logistic Regression

Support Vector Machine (SVM)

Decision Tree

Naive Bayes

K-Nearest Neighbors (KNN)

AdaBoost

## 📊 Outputs
The pipeline produces the following artifacts:

ROC curves with confidence intervals → combined_ROC_CI.png

Bootstrapped metrics → metrics_with_CI.xlsx

Best seed summary → best_seed_per_model.csv

Interpretability plots → LIME_TP_TN_FP_FN.png, Permutation_Importance_ET.png

Statistical tests → statistical.xlsx

Calibration and PR curves

📌 Reproducibility
A trained Extra Trees model and a standardized test dataset are included.

Minimal dependencies for testing are specified in requirements_for_testing.txt.

Full training and evaluation can be reproduced with requirements.txt.
```


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
