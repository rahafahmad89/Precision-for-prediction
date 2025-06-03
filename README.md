# Precision in Prediction: Tailoring Machine Learning for Breast Cancer Missense Variants

This repository contains the implementation code for a breast cancer missense variant pathogenicity prediction pipeline, as described in the manuscript:
"Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variants Pathogenicity Prediction".

## 📂 Project Structure

```
.
├── data/                     # Folder for input CSV datasets
├── outputs/                 # Stores results, plots, and evaluation files
├── src/                     # Python scripts for training and evaluation
│   └── main.py              # Main script for end-to-end execution
├── notebooks/               # Optional exploratory analysis
├── requirements.txt         # Required Python packages
├── README.md                # Project overview and usage instructions
└── LICENSE                  # Licensing information
```

## 🧪 Requirements

Install all packages using pip:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Place your dataset (`dataset.csv`) in the `data/` folder.
2. Run the pipeline with:
```bash
python src/main.py
```

## 📊 Outputs

- `results1.xlsx`, `results2.xlsx`: Metric outputs from regular and cross-validation
- `roc_curve.png`, `precision_recall_curve.png`: Performance plots
- `statistical.xlsx`: Statistical analysis results
- `correlation_heatmap.png`: Feature correlation heatmap

## 🧠 Models Evaluated

- Random Forest
- XGBoost
- Logistic Regression
- SVM
- KNN
- Naive Bayes
- Decision Tree
- AdaBoost
- Extra Trees

## 📖 Citation

If you use this code, please cite:

Ahmad, R.M., Mohamad, M.S., Ali, B.R. "Precision in Prediction: Tailoring Machine Learning Models for Breast Cancer Missense Variants Pathogenicity Prediction", [Preprint Manuscript, 2025].

## 🔐 License

This project is licensed under the MIT License - see the LICENSE file for details.
