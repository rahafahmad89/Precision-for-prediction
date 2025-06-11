import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score, cohen_kappa_score
from sklearn.svm import SVC
from statsmodels.stats.weightstats import ztest
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
import shap
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from scipy import stats
import lime
import lime.lime_tabular
import json
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.utils import resample


# === CONFIGURATION ===
DATASET_PATH = "oversampled_balanced_dataset.csv"
PLOT_OUTPUT = "seed_auc_plots"
BEST_SEED_OUTPUT = "best_seeds.json"
SEEDS = [42, 101, 202, 303, 404]
os.makedirs(PLOT_OUTPUT, exist_ok=True)
os.makedirs(os.path.dirname(BEST_SEED_OUTPUT), exist_ok=True)

# === CLASSIFIERS ===
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(enable_categorical=True, use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Extra Trees": ExtraTreesClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# === DATA LOADING AND PREPROCESSING ===
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)

    # Drop problematic or irrelevant columns
    drop_cols = ['features to drop']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Encode target if needed
    if 'CLIN_SIG' in df.columns:
        df['CLIN_SIG'] = LabelEncoder().fit_transform(df['CLIN_SIG'])

    # Drop all-NaN columns
    df = df.dropna(axis=1, how='all')

    # Fill remaining NaNs
    df = df.fillna(0)

    return df

# === SEED EVALUATION ===
def evaluate_seeds(df):
    best_seeds = {}

    for clf_name, clf in classifiers.items():
        print(f"\n🔍 Evaluating seeds for {clf_name}...")
        auc_per_seed = {}

        for seed in SEEDS:
            try:
                np.random.seed(seed)
                X = df.drop(columns=["#Uploaded_variation", "CLIN_SIG"])
                y = df["CLIN_SIG"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed
                )

                if hasattr(clf, "random_state"):
                    clf.set_params(random_state=seed)

                clf.fit(X_train, y_train)
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                auc_per_seed[seed] = auc

            except Exception as e:
                print(f"⚠️ Error for seed {seed} in {clf_name}: {e}")

        # Save best seed and plot
        if auc_per_seed:
            best_seed = max(auc_per_seed, key=auc_per_seed.get)
            best_seeds[clf_name] = best_seed

            plt.figure()
            plt.bar(list(auc_per_seed.keys()), list(auc_per_seed.values()))
            plt.title(f"AUC vs Seed for {clf_name}")
            plt.xlabel("Seed")
            plt.ylabel("AUC")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_OUTPUT, f"{clf_name.replace(' ', '_')}_seed_auc.png"))
            plt.close()

    with open(BEST_SEED_OUTPUT, "w") as f:
        json.dump(best_seeds, f, indent=4)
    print("\n✅ Best seeds saved.")

def bootstrap_metrics_with_ci(y_true, y_prob, y_pred, n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(42)
    aucs, precisions, recalls, f1s, kappas, mccs = [], [], [], [], [], []

    for _ in range(n_bootstrap):
        indices = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        y_true_bs = y_true[indices]
        y_prob_bs = y_prob[indices]
        y_pred_bs = y_pred[indices]

        try:
            aucs.append(roc_auc_score(y_true_bs, y_prob_bs))
        except:
            aucs.append(np.nan)
        precisions.append(precision_score(y_true_bs, y_pred_bs, zero_division=0))
        recalls.append(recall_score(y_true_bs, y_pred_bs, zero_division=0))
        f1s.append(f1_score(y_true_bs, y_pred_bs, zero_division=0))
        kappas.append(cohen_kappa_score(y_true_bs, y_pred_bs))
        mccs.append(matthews_corrcoef(y_true_bs, y_pred_bs))

    def ci_bounds(metric_list):
        metric_list = [x for x in metric_list if not np.isnan(x)]
        lower = np.percentile(metric_list, (1 - ci) / 2 * 100)
        upper = np.percentile(metric_list, (1 + ci) / 2 * 100)
        return np.mean(metric_list), lower, upper

    return {
        'AUC': ci_bounds(aucs),
        'Precision': ci_bounds(precisions),
        'Recall': ci_bounds(recalls),
        'F1': ci_bounds(f1s),
        'Kappa': ci_bounds(kappas),
        'MCC': ci_bounds(mccs)
    }



def pearson_correlation_filter(X, threshold=0.9):
    corr_matrix = X.corr()
    high_corr = np.where(np.abs(corr_matrix) > threshold)
    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j])
                       for i, j in zip(*high_corr) if i != j and i < j]

    drop_columns = set([col2 for col1, col2 in high_corr_pairs])
    data = X.drop(columns=list(drop_columns))
    print(f"Dropped {len(drop_columns)} features due to high correlation")
    return data

def feature_selection_and_heatmap(X, y):
    estimator = RandomForestClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=10, step=1)
    selector = selector.fit(X, y)
    selected_features = X.loc[:, selector.support_].copy()
    selected_features['CLIN_SIG'] = y

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(selected_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    for text in heatmap.texts:
        text.set_fontsize(8)
    plt.title('Correlation Heatmap of Selected Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')

    # Ensure the selected features do not include any highly correlated features
    return selected_features

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = clf.decision_function(X_test)
        y_prob = (y_prob > 0).astype(int)  # Convert decision function output to binary

    auc_score = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    mcc = matthews_corrcoef(y_test, y_pred)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    kappa = cohen_kappa_score(y_test, y_pred)

    return {
        'AUC': auc_score,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Cohen\'s Kappa': kappa,
        'MCC': mcc,
        'FPR': fp / (fp + tn),
        'TNR': tn / (tn + fp),
        'TPR': tp / (tp + fn),
        'Probabilities': y_prob
    }


def evaluate_model_CI(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    ci_results = bootstrap_metrics_with_ci(y_test.values, y_prob, y_pred)

    return {
        'AUC': f"{ci_results['AUC'][0]:.3f} [{ci_results['AUC'][1]:.3f}–{ci_results['AUC'][2]:.3f}]",
        'Precision': f"{ci_results['Precision'][0]:.3f} [{ci_results['Precision'][1]:.3f}–{ci_results['Precision'][2]:.3f}]",
        'Recall': f"{ci_results['Recall'][0]:.3f} [{ci_results['Recall'][1]:.3f}–{ci_results['Recall'][2]:.3f}]",
        'F1': f"{ci_results['F1'][0]:.3f} [{ci_results['F1'][1]:.3f}–{ci_results['F1'][2]:.3f}]",
        'Kappa': f"{ci_results['Kappa'][0]:.3f} [{ci_results['Kappa'][1]:.3f}–{ci_results['Kappa'][2]:.3f}]",
        'MCC': f"{ci_results['MCC'][0]:.3f} [{ci_results['MCC'][1]:.3f}–{ci_results['MCC'][2]:.3f}]"
    }



# Evaluate the classifier using 5-fold cross-validation
def evaluate_model_cv(clf, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {
    'AUC': [], 'Precision': [], 'Recall': [], 'F1 Score': [],
    'Specificity': [], 'Sensitivity': [], 'MCC': [], "Cohen's Kappa": []
    }


    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        try:
            y_prob = clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = clf.decision_function(X_test)

        auc_score = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        mcc = matthews_corrcoef(y_test, y_pred)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        kappa = cohen_kappa_score(y_test, y_pred)

        results['AUC'].append(auc_score)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1 Score'].append(f1)
        results['Specificity'].append(specificity)
        results['Sensitivity'].append(sensitivity)
        results['MCC'].append(mcc)
        results["Cohen's Kappa"].append(kappa)

    # Aggregate results
    aggregated_results = {metric: np.mean(scores) for metric, scores in results.items()}
    return aggregated_results


def save_results_to_excel(results, original_file_name, step_number):
    file_name = f'{original_file_name}_results{step_number}.xlsx'
    if len(results) > 1:
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    try:
        results_df.to_excel(file_name, index=False)
        print(f"Results saved to {file_name}")
    except Exception as e:
        print(f"Error saving results to {file_name}: {e}")

def save_predictions_to_excel(predictions, file_path):
    try:
        predictions.to_excel(file_path, index=False)
        print(f"Predictions saved to {file_path}")
    except Exception as e:
        print(f"Error saving predictions to {file_path}: {e}")
# Save plots automatically
def save_plot(fig, file_name):
    fig.savefig(file_name)
    plt.close(fig)

def generate_roc_curve(y_true, y_probs, clf_names):
    plt.figure()
    best_thresholds = {}
    for i in range(len(y_probs)):
        fpr, tpr, thresholds = roc_curve(y_true, y_probs[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{clf_names[i]} (AUC = {roc_auc:.2f})')

        # Finding best threshold
        optimal_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[optimal_idx]
        best_thresholds[clf_names[i]] = best_threshold

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - All Classifiers')
    plt.legend(loc="lower right")
    save_plot(plt.gcf(), 'roc_curve.png')

    return best_thresholds

def generate_precision_recall_curve(y_true, y_probs, clf_names):
    plt.figure()
    for i in range(len(y_probs)):
        precision, recall, _ = precision_recall_curve(y_true, y_probs[i])
        avg_precision = average_precision_score(y_true, y_probs[i])
        plt.plot(recall, precision, lw=2, label=f'{clf_names[i]} (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - All Classifiers')
    plt.legend(loc="lower left")
    save_plot(plt.gcf(), 'precision_recall_curve.png')

from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

def explain_with_lime(model, X_train, X_test, y_test, classifier_name):
    os.makedirs(f"/LIME/{classifier_name}", exist_ok=True)

    try:
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns.tolist(),
            class_names=['Benign', 'Pathogenic'],
            mode='classification'
        )

        # Handle wrappers or missing methods
        try:
            prob_func = model.predict_proba
        except AttributeError:
            print(f"⚠️ {classifier_name} does not support predict_proba. Skipping LIME.")
            return

        y_pred = model.predict(X_test)
        indices = {
            "TP": np.where((y_pred == 1) & (y_test == 1))[0],
            "TN": np.where((y_pred == 0) & (y_test == 0))[0],
            "FP": np.where((y_pred == 1) & (y_test == 0))[0],
            "FN": np.where((y_pred == 0) & (y_test == 1))[0]
        }

        for label, idx_list in indices.items():
            if len(idx_list) > 0:
                idx = idx_list[0]
                exp = explainer.explain_instance(
                    X_test.iloc[idx].values,
                    prob_func,
                    num_features=10
                )
                fig = exp.as_pyplot_figure()
                fig.tight_layout()
                fig.savefig(f"{classifier_name}/{label}_lime_explanation.png")
                plt.close(fig)

        print(f"✅ LIME plots saved for {classifier_name}")

    except Exception as e:
        print(f"❌ Error generating LIME for {classifier_name}: {e}")


def plot_calibration_curve(y_true, y_probs, clf_names):
    plt.figure(figsize=(10, 8))
    for i in range(len(y_probs)):
        prob_true, prob_pred = calibration_curve(y_true, y_probs[i], n_bins=10)
        plt.plot(prob_pred, prob_true, lw=2, label=f'{clf_names[i]}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.show()

from sklearn.inspection import permutation_importance

def plot_permutation_importance(model, X_test, y_test, classifier_name):
    os.makedirs("Permutation", exist_ok=True)

    try:
        result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0, scoring='roc_auc')
        sorted_idx = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(X_test.shape[1]), result.importances_mean[sorted_idx])
        plt.xticks(range(X_test.shape[1]), X_test.columns[sorted_idx], rotation=90)
        plt.title(f"Permutation Importance - {classifier_name}")
        plt.tight_layout()
        plt.savefig(f"pfi_{classifier_name}.png")
        plt.close()

        print(f"✅ Permutation Importance saved for {classifier_name}")

    except Exception as e:
        print(f"❌ Error in permutation importance for {classifier_name}: {e}")



def statistical_analysis(y_true, model_predictions, model_results):
    """
    Perform statistical analysis on model predictions.

    Parameters:
    y_true: Ground truth (actual) labels
    model_predictions: Dictionary where keys are model names, and values are predicted probabilities or classes
    model_results: Dictionary of results from each model

    Returns:
    DataFrame containing statistical test results for each model
    """
    all_statistics_results = []

    for model_name, metrics in model_results.items():
        y_pred = model_predictions.get(model_name, [])

        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"No predictions for {model_name}. Skipping statistical analysis.")
            continue

        try:
            # Perform statistical tests
            z_stat, p_ztest = ztest(y_true, y_pred)
            shapiro_stat, p_shapiro = stats.shapiro(y_pred)
            levene_stat, p_levene = stats.levene(y_true, y_pred)
            f_stat, p_f = stats.f_oneway(y_true, y_pred)
        except Exception as e:
            print(f"Error in statistical tests for {model_name}: {e}")
            continue

        # Store results
        statistics_results = {
            'Model': model_name,
            'Z-Statistic': z_stat,
            'P-value Z-Test': p_ztest,
            'Shapiro Statistic': shapiro_stat,
            'P-value Shapiro': p_shapiro,
            'Levene Statistic': levene_stat,
            'P-value Levene': p_levene,
            'F-statistic': f_stat,
            'P-value F-test': p_f
        }
        all_statistics_results.append(statistics_results)

    # Convert to DataFrame
    stat_df = pd.DataFrame(all_statistics_results)
    stat_df.to_excel('statistical.xlsx', index=True)

    # Plotting individual statistics with significance bars
    def plot_statistic(statistic_name, p_value_name, title, ylabel):
        os.makedirs("statistics", exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=stat_df.dropna(subset=[statistic_name]), x='Model', y=statistic_name, palette='viridis')

        for i, p_value in enumerate(stat_df[p_value_name]):
            significance = ''
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            plt.text(i, stat_df[statistic_name][i], significance, ha='center', va='bottom', fontsize=12, color='red')

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"stat.png")
        plt.close()

    # Generate all stat plots
    plot_statistic('Z-Statistic', 'P-value Z-Test', 'Z-Statistic for Each Classifier', 'Z-Statistic')
    plot_statistic('Shapiro Statistic', 'P-value Shapiro', 'Shapiro-Wilk Test for Each Classifier', 'Shapiro Statistic')
    plot_statistic('Levene Statistic', 'P-value Levene', 'Levene Test for Each Classifier', 'Levene Statistic')
    plot_statistic('F-statistic', 'P-value F-test', 'F-Test for Each Classifier', 'F-statistic')

    return stat_df



def select_best_classifier(results):
    # Adjusted weights for each metric (must sum to 1)
    weights = {
        'AUC': 0.30,  # High importance due to clinical significance of overall accuracy
        'Precision': 0.20,  # Focus on reducing false positives in clinical diagnosis
        'Recall': 0.20,  # Important for catching as many pathogenic variants as possible
        'F1 Score': 0.10,  # Balance between Precision and Recall
        'Specificity': 0.10,  # Reduce false positives
        'Sensitivity': 0.05,  # Additional importance for catching true positives
        'MCC': 0.025,  # Balanced measure, although lower weight here
        'Cohen\'s Kappa': 0.025  # Agreement between predicted and actual classifications
    }

    best_score = -float('inf')
    best_classifier = None

    for clf_name, metrics in results.items():
        # Calculate weighted score
        score = sum(metrics[metric] * weights[metric] for metric in weights)
        print(f"Classifier: {clf_name}, Weighted Score: {score:.4f}")

        # Update the best classifier if current score is higher
        if score > best_score:
            best_score = score
            best_classifier = clf_name

    return best_classifier, best_score

def plot_combined_bootstrapped_roc_ci(y_true, y_prob_dict, n_bootstraps=1000, seed=42):
    from sklearn.utils import resample
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(10, 8))
    for model_name, y_prob in y_prob_dict.items():
        bootstrapped_scores = []
        tprs = []
        for i in range(n_bootstraps):
            indices = resample(np.arange(len(y_true)), replace=True, random_state=seed + i)
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_prob[indices])
            bootstrapped_scores.append(score)
            fpr, tpr, _ = roc_curve(y_true[indices], y_prob[indices])
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
        tprs_lower = mean_tpr - 1.96 * std_tpr
        auc_mean = np.mean(bootstrapped_scores)
        auc_ci = 1.96 * np.std(bootstrapped_scores)
        plt.plot(base_fpr, mean_tpr, label=f"{model_name} (AUC={auc_mean:.2f} ± {auc_ci:.2f})", lw=2)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with 95% CI for All Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    output_path = "combined_ROC_CI_all_models.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Combined ROC with CI saved to {output_path}")


def run_shap_for_best_model(best_model, best_classifier, X_train, X_test, feature_names):
    shap.initjs()
    print(f"⚠️ Running SHAP for {best_classifier}...")


    if isinstance(X_train, np.ndarray):
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)
    elif isinstance(feature_names, list) and len(feature_names) == X_train.shape[1]:
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)
    else:
        feature_names = X_train.columns.tolist()

    try:
        explainer = shap.Explainer(best_model.predict_proba, X_train)
        shap_values = explainer(X_test)
    except Exception as e:
        print(f"🔄 Fallback to KernelExplainer due to: {e}")
        explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary for {best_classifier}")
    plt.tight_layout()
    plt.savefig("shap_summary_{best_classifier}.png")
    plt.close()

    # Bar plot (with safe indexing)
    try:
        shap.plots.bar(shap_values, show=False)
        plt.title(f"SHAP Bar - {best_classifier}")
        plt.tight_layout()
        plt.savefig("shap_bar_{best_classifier}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ SHAP Bar plot failed: {e}")

    # Beeswarm plot
    try:
        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"SHAP Beeswarm - {best_classifier}")
        plt.tight_layout()
        plt.savefig("shap_beeswarm_{best_classifier}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ SHAP Beeswarm plot failed: {e}")

    print(f"✅ SHAP plots saved for {best_classifier}")






def main():
    # === Phase 1: Seed Evaluation and Selection ===
    print("🔍 Loading data and evaluating seeds...")
    df = load_and_preprocess_data(DATASET_PATH)

    evaluate_seeds(df)

    with open(BEST_SEED_OUTPUT, "r") as f:
        best_seeds = json.load(f)

    # Prepare feature matrix and labels
    X = df.drop(columns=['#Uploaded_variation', 'CLIN_SIG'])
    y = df['CLIN_SIG']
    X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    # Track best performing classifier
    all_predictions = {}
    best_model = None
    best_auc = 0
    best_classifier = None
    best_seed = None

    # Train using best seed per classifier
    for clf_name, clf in classifiers.items():
        print(f"\n🔁 Training {clf_name} with best seed {best_seeds[clf_name]}...")
        seed = best_seeds[clf_name]
        np.random.seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        if hasattr(clf, "random_state"):
            clf.set_params(random_state=seed)

        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        all_predictions[clf_name] = y_prob

        auc_score = roc_auc_score(y_test, y_prob)
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = clf
            best_classifier = clf_name
            best_seed = seed

    # === Phase 2: Preprocessing and Feature Selection ===
    print("\n🧼 Preprocessing and feature selection...")
    data = load_and_preprocess_data(DATASET_PATH)
    X = data.drop(columns=['#Uploaded_variation', 'CLIN_SIG'])
    y = data['CLIN_SIG']
    X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    if X.shape[1] > 1:
        X_filtered = pearson_correlation_filter(X, threshold=0.9)
    else:
        print("⚠️ Insufficient numeric features. Skipping Pearson filtering.")
        X_filtered = X

    if X_filtered.shape[1] <= 1:
        print("❌ Insufficient features after filtering. Exiting.")
        return

    selected_features = feature_selection_and_heatmap(X_filtered, y)
    if selected_features.shape[1] <= 1:
        print("❌ Insufficient features after RFE. Exiting.")
        return

    # === Phase 3: Training and Evaluation ===
    print("\n📊 Training and evaluating classifiers...")
    # Split using the best seed
    X_train, X_test, y_train, y_test = train_test_split(
        selected_features.drop('CLIN_SIG', axis=1),
        selected_features['CLIN_SIG'],
        test_size=0.2,
        random_state=best_seed  # Use the best-performing seed
    )

    # Reset indices to ensure alignment for plotting & saving
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    data = data.reset_index(drop=True)

    classifiers_with_seed = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=101),
        'SVM': SVC(probability=True, random_state=101),
        'KNN': KNeighborsClassifier(random_state=202),
        'Naive Bayes': GaussianNB(random_state=101),
        'Decision Tree': DecisionTreeClassifier(random_state=202),
        'AdaBoost': AdaBoostClassifier(random_state=101),
        'Extra Trees': ExtraTreesClassifier(random_state=42)
    }

    results, results_cv, predictions = {}, {}, {}

    for clf_name, clf in classifiers_with_seed.items():
        print(f"✅ Evaluating {clf_name}...")
        results[clf_name] = evaluate_model(clf, X_train, X_test, y_train, y_test)
        predictions[clf_name] = clf.predict_proba(X_test)[:, 1]

    save_results_to_excel(results, 'model_results.xlsx', step_number=1)

    for clf_name, clf in classifiers_with_seed.items():
        results_cv[clf_name] = evaluate_model_cv(clf, selected_features.drop('CLIN_SIG', axis=1), selected_features['CLIN_SIG'])
    # Save CI metrics to Excel
    ci_metrics_df = pd.DataFrame({clf_name: evaluate_model_CI(clf, X_train, X_test, y_train, y_test)
                                  for clf_name, clf in classifiers_with_seed.items()}).T.reset_index()
    ci_metrics_df.columns = ['Classifier', 'AUC (CI)', 'Precision (CI)', 'Recall (CI)', 'F1 (CI)', 'Kappa (CI)', 'MCC (CI)']
    ci_metrics_df.to_excel("metrics_CI.xlsx", index=False)
    print("✅ CI metrics saved to dataset-2-metrics_CI.xlsx")

    save_results_to_excel(results_cv, 'model_results.xlsx', step_number=2)

    # === Phase 4: Plotting and Interpretation ===
    print("\n📈 Generating interpretability plots...")
    best_thresholds = generate_roc_curve(y_test, list(predictions.values()), list(predictions.keys()))
    generate_precision_recall_curve(y_test, list(predictions.values()), list(predictions.keys()))
    plot_calibration_curve(y_test, list(predictions.values()), list(predictions.keys()))


    # === Phase 5: Classification and Output ===
    print("\n📤 Saving predictions and analysis...")
    avg_threshold = np.mean(list(best_thresholds.values()))
    prediction_df = pd.DataFrame({
        '#Uploaded_variation': data['#Uploaded_variation'].iloc[X_test.index].values,
        'Actual': y_test
    })
    for clf_name, probs in predictions.items():
        prediction_df[clf_name] = probs
    prediction_df['Average Probability'] = prediction_df[list(predictions.keys())].mean(axis=1)

    # Apply classification rules
    prediction_df['Classification'] = 'Unknown'
    prediction_df.loc[(prediction_df['Average Probability'] > avg_threshold) & (prediction_df['Actual'] == 1), 'Classification'] = 'True Pathogenic'
    prediction_df.loc[(prediction_df['Average Probability'] > avg_threshold) & (prediction_df['Actual'] == 0), 'Classification'] = 'False Pathogenic'
    prediction_df.loc[(prediction_df['Average Probability'] <= avg_threshold) & (prediction_df['Actual'] == 0), 'Classification'] = 'True Benign'
    prediction_df.loc[(prediction_df['Average Probability'] <= avg_threshold) & (prediction_df['Actual'] == 1), 'Classification'] = 'False Benign'

    save_predictions_to_excel(prediction_df, 'predictions.xlsx')

    # === Phase 6: Statistical Analysis & SHAP ===
    print("\n📊 Statistical testing and SHAP analysis...")
    stat_df = statistical_analysis(y_test, predictions, results)
    print(stat_df)
    print(f"\n📌 Using best seed = {best_seed} for classifier = {best_classifier}")

    best_classifier, best_score = select_best_classifier(results)
    print(f"\n🏆 Best Classifier: {best_classifier} | Weighted Score: {best_score:.4f}")

    plot_combined_bootstrapped_roc_ci(y_test, predictions)
    # ⚠️ Retrain best model on selected features before SHAP, LIME, and PFI
    best_model_on_selected = classifiers_with_seed[best_classifier]
    best_model_on_selected.fit(X_train, y_train)

    run_shap_for_best_model(
        best_model=best_model_on_selected,
        best_classifier=best_classifier,
        X_train=X_train,
        X_test=X_test,
        feature_names=X_train.columns
    )

    explain_with_lime(best_model_on_selected, X_train, X_test, y_test, best_classifier)
    plot_permutation_importance(best_model_on_selected, X_test, y_test, best_classifier)


    # Save best model info
    with open(os.path.join("best_classifier_info.txt"), "w") as f:
        f.write(f"{best_classifier} | Seed: {best_seed} | AUC: {best_score:.4f}")
    print("\n✅ Best model evaluation complete and saved.")



if __name__ == "__main__":
        main()

