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
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold


def set_random_seeds(seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.info())
    print(data.describe())

    if 'CLIN_SIG' in data.columns:
        le = LabelEncoder()
        data['CLIN_SIG'] = le.fit_transform(data['CLIN_SIG'])


    # Drop columns that should not be used for training
    columns_to_drop = [not needed features]
    data.drop(columns=columns_to_drop, inplace=True)

    return data

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
    selected_features['clinical-significance'] = y

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(selected_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    for text in heatmap.texts:
        text.set_fontsize(8)
    plt.title('Correlation Heatmap of Selected Features')
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


def plot_lime_explanations(clf, X_train, X_test, clf_name, num_features=10):
    """
    Generate LIME explanations for a classifier.

    Parameters:
    clf: Trained classifier
    X_train: Training data for LIME to learn distribution
    X_test: Test data where the instance to explain comes from
    clf_name: Name of the classifier
    num_features: Number of features to include in the LIME explanation
    """

    # Initialize the LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,  # Training data
        feature_names=X_train.columns,  # Feature names
        class_names=['0', '1'],  # Class names (binary classification)
        mode='classification'
    )

    # Select an instance to explain
    instance_to_explain = X_test.iloc[0]  # First instance from the test set

    # Get the explanation for the selected instance
    exp = explainer.explain_instance(
        data_row=instance_to_explain,  # Instance data
        predict_fn=clf.predict_proba,  # Classifier's predict_proba function
        num_features=num_features  # Number of features to display in the explanation
    )

    # Print the explanation in the console
    print(f"LIME explanation for {clf_name}:")
    exp.show_in_notebook(show_table=True)

    # Plot the explanation
    exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for {clf_name}")
    plt.show()
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

def permutation_importance_analysis(clf, X_train, y_train, clf_name):
    perm_importance = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42)
    perm_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance Mean': perm_importance.importances_mean,
        'Importance Std': perm_importance.importances_std
    }).sort_values(by='Importance Mean', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance Mean', y='Feature', data=perm_importance_df)
    plt.title(f'Permutation Feature Importance for {clf_name}')
    plt.show()


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
        plt.show()

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

def main():
    set_random_seeds(42)
    file_path = 'dataset.csv'
    data = load_and_preprocess_data(file_path)

    # Splitting the data into features and target variable
    X = data.drop(columns=['#Uploaded_variation', 'CLIN_SIG'])
    y = data['CLIN_SIG']

    # Debug numeric columns
    print("Numeric columns before filtering:", X.select_dtypes(include=[np.number]).columns)
    print("Total numeric features:", len(X.select_dtypes(include=[np.number]).columns))

    # Convert object columns to numeric where possible
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')  # Drop columns with all NaN values
    print(f"Numeric columns after conversion: {X.columns.tolist()}")
    print(f"Total numeric features after conversion: {X.shape[1]}")

    # Applying Pearson correlation filter
    if X.shape[1] > 1:
        filtered_X = pearson_correlation_filter(X, threshold=0.9)
    else:
        print("Insufficient numeric features. Skipping Pearson correlation filtering.")
        filtered_X = X

    print(f"Number of features after correlation filtering: {filtered_X.shape[1]}")
    if filtered_X.shape[1] <= 1:
        print("Insufficient features after filtering. Exiting program.")
        return

    # Feature selection and heatmap generation
    selected_features = feature_selection_and_heatmap(filtered_X, y)

    # Check if sufficient features are selected
    if selected_features.shape[1] <= 1:
        print("Insufficient features after RFE. Exiting program.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(selected_features.drop('clinical-significance', axis=1),
                                                        selected_features['clinical-significance'],
                                                        test_size=0.2, random_state=42)

    print("Unique values in y_test:", np.unique(y_test))
    # Proceed with modeling and evaluation

    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42)
    }

    results = {}
    results_cv = {}
    predictions = {}  

    for clf_name, clf in classifiers.items():
        print(f'Evaluating {clf_name}...')
        result = evaluate_model(clf, X_train, X_test, y_train, y_test)
        results[clf_name] = result
        y_prob = clf.predict_proba(X_test)[:, 1]  
        predictions[clf_name] = y_prob  # Store each model's predictions as a key-value pair

    save_results_to_excel(results, 'results.xlsx',1)



    for clf_name, clf in classifiers.items():
        print(f"Evaluating {clf_name}...")
        results_cv[clf_name] = evaluate_model_cv(clf, selected_features.drop('CLIN_SIG', axis=1), selected_features['CLIN_SIG'])

    # Save results
    save_results_to_excel(results_cv, 'results.xlsx',2)
    best_thresholds = generate_roc_curve(y_test, list(predictions.values()), list(classifiers.keys()))

    print("Best thresholds for each classifier:")
    for clf_name, threshold in best_thresholds.items():
        print(f"{clf_name}: {threshold:.2f}")

    generate_precision_recall_curve(y_test, list(predictions.values()), list(classifiers.keys()))

    for clf_name, clf in classifiers.items():
        plot_lime_explanations(clf, X_train, X_test, clf_name, num_features=10)
        permutation_importance_analysis(clf, X_train, y_train, clf_name)

    plot_calibration_curve(y_test, list(predictions.values()), list(classifiers.keys()))

    # Pass the predictions dictionary to statistical_analysis
    stat_df = statistical_analysis(y_test, predictions, results)
    print(stat_df)
  # Select the best classifier based on the weighted scores
    best_classifier, best_score = select_best_classifier(results)
    print(f"\nBest Classifier: {best_classifier} with a score of {best_score:.4f}")


if __name__ == "__main__":
        main()
