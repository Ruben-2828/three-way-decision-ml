import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, ConfusionMatrixDisplay, confusion_matrix, \
    accuracy_score, precision_recall_curve, average_precision_score
from sklearn.utils import resample
from ucimlrepo import fetch_ucirepo
import csv


###############################################
# Dataset loading functions
###############################################


def load_bank_marketing_dataset():
    print("Loading Bank Marketing dataset...")
    bank_marketing = fetch_ucirepo(id=222)

    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    print("Dataset metadata:")
    print(bank_marketing.metadata)
    print("Dataset variables:")
    print(bank_marketing.variables)

    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]

    if y.dtype == object or str(y.dtype).startswith('category'):
        y_mapped = y.map({'yes': 1, 'no': 0})
        if not y_mapped.isna().any():
            y = y_mapped.astype(int)

    return X, y, "Bank Marketing"


def load_adult_census_dataset():
    print("Loading Adult Census Income dataset...")

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]

    try:
        df = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)
        print(f"Loaded {len(df)} samples from Adult Census Income dataset")
    except Exception as e:
        print(f"Error loading dataset from URL: {e}")
        print("Trying alternative approach...")
        # Alternative: try to load from local file if URL fails
        return None, None, None

    df_clean = df.dropna()
    print(f"After removing missing values: {len(df_clean)} samples")

    X = df_clean.drop('income', axis=1)
    y = df_clean['income']

    # Balance the dataset by creating a subset of 14k samples (7k per class)
    print("Creating balanced subset...")
    df_majority = df_clean[df_clean['income'] == '<=50K']
    df_minority = df_clean[df_clean['income'] == '>50K']

    n_samples_per_class = 7000
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=n_samples_per_class,
        random_state=42
    )
    df_minority_downsampled = resample(
        df_minority,
        replace=False,
        n_samples=n_samples_per_class,
        random_state=42
    )
    df_balanced = pd.concat([df_majority_downsampled, df_minority_downsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_balanced.drop('income', axis=1)
    y = df_balanced['income']

    print(f"Balanced dataset created with {len(df_balanced)} samples")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Map target to binary: '<=50K' -> 0, '>50K' -> 1
    y = y.map({'<=50K': 0, '>50K': 1}).astype(int)

    return X, y, "Adult Census Income"


def prepare_dataset(dataset_name):
    if dataset_name.lower() == "bank":
        X, y, name = load_bank_marketing_dataset()
    elif dataset_name.lower() == "adult":
        X, y, name = load_adult_census_dataset()
    else:
        raise ValueError("Dataset must be 'bank' or 'adult'")

    if X is None or y is None:
        raise ValueError(f"Failed to load {dataset_name} dataset")

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)

    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y, name

# ==============================
# CSV report functions
# ==============================


def init_classification_metrics_csv(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'accuracy', 'precision', 'recall', 'f1', 'support', 'alpha', 'beta', 'defer_ratio',
                         'class', 'dataset'])


def save_classification_metrics(report_dict, model, accuracy, alpha, beta, defer, dataset, csv_path):
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        for cls in ['0', '1']:
            precision = round(report_dict[cls]['precision'], 2)
            recall = round(report_dict[cls]['recall'], 2)
            f1 = round(report_dict[cls]['f1-score'], 2)
            support = int(report_dict[cls]['support'])
            row = [model, round(accuracy, 2), precision, recall, f1, support, alpha, beta, round(defer, 2), int(cls),
                   dataset]
            writer.writerow(row)

# ==============================
# ROC/PR functions
# ==============================


def plot_positive_roc(y_true, prob_values, model_name, out_path, alpha_points):
    fpr, tpr, _ = roc_curve(y_true, prob_values)
    auc = roc_auc_score(y_true, prob_values)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

    for a in alpha_points:
        y_pred_a = (prob_values >= a).astype(int)
        tp_a = np.sum((y_true == 1) & (y_pred_a == 1))
        fp_a = np.sum((y_true == 0) & (y_pred_a == 1))
        tn_a = np.sum((y_true == 0) & (y_pred_a == 0))
        fn_a = np.sum((y_true == 1) & (y_pred_a == 0))
        fpr_a = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0.0
        tpr_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        plt.scatter([fpr_a], [tpr_a], label=f"alpha={a:.2f}", s=35)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - Positive class ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_negative_roc(y_true, prob_values, model_name, out_path, beta_points):
    y_arr = y_true.values if hasattr(y_true, 'values') else y_true
    y_neg = (1 - y_arr).astype(int)
    p_neg = 1.0 - prob_values

    fpr_neg, tpr_neg, _ = roc_curve(y_neg, p_neg)
    auc_neg = roc_auc_score(y_neg, p_neg)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_neg, tpr_neg, label=f"{model_name} ROC Neg (AUC={auc_neg:.3f})")

    for b in beta_points:
        thr_b = 1.0 - b
        y_pred_b = (p_neg >= thr_b).astype(int)
        tp_b = np.sum((y_neg == 1) & (y_pred_b == 1))
        fp_b = np.sum((y_neg == 0) & (y_pred_b == 1))
        tn_b = np.sum((y_neg == 0) & (y_pred_b == 0))
        fn_b = np.sum((y_neg == 1) & (y_pred_b == 0))
        fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0
        tpr_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        plt.scatter([fpr_b], [tpr_b], label=f"beta={b:.2f}")

    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("FPR (neg)")
    plt.ylabel("TPR (neg)")
    plt.title(f"ROC - Negative class ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ==============================
# Calibration functions
# ==============================
def plot_calibration(y_true, prob_values, alphas, betas, title, out_name):
    plt.figure(figsize=(6, 5))

    frac_pos_all, mean_pred_all = calibration_curve(y_true, prob_values, n_bins=10, strategy="uniform")
    brier_all = brier_score_loss(y_true, prob_values)

    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfectly calibrated")
    plt.plot(mean_pred_all, frac_pos_all, marker="o", label=f"All (Brier={brier_all:.3f})")

    for (a, b) in zip(alphas, betas):
        decisions = np.where(prob_values >= a, 1, np.where(prob_values <= b, 0, -1))
        decisions_mask = decisions != -1

        y_certain = y_true[decisions_mask]
        p_certain = prob_values[decisions_mask]
        frac_pos_certain, mean_pred_certain = calibration_curve(
            y_certain,
            p_certain,
            n_bins=min(10, max(2, int(np.sqrt(len(p_certain))))),
            strategy="uniform"
        )
        brier_certain = brier_score_loss(y_certain, p_certain)

        plt.plot(mean_pred_certain, frac_pos_certain,
                 marker="s", label=f"alpha={a}, beta={b} (Brier={brier_certain:.3f})")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration - {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()

# ==============================
# Confusion matrix functions
# ==============================


def plot_confusion_matrices(y_true, y_pred_binary_model, probs, alphas, betas, title, out_name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred_binary_model, labels=[0, 1]),
        display_labels=[0, 1]
    ).plot(ax=axes[0][0], colorbar=False, cmap="Blues", values_format="d")
    n_all = len(y_true)
    axes[0][0].set_title(f"{title} - Binary (n={n_all})")

    for i, (a, b) in enumerate(zip(alphas, betas), 1):
        decisions = np.where(probs >= a, 1, np.where(probs <= b, 0, -1))
        accepted_indices = decisions != -1
        y_true_certain = y_true[accepted_indices]
        y_pred_certain = decisions[accepted_indices]

        ConfusionMatrixDisplay(
            confusion_matrix(y_true_certain, y_pred_certain, labels=[0, 1]),
            display_labels=[0, 1]
        ).plot(ax=axes[i // 2][i % 2], colorbar=False, cmap="Blues", values_format="d")
        n_certain = len(y_true_certain)
        axes[i // 2][i % 2].set_title(f"{title} - Alpha={a}, Beta={b} - (n={n_certain})")

    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()

# ==============================
# Histogram functions
# ==============================
def plot_histogram(prob_rf, prob_mlp, title, out_name):
    plt.figure(figsize=(6, 4))
    plt.hist(prob_rf, bins=30, color="#4C78A8", alpha=0.7, label="RF", edgecolor="#4C78A8")
    plt.hist(prob_mlp, bins=30, color="#F58518", alpha=0.5, label="MLP", edgecolor="#F58518")
    plt.xlabel("Positive class probability")
    plt.ylabel("Count")
    plt.title(f"{title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()

# ==============================
# Coverage Accuracy functions
# ==============================
def coverage_accuracy_curve(y_true, prob_values, alphas, betas):
    coverages = []
    accuracies = []
    for a, b in zip(alphas, betas):
        decisions_tmp = np.where(prob_values >= a, 1, np.where(prob_values <= b, 0, -1))
        mask = decisions_tmp != -1
        coverage = np.mean(mask)
        if coverage > 0:
            acc = accuracy_score(y_true[mask], decisions_tmp[mask])
        else:
            acc = np.nan
        coverages.append(coverage)
        accuracies.append(acc)
    return np.array(coverages), np.array(accuracies)


def plot_coverage_accuracy(y_test, probs_rf, probs_mlp, alphas, betas, out_name):
    cov_rf, acc_rf = coverage_accuracy_curve(y_test, probs_rf, alphas, betas)
    cov_mlp, acc_mlp = coverage_accuracy_curve(y_test, probs_mlp, alphas, betas)

    plt.figure(figsize=(6, 5))
    plt.plot(cov_rf, acc_rf, marker="o", label="RF")
    plt.plot(cov_mlp, acc_mlp, marker="s", label="MLP")
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title(f"Coverage vs Accuracy (3WD sweep)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()

# ==============================
# Precision-Recall functions
# ==============================
def pr_positive_with_alpha(y_true, prob_values, model_name, alpha_points, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prec, rec, _ = precision_recall_curve(y_true, prob_values)
    ap = average_precision_score(y_true, prob_values)

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"{model_name} PR (AP={ap:.3f})")

    for a in alpha_points:
        y_pred_alpha = (prob_values >= a).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_alpha == 1))
        fp = np.sum((y_true == 0) & (y_pred_alpha == 1))
        fn = np.sum((y_true == 1) & (y_pred_alpha == 0))
        prec_alpha = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_alpha = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        plt.scatter([rec_alpha], [prec_alpha], label=f"alpha={a:.2f}", marker="o")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall with OPs - {model_name}")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name.lower()}_pr_multi_3wd.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def pr_negative_with_beta(y_true, prob_values, model_name, beta_points, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    y_neg = 1 - y_true
    p_neg = 1 - prob_values
    prec, rec, _ = precision_recall_curve(y_neg, p_neg)
    ap = average_precision_score(y_neg, p_neg)

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"{model_name} PR Neg (AP={ap:.3f})")

    for b in beta_points:
        thr_neg = 1.0 - b
        y_pred_thr = (p_neg >= thr_neg).astype(int)
        tp = np.sum((y_neg == 1) & (y_pred_thr == 1))
        fp = np.sum((y_neg == 0) & (y_pred_thr == 1))
        fn = np.sum((y_neg == 1) & (y_pred_thr == 0))
        prec_b = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_b = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        plt.scatter([rec_b], [prec_b], label=f"beta={b:.2f}", marker="x")

    plt.xlabel("Recall (neg)")
    plt.ylabel("Precision (neg)")
    plt.title(f"Precision-Recall with OPs (Negative class) - {model_name}")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name.lower()}_pr_negative_beta.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
