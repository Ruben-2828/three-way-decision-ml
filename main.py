from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import matplotlib

from utils.functions import (
    prepare_dataset, plot_negative_roc, plot_positive_roc, plot_calibration, plot_confusion_matrices, plot_histogram,
    plot_coverage_accuracy, pr_positive_with_beta, pr_negative_with_alpha, init_classification_metrics_csv,
    save_classification_metrics
)

datasets = ["bank", "adult"]  # Options: "bank" or/and "adult"

matplotlib.use('Agg')

for dataset in datasets:
    print(f"\n{'=' * 80}")
    print(f"PROCESSING DATASET: {dataset.upper()}")
    print(f"{'=' * 80}")

    X, y, dataset_name = prepare_dataset(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    beta_points = [0.70, 0.80, 0.90]
    alpha_points = [0.30, 0.20, 0.10]

    dataset_suffix = dataset_name.lower().replace(" ", "_")
    RUN_DIR = os.path.join("outputs", f"{dataset_suffix}")
    csv_report_path = os.path.join(RUN_DIR, f"{dataset_suffix}_report.csv")
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "pr_3wd"), exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "roc_3wd"), exist_ok=True)

    init_classification_metrics_csv(csv_report_path)

    # ==============================
    # Random Forest
    # ==============================
    print(f"\n>> Training Random Forest on {dataset_name} dataset...")

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Binary prediction
    y_pred_binary = clf.predict(X_test)

    accuracy_binary = accuracy_score(y_test, y_pred_binary)
    class_report_binary = classification_report(y_test, y_pred_binary, output_dict=True)
    save_classification_metrics(class_report_binary, "RF_binary", accuracy_binary, None, None, 0, dataset_name,
                                csv_report_path)

    # 3WD prediction
    probs = clf.predict_proba(X_test)[:, 1]

    for (beta, alpha) in zip(beta_points, alpha_points):
        decisions = np.where(probs >= beta, 1, np.where(probs <= alpha, 0, -1))

        accepted_indices = decisions != -1
        y_true_certain = y_test[accepted_indices]
        y_pred_certain = decisions[accepted_indices]

        accuracy_certain = accuracy_score(y_true_certain, y_pred_certain)
        deferred_ratio = np.mean(decisions == -1)
        class_report_certain = classification_report(y_true_certain, y_pred_certain, output_dict=True)
        save_classification_metrics(class_report_certain, "RF_3WD", accuracy_certain, alpha, beta, deferred_ratio,
                                    dataset_name, csv_report_path)

    print(">> Random Forest Training and Evaluation Completed.")

    # ==============================
    # Multi Layer Perceptron
    # ==============================

    print(f"\n>> Training Neural Network (MLP) on {dataset_name} dataset...")

    # Feature scaling for MLP
    scaler = StandardScaler()
    X_train_nn = scaler.fit_transform(X_train)
    X_test_nn = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
    )
    mlp.fit(X_train_nn, y_train)

    # Binary prediction
    y_pred_binary_nn = mlp.predict(X_test_nn)

    accuracy_binary_nn = accuracy_score(y_test, y_pred_binary_nn)
    class_report_binary_nn = classification_report(y_test, y_pred_binary_nn, output_dict=True)
    save_classification_metrics(class_report_binary_nn, "MLP_binary", accuracy_binary_nn, None, None, 0, dataset_name,
                                csv_report_path)

    # 3WD prediction
    probs_nn = mlp.predict_proba(X_test_nn)[:, 1]

    for (beta, alpha) in zip(beta_points, alpha_points):
        decisions_nn = np.where(probs_nn >= beta, 1, np.where(probs_nn <= alpha, 0, -1))

        accepted_indices_nn = decisions_nn != -1
        y_true_certain_nn = y_test[accepted_indices_nn]
        y_pred_certain_nn = decisions_nn[accepted_indices_nn]

        accuracy_certain_nn = accuracy_score(y_true_certain_nn, y_pred_certain_nn)
        deferred_ratio_nn = np.mean(decisions_nn == -1)
        class_report_certain_nn = classification_report(y_true_certain_nn, y_pred_certain_nn, output_dict=True)
        save_classification_metrics(class_report_certain_nn, "MLP_3WD", accuracy_certain_nn, alpha, beta,
                                    deferred_ratio_nn, dataset_name, csv_report_path)

    print(">> Neural Network Training and Evaluation Completed.")

    # ==============================================
    # 1) Coverage-Accuracy sweep (RF e MLP)
    # ==============================================

    # symmetric sweep: alpha = 1 - beta
    betas_cov_acc = np.linspace(0.55, 0.95, 21)
    alphas_cov_acc = 1.0 - betas_cov_acc

    plot_coverage_accuracy(y_test, probs, probs_nn, betas_cov_acc, alphas_cov_acc,
                           os.path.join(RUN_DIR, "coverage_accuracy_rf_mlp.png"))

    # ==============================================
    # 2) ROC/PR
    # ==============================================

    os.makedirs(os.path.join(RUN_DIR, "roc_3wd"), exist_ok=True)
    plot_positive_roc(y_test, probs, "RF", os.path.join(RUN_DIR, "roc_3wd", "rf_roc_positive.png"),
                      beta_points=beta_points)
    plot_positive_roc(y_test, probs_nn, "MLP", os.path.join(RUN_DIR, "roc_3wd", "mlp_roc_positive.png"),
                      beta_points=beta_points)
    plot_negative_roc(y_test, probs, "RF", os.path.join(RUN_DIR, "roc_3wd", "rf_roc_negative.png"),
                      alpha_points=alpha_points)
    plot_negative_roc(y_test, probs_nn, "MLP", os.path.join(RUN_DIR, "roc_3wd", "mlp_roc_negative.png"),
                      alpha_points=alpha_points)

    # ==========================================================
    # 3) Calibration diagrams + Brier
    # ==========================================================

    mask_rf_certain = decisions != -1
    plot_calibration(y_test, probs, beta_points, alpha_points, "RF", os.path.join(RUN_DIR, "rf_calibration.png"))

    mask_mlp_certain = decisions_nn != -1
    plot_calibration(y_test, probs_nn, beta_points, alpha_points, "MLP", os.path.join(RUN_DIR, "mlp_calibration.png"))

    # ==================================================
    # 4) Probability histograms
    # ==================================================

    plot_histogram(probs, probs_nn, "Distribuzione p(y=1|x)", os.path.join(RUN_DIR, "prob_hist.png"))

    # ==========================================================
    # 5) Confusion matrices
    # ==========================================================

    plot_confusion_matrices(y_test, y_pred_binary, probs, beta_points, alpha_points, "RF",
                            os.path.join(RUN_DIR, "rf_confusion_matrices.png"))

    plot_confusion_matrices(y_test, y_pred_binary_nn, probs_nn, beta_points, alpha_points, "MLP",
                            os.path.join(RUN_DIR, "mlp_confusion_matrices.png"))

    # ==========================================================
    # 6) Precision-Recall curves
    # ==========================================================
    pr_positive_with_beta(
        y_test.values if hasattr(y_test, 'values') else y_test,
        probs,
        "RF",
        beta_points,
        os.path.join(RUN_DIR, "pr_3wd"),
    )
    pr_positive_with_beta(
        y_test.values if hasattr(y_test, 'values') else y_test,
        probs_nn,
        "MLP",
        beta_points,
        os.path.join(RUN_DIR, "pr_3wd"),
    )
    pr_negative_with_alpha(
        y_test.values if hasattr(y_test, 'values') else y_test,
        probs,
        "RF",
        alpha_points,
        os.path.join(RUN_DIR, "pr_3wd"),
    )
    pr_negative_with_alpha(
        y_test.values if hasattr(y_test, 'values') else y_test,
        probs_nn,
        "MLP",
        alpha_points,
        os.path.join(RUN_DIR, "pr_3wd"),
    )

    # ==============================
    # Utility: show outputs folder
    # ==============================

    print(f"\n{'=' * 60}")
    print(f"THREE-WAY DECISION ANALYSIS - {dataset_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Output directory: {RUN_DIR}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"{'=' * 60}\n")

print(f"\n{'=' * 80}")
print("ALL DATASETS PROCESSED SUCCESSFULLY!")
print("Check the 'outputs/' directory for results")
print(f"{'=' * 80}")
