# Fonctions d'affichage dans le terminal

from sklearn.metrics import confusion_matrix, classification_report


def print_header():
    print("\n")
    print("=" * 85)
    print("                    CREDIT CARD FRAUD DETECTION")
    print("=" * 85)
    print("                  Random Forest vs Logistic Regression")
    print("=" * 85)


def print_section(title):
    print("\n")
    print("-" * 85)
    print(f" {title}")
    print("-" * 85)


def print_metrics(metrics, train_time=None, predict_time=None):
    print("\nEVALUATION METRICS")
    print("-" * 45)

    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        print(f"{metric:<20}: {metrics[metric]:.4f}")

    if train_time is not None:
        print(f"{'Training Time':<20}: {train_time:.2f} ms")

    if predict_time is not None:
        print(f"{'Prediction Time':<20}: {predict_time:.2f} ms")


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nCONFUSION MATRIX")
    print("-" * 55)

    print(f"{'Actual / Predicted':<22}{'Legitimate':>15}{'Fraud':>12}")
    print("-" * 55)

    print(f"{'Legitimate':<22}{cm[0, 0]:>15}{cm[0, 1]:>12}")
    print(f"{'Fraud':<22}{cm[1, 0]:>15}{cm[1, 1]:>12}")

    print("-" * 55)


def print_classification_report(y_true, y_pred):
    print("\nCLASSIFICATION REPORT")
    print("-" * 65)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )


def print_fold_results(results_df, average):
    print("\nRANDOM FOREST - 6 FOLD RESULTS")
    print("=" * 90)

    print(
        f"{'Fold':<12}"
        f"{'Accuracy':>12}"
        f"{'F1-Score':>12}"
        f"{'Recall':>12}"
        f"{'Train (ms)':>16}"
        f"{'Predict (ms)':>16}"
    )
    print("-" * 90)

    # une ligne par fold
    for _, row in results_df.iterrows():
        print(
            f"{row['Fold']:<12}"
            f"{row['Accuracy']:>12.4f}"
            f"{row['F1-Score']:>12.4f}"
            f"{row['Recall']:>12.4f}"
            f"{row['Train (ms)']:>16.2f}"
            f"{row['Predict (ms)']:>16.2f}"
        )

    print("-" * 90)

    # ligne de moyenne
    print(
        f"{'AVERAGE':<12}"
        f"{average['Accuracy']:>12.4f}"
        f"{average['F1-Score']:>12.4f}"
        f"{average['Recall']:>12.4f}"
        f"{average['Train (ms)']:>16.2f}"
        f"{average['Predict (ms)']:>16.2f}"
    )

    print("=" * 90)


def print_comparison(comparison):
    print("\n")
    print("=" * 115)
    print("                         FINAL MODEL COMPARISON")
    print("=" * 115)

    print(
        f"{'Model':<23}"
        f"{'Accuracy':>11}"
        f"{'Precision':>12}"
        f"{'Recall':>11}"
        f"{'F1-Score':>11}"
        f"{'ROC-AUC':>11}"
        f"{'Train(ms)':>13}"
        f"{'Predict(ms)':>14}"
    )
    print("-" * 115)

    for _, row in comparison.iterrows():
        print(
            f"{row['Model']:<23}"
            f"{row['Accuracy']:>11.4f}"
            f"{row['Precision']:>12.4f}"
            f"{row['Recall']:>11.4f}"
            f"{row['F1-Score']:>11.4f}"
            f"{row['ROC-AUC']:>11.4f}"
            f"{row['Train (ms)']:>13.2f}"
            f"{row['Predict (ms)']:>14.2f}"
        )

    print("=" * 115)