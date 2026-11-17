# Projet : détection de fraude par carte bancaire
# Random Forest (6-Fold CV) vs Régression Logistique (80/20)

import pandas as pd

from utils.data_loader import load_data
from models.random_forest import run_random_forest
from models.logistic_regression import run_logistic_regression
from ui.dashboard import (
    print_header,
    print_section,
    print_metrics,
    print_confusion_matrix,
    print_classification_report,
    print_fold_results,
    print_comparison,
)


print_header()

# ---------- 1. Chargement des données ----------
print_section("DATASET LOADING")
X, y = load_data()

# ---------- 2. Random Forest ----------
rf = run_random_forest(X, y)

print_section("RANDOM FOREST RESULTS")
print_fold_results(rf["results_df"], rf["average"])
print_metrics(rf["metrics"], rf["avg_train"], rf["avg_predict"])
print_confusion_matrix(y.to_numpy(), rf["predictions"])
print_classification_report(y.to_numpy(), rf["predictions"])

# ---------- 3. Régression logistique ----------
lr = run_logistic_regression(X, y)

print_metrics(lr["metrics"], lr["train_time"], lr["predict_time"])
print_confusion_matrix(lr["y_test"], lr["predictions"])
print_classification_report(lr["y_test"], lr["predictions"])

# ---------- 4. Comparaison finale ----------
print_section("FINAL MODEL COMPARISON")

comparison = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Accuracy": rf["metrics"]["Accuracy"],
        "Precision": rf["metrics"]["Precision"],
        "Recall": rf["metrics"]["Recall"],
        "F1-Score": rf["metrics"]["F1-Score"],
        "ROC-AUC": rf["metrics"]["ROC-AUC"],
        "Train (ms)": rf["avg_train"],
        "Predict (ms)": rf["avg_predict"],
    },
    {
        "Model": "Logistic Regression",
        "Accuracy": lr["metrics"]["Accuracy"],
        "Precision": lr["metrics"]["Precision"],
        "Recall": lr["metrics"]["Recall"],
        "F1-Score": lr["metrics"]["F1-Score"],
        "ROC-AUC": lr["metrics"]["ROC-AUC"],
        "Train (ms)": lr["train_time"],
        "Predict (ms)": lr["predict_time"],
    },
])

print_comparison(comparison)

# ---------- 5. Meilleur modèle selon le F1-Score ----------
best_model = comparison.loc[comparison["F1-Score"].idxmax()]

print("\nBEST MODEL ACCORDING TO F1-SCORE")
print("-" * 55)
print(f"Model     : {best_model['Model']}")
print(f"F1-Score  : {best_model['F1-Score']:.4f}")
print(f"Precision : {best_model['Precision']:.4f}")
print(f"Recall    : {best_model['Recall']:.4f}")

# ---------- 6. Sauvegarde des résultats ----------
rf["results_df"].to_csv("random_forest_6fold_results.csv", index=False)
comparison.to_csv("model_comparison.csv", index=False)

print_section("EXPERIMENT COMPLETED")
print("Results saved successfully:")
print("  1. random_forest_6fold_results.csv")
print("  2. model_comparison.csv")

print("\nThank you for using the Credit Card Fraud Detection system!")
print("=" * 85)