# Random Forest avec validation croisée à 6 folds

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

from utils.evaluation import evaluate_model
from ui.dashboard import print_section


def run_random_forest(X, y):
    print_section("RANDOM FOREST - 6 FOLD CROSS VALIDATION")

    print("Configuration:")
    print("  - Number of trees: 50")
    print("  - Maximum samples per tree: 80%")
    print("  - CPU parallelization: Enabled")
    print("  - Cross-validation: 6 folds")

    kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

    # on garde les prédictions de chaque fold pour l'évaluation globale
    rf_predictions = np.zeros(len(y), dtype=int)
    rf_probabilities = np.zeros(len(y), dtype=float)

    fold_results = []
    total_train = 0
    total_predict = 0

    print("\nStarting Random Forest training...\n")

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):

        print(f"\nProcessing Fold {fold}/6...")

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = RandomForestClassifier(
            n_estimators=50,
            max_samples=0.8,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        # temps d'entraînement
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = (time.perf_counter() - start_train) * 1000

        # temps de prédiction
        start_predict = time.perf_counter()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        predict_time = (time.perf_counter() - start_predict) * 1000

        # on sauvegarde les prédictions du fold
        rf_predictions[test_index] = y_pred
        rf_probabilities[test_index] = y_proba

        total_train += train_time
        total_predict += predict_time

        # métriques du fold
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        fold_results.append({
            "Fold": f"Fold {fold}",
            "Accuracy": accuracy,
            "F1-Score": f1,
            "Recall": recall,
            "Train (ms)": train_time,
            "Predict (ms)": predict_time
        })

        print(f"Fold {fold} completed!")
        print(f"  Accuracy     : {accuracy:.4f}")
        print(f"  F1-Score     : {f1:.4f}")
        print(f"  Recall       : {recall:.4f}")
        print(f"  Training     : {train_time:.2f} ms")
        print(f"  Prediction   : {predict_time:.2f} ms")

    # résultats de tous les folds
    results_df = pd.DataFrame(fold_results)
    average = results_df.drop(columns=["Fold"]).mean()

    avg_train = total_train / 6
    avg_predict = total_predict / 6

    # évaluation globale (toutes les prédictions out-of-fold)
    metrics = evaluate_model(
        "Random Forest",
        y.to_numpy(),
        rf_predictions,
        rf_probabilities
    )

    return {
        "results_df": results_df,
        "average": average,
        "avg_train": avg_train,
        "avg_predict": avg_predict,
        "metrics": metrics,
        "predictions": rf_predictions,
    }