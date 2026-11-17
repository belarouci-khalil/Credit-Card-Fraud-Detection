# Régression logistique avec split 80/20

import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.evaluation import evaluate_model
from ui.dashboard import print_section


def run_logistic_regression(X, y):
    print_section("LOGISTIC REGRESSION - 80/20 SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records : {len(X_test):,}")

    # normalisation + régression logistique
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # entraînement
    print("\nTraining Logistic Regression...")

    start_train = time.perf_counter()
    lr_model.fit(X_train, y_train)
    train_time = (time.perf_counter() - start_train) * 1000

    # prédiction
    start_predict = time.perf_counter()
    predictions = lr_model.predict(X_test)
    probabilities = lr_model.predict_proba(X_test)[:, 1]
    predict_time = (time.perf_counter() - start_predict) * 1000

    print("Logistic Regression training completed!")

    # évaluation
    metrics = evaluate_model(
        "Logistic Regression",
        y_test,
        predictions,
        probabilities
    )

    return {
        "metrics": metrics,
        "train_time": train_time,
        "predict_time": predict_time,
        "y_test": y_test,
        "predictions": predictions,
    }