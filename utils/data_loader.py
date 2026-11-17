# Téléchargement et chargement du dataset

from pathlib import Path
import kagglehub
import pandas as pd


def load_data():
    print("Downloading Credit Card Fraud Detection dataset...")

    # téléchargement depuis Kaggle
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    # on cherche le fichier csv dans le dossier téléchargé
    csv_files = list(Path(path).rglob("*.csv"))

    if not csv_files:
        raise SystemExit("ERROR: No CSV file found.")

    df = pd.read_csv(csv_files[0])

    # on enlève les doublons
    df = df.drop_duplicates().reset_index(drop=True)

    # X = les features, y = la colonne cible
    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("\nDataset loaded successfully!")

    print(f"Total transactions       : {len(df):,}")
    print(f"Number of features       : {X.shape[1]}")
    print(f"Legitimate transactions  : {(y == 0).sum():,}")
    print(f"Fraudulent transactions   : {(y == 1).sum():,}")

    return X, y