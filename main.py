# main.py
import logging
import pandas as pd
from pathlib import Path
from scripts.features import build_features_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

if __name__ == "__main__":

    # Définition des chemins
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "raw"

    print("📥 Chargement des données...")
    produits = pd.read_csv(data_dir / "produits.csv")
    substitutions = pd.read_csv(data_dir / "substitutions.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv")
    print(transactions.shape)

    print("⚙️ Lancement du processing des features...")
    df_features = build_features_dataset(
        produits=produits,
        substitutions=substitutions,
        transactions=transactions
    )


    print("✅ Processing terminé")
    print(f"📊 Shape du dataset : {df_features.shape}")
    print("🧪 Aperçu des données :")
    print(df_features.head())
