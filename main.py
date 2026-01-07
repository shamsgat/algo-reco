# main.py
import logging
from flask import json
import pandas as pd
from pathlib import Path
from scripts.features import build_features_dataset
from scripts.processing import (
    define_target,
    get_feature_lists,
    select_features,
    temporal_train_test_split,
    build_preprocessor,
    fit_transform_preprocessor,
)


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
    
    with open(data_dir / "features.json", "r") as f:
        features = json.load(f)
    print(f'features = {features}')
    print(features['numerical'])

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
    
    # ------------------------------------------------------------------
    # Processing (TEST)
    # ------------------------------------------------------------------
    logging.info("MAIN : Starting processing pipeline test")

    # Target
    y = define_target(df_features)

    # Features
    features_num, features_cat = get_feature_lists()
    X = select_features(df_features, features_num, features_cat)

    # Temporal split
    X_train_raw, X_test_raw, y_train, y_test = temporal_train_test_split(
        X,
        y,
        train_ratio=0.8,
    )

    # Preprocessing
    preprocessor = build_preprocessor(features_num, features_cat)

    X_train, X_test, preprocessor = fit_transform_preprocessor(
        preprocessor,
        X_train_raw,
        X_test_raw,
    )
    
    #DUMPING  X_train, X_test, y_train, y_test to GCS
    from scripts.utils import dump_data_gcs
    dump_data_gcs(pd.DataFrame(X_train),"gs://algo_reco/features/train","x_train_transformed")
    dump_data_gcs(pd.DataFrame(X_test), "gs://algo_reco/features/train","x_test_transformed")
    dump_data_gcs(pd.DataFrame(y_train), "gs://algo_reco/features/train","y_train")
    dump_data_gcs(pd.DataFrame(y_test), "gs://algo_reco/features/train","y_test")

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    logging.info("MAIN : Processing pipeline test completed successfully")
    logging.info(
        "MAIN : Final shapes | X_train=%s | X_test=%s | y_train=%s | y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )

    print("✅ Processing terminé avec succès")
