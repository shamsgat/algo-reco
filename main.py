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
import joblib  
from scripts.utils import load_data_gcs, dump_data_gcs
from scripts.inference import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

if __name__ == "__main__":

    # # Définition des chemins
    # base_dir = Path(__file__).resolve().parent
    # data_dir = base_dir / "data" / "raw"

    # print("📥 Chargement des données...")
    # produits = pd.read_csv(data_dir / "produits.csv")
    # substitutions = pd.read_csv(data_dir / "substitutions.csv")
    # transactions = pd.read_csv(data_dir / "transactions.csv")
    # print(transactions.shape)
    
    # with open(data_dir / "features.json", "r") as f:
    #     features = json.load(f)
    # print(f'features = {features}')
    # print(features['numerical'])

    # print("⚙️ Lancement du processing des features...")
    # df_features = build_features_dataset(
    #     produits=produits,
    #     substitutions=substitutions,
    #     transactions=transactions
    # )


    # print("✅ Processing terminé")
    # print(f"📊 Shape du dataset : {df_features.shape}")
    # print("🧪 Aperçu des données :")
    # print(df_features.head())
    
    # # ------------------------------------------------------------------
    # # Processing (TEST)
    # # ------------------------------------------------------------------
    # logging.info("MAIN : Starting processing pipeline test")

    # # Target
    # y = define_target(df_features)

    # # Features
    # features_num, features_cat = get_feature_lists()
    # X = select_features(df_features, features_num, features_cat)

    # # Temporal split
    # X_train_raw, X_test_raw, y_train, y_test = temporal_train_test_split(
    #     X,
    #     y,
    #     train_ratio=0.8,
    # )

    # # Preprocessing
    # preprocessor = build_preprocessor(features_num, features_cat)

    # X_train, X_test, preprocessor = fit_transform_preprocessor(
    #     preprocessor,
    #     X_train_raw,
    #     X_test_raw,
    # )
    
    # #DUMPING  X_train, X_test, y_train, y_test to GCS
    # from scripts.utils import dump_data_gcs
    # dump_data_gcs(pd.DataFrame(X_train),"gs://algo_reco/features/train","x_train_transformed")
    # dump_data_gcs(pd.DataFrame(X_test), "gs://algo_reco/features/train","x_test_transformed")
    # dump_data_gcs(pd.DataFrame(y_train), "gs://algo_reco/features/train","y_train")
    # dump_data_gcs(pd.DataFrame(y_test), "gs://algo_reco/features/train","y_test")

    # # ------------------------------------------------------------------
    # # Sanity checks
    # # ------------------------------------------------------------------
    # logging.info("MAIN : Processing pipeline test completed successfully")
    # logging.info(
    #     "MAIN : Final shapes | X_train=%s | X_test=%s | y_train=%s | y_test=%s",
    #     X_train.shape,
    #     X_test.shape,
    #     y_train.shape,
    #     y_test.shape,
    # )

    # print("✅ Processing terminé avec succès")
    
    # ------------------------------------------------------------------
    # Inference test
    # ------------------------------------------------------------------  
    logging.info("INFERENCE : Starting local inference test")

    # ----------------------------
    # GCS paths
    # ----------------------------
    MODEL_GCS_PATH = "gs://algo_reco/models/best_model.joblib"
    BEST_PARAMS_GCS_PATH = "gs://algo_reco/models/best_params.json"
    Z_TRANSFORMED_GCS_PATH = "gs://algo_reco/features/train/x_test_preprocessed.csv"

    OUTPUT_GCS_PATH = "gs://algo_reco/predictions"
    OUTPUT_FILENAME = "x_test_with_predictions"

    THRESHOLD = 0.5

    # ----------------------------
    # Load model
    # ----------------------------
    logging.info("INFERENCE : Loading trained model from %s", MODEL_GCS_PATH)
    model = load_data_gcs(MODEL_GCS_PATH)

    # sécurité si load_data_gcs retourne un path local
    if isinstance(model, str):
        logging.info("INFERENCE : Loading model via joblib from %s", model)
        model = joblib.load(model)
    logging.info("INFERENCE : Model loaded successfully (%s)", type(model).__name__)
    
    # ----------------------------
    # Load best params
    # ----------------------------
    logging.info("INFERENCE : Loading best params from %s", BEST_PARAMS_GCS_PATH)
    best_params = load_data_gcs(BEST_PARAMS_GCS_PATH)

    if not isinstance(best_params, dict):
        raise TypeError(
            "INFERENCE : best_params must be a dict"
        )

    # ----------------------------
    # Load transformed features
    # ----------------------------
    logging.info("INFERENCE : Loading transformed features from %s",Z_TRANSFORMED_GCS_PATH)
    Z_transformed = load_data_gcs(Z_TRANSFORMED_GCS_PATH)

    if not isinstance(Z_transformed, pd.DataFrame):
        raise TypeError(
            "INFERENCE : Z_transformed must be a pandas DataFrame"
        )

    # ----------------------------
    # Run inference
    # ----------------------------
    Z_pred = run_inference(
        model=model,
        best_params=best_params,
        Z_transformed=Z_transformed,
        threshold=THRESHOLD,
        add_proba=True
    )

    logging.info("INFERENCE : Inference done – result shape = %s",Z_pred.shape)

    # ----------------------------
    # Dump results
    # ----------------------------
    dump_data_gcs(
        data=Z_pred,
        path=OUTPUT_GCS_PATH,
        filename=OUTPUT_FILENAME
    )

    logging.info("INFERENCE : Predictions successfully dumped to GCS")

