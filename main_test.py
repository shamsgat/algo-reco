from scripts.ingestion import Ingestion
from scripts.utils import (
    load_data_bq,
    dump_table_into_bq,
    dump_data_gcs,
    load_data_gcs
)
from scripts.features import build_features_dataset
from scripts.processing import (
    get_feature_lists,
    build_preprocessor,
    fit_transform_preprocessor
)

# ------------------------------------------------
# Simulation de la conf DAG (TriggerDagRunOperator)
# ------------------------------------------------
CONF = {
    "INGESTION_MODE": "inference",
    "INGESTION_DATES": ["2023-12-20", "2023-12-21"],
}

# ------------------------------------------------
# INGESTION
# ------------------------------------------------
def ingest_from_gcs(gcs_path: str, table_name: str):
    ingestion.gcs_to_bq(gcs_path=gcs_path, table_name=table_name)
    print(f"[INGESTION] {table_name} loaded from {gcs_path}")

def ingest_multiple_dates(dates: list):
    for ds in dates:
        ds_table = ds.replace("-", "_")

        ingestion_sources = {
            "produits": "gs://algo_reco/raw/produits/produits.csv",
            f"transactions_inference_{ds_table}":
                f"gs://algo_reco/raw/transactions/transactions_inference/transactions_{ds}.csv",
            f"substitutions_inference_{ds_table}":
                f"gs://algo_reco/raw/substitutions/substitutions_inference/substitutions_{ds}.csv",
        }

        for table_name, gcs_path in ingestion_sources.items():
            ingest_from_gcs(gcs_path, table_name)

# ------------------------------------------------
# FEATURES
# ------------------------------------------------
def build_features_for_dates(ingestion, dates: list):
    project_id = ingestion.project_id
    dataset_id = ingestion.dataset_id

    for ds in dates:
        ds_table = ds.replace("-", "_")
        print(f"[FEATURES] Building features for {ds}")

        produits = load_data_bq(project_id, dataset_id, "produits")
        transactions = load_data_bq(
            project_id, dataset_id, f"transactions_inference_{ds_table}"
        )
        substitutions = load_data_bq(
            project_id, dataset_id, f"substitutions_inference_{ds_table}"
        )

        df_features = build_features_dataset(
            produits, substitutions, transactions
        )

        features_table = f"features_inference_{ds_table}"
        dump_table_into_bq(
            df_features, project_id, dataset_id, features_table
        )

        print(f"[FEATURES] {features_table} created")

# ------------------------------------------------
# PROCESSING – PREPROCESSING (INFERENCE)
# ------------------------------------------------
def preprocessing_inference(ingestion, dates: list):
    project_id = ingestion.project_id
    dataset_id = ingestion.dataset_id

    # 👉 même feature schema que le train
    features_num, features_cat = get_feature_lists()
    features_dict = {
        "numerical": features_num,
        "categorical": features_cat,
    }

    # Sauvegardé une seule fois (comme dans le DAG train)
    dump_data_gcs(
        features_dict,
        "gs://algo_reco/features/train",
        "features",
    )

    # 👉 charger le preprocessor entraîné
    preprocessor = load_data_gcs(
        "gs://algo_reco/features/train/preprocessor.joblib"
    )

    for ds in dates:
        ds_table = ds.replace("-", "_")
        print(f"[PROCESSING] Preprocessing inference for {ds}")

        features_df = load_data_bq(
            project_id,
            dataset_id,
            f"features_inference_{ds_table}",
        )

        X = features_df[features_num + features_cat]

        X_transformed, _, _ = fit_transform_preprocessor(
            preprocessor, X, X
        )

        dump_data_gcs(
            X_transformed,
            "gs://algo_reco/features/inference",
            f"x_inference_preprocessed_{ds}",
        )

        print(
            f"[PROCESSING] x_inference_preprocessed_{ds} dumped to GCS"
        )

# ------------------------------------------------
# MAIN PIPELINE (LOCAL TEST)
# ------------------------------------------------
if __name__ == "__main__":
    ingestion = Ingestion()

    MODE = CONF.get("INGESTION_MODE", "inference")
    INGESTION_DATES = CONF.get("INGESTION_DATES", [])

    if MODE == "train":
        raise NotImplementedError(
            "Train mode preprocessing not tested in this main"
        )

    if not INGESTION_DATES:
        raise ValueError("INFERENCE mode requires INGESTION_DATES")

    print("🚀 START PIPELINE")

    # 1️⃣ INGESTION
    ingest_multiple_dates(INGESTION_DATES)

    # 2️⃣ FEATURES
    build_features_for_dates(ingestion, INGESTION_DATES)

    # 3️⃣ PROCESSING (PREPROCESSING)
    preprocessing_inference(ingestion, INGESTION_DATES)

    print("✅ PIPELINE Ingestion → Features → Processing completed")
