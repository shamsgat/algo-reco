import os
import logging
from datetime import datetime
from scripts.ingestion import Ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logger.info("🚀 Starting ingestion DAG test (local)")

    ingestion = Ingestion()

    # ---------------------------------------------------
    # MODE : 'train' ou 'inference'
    # ---------------------------------------------------
    MODE = os.getenv("INGESTION_MODE", "inference")  # simule la Variable Airflow
    logger.info(f"Using INGESTION_MODE = {MODE}")

    # ---------------------------------------------------
    # Construire ingestion_sources
    # ---------------------------------------------------
    if MODE == "train":
        ingestion_sources = {
            "produits": "gs://algo_reco/raw/produits/produits.csv",
            "transactions_train": (
                "gs://algo_reco/raw/transactions/"
                "transactions_train/transactions_2023-01-01_2023-12-19.csv"
            ),
            "substitutions_train": (
                "gs://algo_reco/raw/substitutions/"
                "substitutions_train/substitutions_2023-01-01_2023-12-19.csv"
            ),
        }
    else:
        # Pour tester l’inference localement, on peut fixer la date ici
        ds = datetime.today().strftime("%Y-%m-%d")
        ingestion_sources = {
            "produits": "gs://algo_reco/raw/produits/produits.csv",
            f"transactions_inference": (
                f"gs://algo_reco/raw/transactions/transactions_inference/transactions_{ds}.csv"
            ),
            f"substitutions_inference": (
                f"gs://algo_reco/raw/substitutions/substitutions_inference/substitutions_{ds}.csv"
            ),
        }

    # ---------------------------------------------------
    # Simuler la tâche ingest_from_gcs
    # ---------------------------------------------------
    for table_name, gcs_path in ingestion_sources.items():
        logger.info(f"📥 Ingesting table '{table_name}' from {gcs_path}")
        ingestion.gcs_to_bq(gcs_path=gcs_path, table_name=table_name)
        logger.info(f"✅ Table '{table_name}' ingested successfully")

    logger.info("🎉 Ingestion DAG test completed successfully")
