# dags/ingestion_dag.py
"""
ML data ingestion DAG.

Ingests TRAIN or INFERENCE data from GCS to BigQuery
according to the execution context.

- TRAIN ingestion: full datasets
- INFERENCE ingestion: list of dates (daily inference)
Each date will generate its own table in BigQuery:
    transactions_inference_YYYY_MM_DD
    substitutions_inference_YYYY_MM_DD
"""

from airflow.sdk import dag, task
from pendulum import datetime
from airflow.operators.python import get_current_context
from scripts.ingestion import Ingestion

@dag(
    start_date=datetime(2025, 1, 1),
    schedule=None,  # peut être daily pour inference
    catchup=False,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "ingestion"],
)
def ingestion_dag():
    ingestion = Ingestion()

    @task
    def run_ingestion():
        """
        Tâche principale qui gère ingestion TRAIN ou INFERENCE
        """
        context = get_current_context()
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run else {}

        MODE = conf.get("INGESTION_MODE", "inference")
        INGESTION_DATES = conf.get("INGESTION_DATES", [])

        def ingest_from_gcs(gcs_path: str, table_name: str):
            """Charge un fichier GCS dans BigQuery"""
            ingestion.gcs_to_bq(gcs_path=gcs_path, table_name=table_name)
            print(f"[INFO] {table_name} chargé depuis {gcs_path}")

        if MODE == "train":
            ingestion_sources = {
                "produits": "gs://algo_reco/raw/produits/produits.csv",
                "transactions_train": "gs://algo_reco/raw/transactions/transactions_train/transactions_2023-01-01_2023-12-19.csv",
                "substitutions_train": "gs://algo_reco/raw/substitutions/substitutions_train/substitutions_2023-01-01_2023-12-19.csv",
            }
            for table_name, gcs_path in ingestion_sources.items():
                ingest_from_gcs(gcs_path, table_name)

        else:  # INFERENCE
            if not INGESTION_DATES:
                raise ValueError("INFERENCE mode requires INGESTION_DATES")

            for ds in INGESTION_DATES:
                ds_table = ds.replace("-", "_")  # format table compatible BigQuery
                ingestion_sources = {
                    "produits": "gs://algo_reco/raw/produits/produits.csv",
                    f"transactions_inference_{ds_table}": f"gs://algo_reco/raw/transactions/transactions_inference/transactions_{ds}.csv",
                    f"substitutions_inference_{ds_table}": f"gs://algo_reco/raw/substitutions/substitutions_inference/substitutions_{ds}.csv",
                }
                for table_name, gcs_path in ingestion_sources.items():
                    ingest_from_gcs(gcs_path, table_name)

    run_ingestion()

# Instanciation du DAG
ingestion_dag()

# ------------------------------------------------------------------------------------
# # dags/ingestion_dag.py
# """
# ML data ingestion DAG.

# This DAG loads CSV files for the tables products, substitutions, and transactions
# from GCS and ingests them into BigQuery.
# """

# from airflow.sdk import dag, task
# from pendulum import datetime
# from scripts.ingestion import Ingestion

# @dag(
#     start_date=datetime(2025, 1, 1),
#     schedule=None,
#     doc_md=__doc__,
#     default_args={"owner": "ML", "retries": 3},
#     catchup=False,
#     tags=["ml", "ingestion"],
# )
# def ingestion_dag():

#     ingestion = Ingestion()

#     @task
#     def ingest_table_from_gcs(gcs_path: str, table_name: str) -> str:
#         """
#         Task that reads a CSV from GCS and loads it into BigQuery.
#         """
#         ingestion.gcs_to_bq(gcs_path=gcs_path, table_name=table_name)
#         return f"{table_name} loaded successfully."
    
#     @task
#     def export_table_to_gcs(table_name: str, gcs_path: str, filename: str) -> str:
#         """
#         Task that exports a BigQuery table to GCS.
#         """
#         ingestion.bq_to_gcs(table_name=table_name, gcs_path=gcs_path, filename=filename)
#         return f"{table_name} exported successfully to GCS."

#      # Table definitions for ingestion
#     ingestion_tables = {
#         "produits": "gs://algo_reco/raw/produits/produits.csv",
#         "substitutions": "gs://algo_reco/raw/substitutions/substitutions.csv",
#         "transactions": "gs://algo_reco/raw/transactions/transactions.csv",
#     }

#     # Table definitions for export
#     export_tables = {
#         "produits": {"gcs_path": "gs://algo_reco/staging", "filename": "produits.csv"},
#         "substitutions": {"gcs_path": "gs://algo_reco/staging", "filename": "substitutions.csv"},
#         "transactions": {"gcs_path": "gs://algo_reco/staging", "filename": "transactions.csv"},
#     }

#     for table_name, paths in ingestion_tables.items():
#         ingest_table_from_gcs(paths, table_name)

# # Instantiate the DAG
# ingestion_dag()
