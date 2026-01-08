# dags/ingestion_dag.py
"""
ML data ingestion DAG.

Ingests TRAIN or INFERENCE data from GCS to BigQuery
according to the execution context.
"""

from airflow.sdk import dag, task
from pendulum import datetime
from airflow.models import Variable
from scripts.ingestion import Ingestion

@dag(
    start_date=datetime(2025, 1, 1),
    schedule=None,  # inference DAG pourra être daily
    catchup=False,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "ingestion"],
)
def ingestion_dag():

    ingestion = Ingestion()

    MODE = Variable.get("INGESTION_MODE", default_var="inference")
    # MODE ∈ {"train", "inference"}

    @task
    def ingest_from_gcs(gcs_path: str, table_name: str):
        ingestion.gcs_to_bq(
            gcs_path=gcs_path,
            table_name=table_name
        )

    # -------------------
    # TRAIN INGESTION
    # -------------------
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

        for table_name, gcs_path in ingestion_sources.items():
            ingest_from_gcs(
                gcs_path=gcs_path,
                table_name=table_name
            )

    # -------------------
    # INFERENCE INGESTION
    # -------------------
    else:
        # Variable contenant une ou plusieurs dates séparées par des virgules
        dates_str = Variable.get("INGESTION_DATES", default_var="")
        if not dates_str:
            raise ValueError("Airflow Variable 'INGESTION_DATES' must be set for inference mode.")

        dates_list = [d.strip() for d in dates_str.split(",")]

        for ds in dates_list:
            ingestion_sources = {
                "produits": "gs://algo_reco/raw/produits/produits.csv",
                f"transactions_inference_{ds}": (
                    f"gs://algo_reco/raw/transactions/"
                    f"transactions_inference/transactions_{ds}.csv"
                ),
                f"substitutions_inference_{ds}": (
                    f"gs://algo_reco/raw/substitutions/"
                    f"substitutions_inference/substitutions_{ds}.csv"
                ),
            }

            for table_name, gcs_path in ingestion_sources.items():
                ingest_from_gcs(
                    gcs_path=gcs_path,
                    table_name=table_name
                )

# Instantiate the DAG
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
