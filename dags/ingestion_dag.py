"""
ML data ingestion DAG.

This DAG loads CSV files for the tables products, substitutions, and transactions
from GCS and ingests them into BigQuery.
"""

from airflow.sdk import dag, task
from pendulum import datetime
from scripts.ingestion import Ingestion

@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    doc_md=__doc__,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "ingestion"],
)
def ingestion_dag():

    ingestion = Ingestion()

    @task
    def ingest_table_from_gcs(gcs_path: str, table_name: str) -> str:
        """
        Task that reads a CSV from GCS and loads it into BigQuery.
        """
        ingestion.gcs_to_bq(gcs_path=gcs_path, table_name=table_name)
        return f"{table_name} loaded successfully."
    
    @task
    def export_table_to_gcs(table_name: str, gcs_path: str, filename: str) -> str:
        """
        Task that exports a BigQuery table to GCS.
        """
        ingestion.bq_to_gcs(table_name=table_name, gcs_path=gcs_path, filename=filename)
        return f"{table_name} exported successfully to GCS."

     # Table definitions for ingestion
    ingestion_tables = {
        "produits": "gs://algo_reco/raw/produits/produits.csv",
        "substitutions": "gs://algo_reco/raw/substitutions/substitutions.csv",
        "transactions": "gs://algo_reco/raw/transactions/transactions.csv",
    }

    # Table definitions for export
    export_tables = {
        "produits": {"gcs_path": "gs://algo_reco/staging", "filename": "produits.csv"},
        "substitutions": {"gcs_path": "gs://algo_reco/staging", "filename": "substitutions.csv"},
        "transactions": {"gcs_path": "gs://algo_reco/staging", "filename": "transactions.csv"},
    }

    for table_name, paths in ingestion_tables.items():
        ingest_table_from_gcs(paths, table_name)

    for table_name, params in export_tables.items():
        export_table_to_gcs(table_name=table_name, gcs_path=params["gcs_path"],filename=params["filename"])
# Instantiate the DAG
ingestion_dag()
