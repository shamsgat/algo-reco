# dags/features_dag.py
"""
ML data features engineering DAG.

This DAG loads CSV files for the tables products, substitutions, and transactions
from GCS and ingests them into BigQuery.
"""

from airflow.sdk import dag, task
from pendulum import datetime
from scripts.ingestion import Ingestion
from scripts.utils import load_data_bq, dump_table_into_bq
from scripts.features import *

@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    doc_md=__doc__,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "features"]
)

def features_dag():
    
    ingestion = Ingestion()
    @task
    def build_features_bq(features_table_name : str, project_id : str, dataset_id : str, table_names: dict) -> str:
        """
        Load tables from BQ, then build features dataset and dump it into BQ.
        """ 
        produits = load_data_bq(project_id, dataset_id, table_names["produits"])
        transactions = load_data_bq(project_id, dataset_id, table_names["transactions"])
        substitutions = load_data_bq(project_id, dataset_id, table_names["substitutions"])
        
        df = build_features_dataset(produits, substitutions, transactions)
        
        dump_table_into_bq(df, project_id, dataset_id, features_table_name)
        
        return f'FEATURE ENG : {features_table_name} dumped into BQ'
    
    # BigQuery table to be loaded
    project_id = ingestion.project_id
    dataset_id = ingestion.dataset_id
    table_names = {"produits" : "produits", "transactions" : "transactions", "substitutions" : "substitutions"}
    
    features_table_name = "features_dataset"
    build_features_bq(features_table_name, project_id, dataset_id, table_names)
    
# Instantiate the DAG
features_dag()        