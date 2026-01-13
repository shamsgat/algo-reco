# dags/features_dag.py
"""
ML data features engineering DAG.

Supports:
- TRAIN mode: build one features table from *_train tables
- INFERENCE mode: build one features table per date
"""

from airflow.sdk import dag, task
from airflow.operators.python import get_current_context
from pendulum import datetime

from scripts.ingestion import Ingestion
from scripts.utils import load_data_bq, dump_table_into_bq
from scripts.features import build_features_dataset


@dag(
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "features"],
)
def features_dag():

    ingestion = Ingestion()
    project_id = ingestion.project_id
    dataset_id = ingestion.dataset_id

    @task
    def run_features():
        """
        Main task handling TRAIN / INFERENCE feature engineering
        """
        context = get_current_context()
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run else {}

        MODE = conf.get("INGESTION_MODE", "inference")
        INGESTION_DATES = conf.get("INGESTION_DATES", [])

        def load_and_build_features(
            produits_table: str,
            transactions_table: str,
            substitutions_table: str,
            output_table: str,
        ):
            produits = load_data_bq(project_id, dataset_id, produits_table)
            transactions = load_data_bq(project_id, dataset_id, transactions_table)
            substitutions = load_data_bq(project_id, dataset_id, substitutions_table)

            df = build_features_dataset(produits, substitutions, transactions)
            dump_table_into_bq(df, project_id, dataset_id, output_table)

            print(f"[INFO] Features table {output_table} created")

        # -------------------
        # TRAIN MODE
        # -------------------
        if MODE == "train":
            load_and_build_features(
                produits_table="produits",
                transactions_table="transactions_train",
                substitutions_table="substitutions_train",
                output_table="features_train",
            )

        # -------------------
        # INFERENCE MODE
        # -------------------
        else:
            if not INGESTION_DATES:
                raise ValueError("INFERENCE mode requires INGESTION_DATES")

            for ds in INGESTION_DATES:
                ds_table = ds.replace("-", "_")

                load_and_build_features(
                    produits_table="produits",
                    transactions_table=f"transactions_inference_{ds_table}",
                    substitutions_table=f"substitutions_inference_{ds_table}",
                    output_table=f"features_inference_{ds_table}",
                )

    run_features()


# Instantiate DAG
features_dag()

# ------------------------------------------------------------------------------
# # dags/features_dag.py
# """
# ML data features engineering DAG.

# This DAG loads CSV files for the tables products, substitutions, and transactions
# from GCS and ingests them into BigQuery.
# """

# from airflow.sdk import dag, task
# from pendulum import datetime
# from scripts.ingestion import Ingestion
# from scripts.utils import load_data_bq, dump_table_into_bq
# from scripts.features import *

# @dag(
#     start_date=datetime(2025, 1, 1),
#     schedule=None,
#     doc_md=__doc__,
#     default_args={"owner": "ML", "retries": 3},
#     catchup=False,
#     tags=["ml", "features"]
# )

# def features_dag():
    
#     ingestion = Ingestion()
#     @task
#     def build_features_bq(features_table_name : str, project_id : str, dataset_id : str, table_names: dict) -> str:
#         """
#         Load tables from BQ, then build features dataset and dump it into BQ.
#         """ 
#         produits = load_data_bq(project_id, dataset_id, table_names["produits"])
#         transactions = load_data_bq(project_id, dataset_id, table_names["transactions"])
#         substitutions = load_data_bq(project_id, dataset_id, table_names["substitutions"])
        
#         df = build_features_dataset(produits, substitutions, transactions)
        
#         dump_table_into_bq(df, project_id, dataset_id, features_table_name)
        
#         return f'FEATURE ENG : {features_table_name} dumped into BQ'
    
#     # BigQuery table to be loaded
#     project_id = ingestion.project_id
#     dataset_id = ingestion.dataset_id
#     table_names = {"produits" : "produits", "transactions" : "transactions", "substitutions" : "substitutions"}
    
#     features_table_name = "features_dataset"
#     build_features_bq(features_table_name, project_id, dataset_id, table_names)
    
# # Instantiate the DAG
# features_dag()        