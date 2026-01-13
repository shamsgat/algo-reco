# dags/pipeline_inference_dag.py
"""
ML full pipeline orchestration DAG.

Order:
1. ingestion_dag
2. features_dag
3. processing_dag
4. inference_dag
"""

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from pendulum import datetime

# Variables à passer au DAG ingestion
INGESTION_MODE = "inference" 
INGESTION_DATES = ["2023-12-20", "2023-12-21"]  # liste de dates pour l'inference

with DAG(
    dag_id="inference_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "ML"},
    tags=["ml", "inference_pipeline"],
) as dag:

    trigger_ingestion = TriggerDagRunOperator(
        task_id="trigger_ingestion",
        trigger_dag_id="ingestion_dag",
        conf={
            "INGESTION_MODE": INGESTION_MODE,
            "INGESTION_DATES": INGESTION_DATES
        },
        wait_for_completion=True
    )

    trigger_features = TriggerDagRunOperator(
        task_id="trigger_features",
        trigger_dag_id="features_dag",
        conf={
            "INGESTION_MODE": INGESTION_MODE,
            "INGESTION_DATES": INGESTION_DATES
        },
        wait_for_completion=True
    )

    trigger_processing = TriggerDagRunOperator(
        task_id="trigger_processing",
        trigger_dag_id="processing_dag",
        conf={
            "INGESTION_MODE": INGESTION_MODE,
            "INGESTION_DATES": INGESTION_DATES
        },
        wait_for_completion=True
    )

    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_inference",
        trigger_dag_id="inference_dag",
        wait_for_completion=True
    )

    # Orchestration order
    trigger_ingestion >> trigger_features >> trigger_processing >> trigger_inference
