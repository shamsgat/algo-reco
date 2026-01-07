"""
ML inference DAG.

This DAG loads the preprocessed features (Z_transformed), the trained model, and the best parameters from GCS,
runs predictions, and stores the dataset with prediction columns back to GCS.
"""

from airflow.sdk import dag, task
from pendulum import datetime
from scripts.utils import load_data_gcs, dump_data_gcs
from scripts.inference import run_inference
import logging
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "inference"]
)
def inference_dag():

    @task
    def run_model_inference(
        model_gcs_path: str = "gs://algo_reco/models/best_model.joblib",
        best_params_gcs_path: str = "gs://algo_reco/models/best_params.json",
        z_transformed_gcs_path: str = "gs://algo_reco/features/train/x_test_preprocessed.csv",
        output_gcs_path: str = "gs://algo_reco/inference",
        output_filename: str = "x_test_predicted",
        threshold: float = 0.5
    ):
        # ----------------------------
        # Load artifacts
        # ----------------------------
        logger.info("INFERENCE DAG : Loading model from %s", model_gcs_path)
        model = load_data_gcs(model_gcs_path)  # loads joblib

        logger.info("INFERENCE DAG : Loading best_params from %s", best_params_gcs_path)
        best_params = load_data_gcs(best_params_gcs_path)  # loads JSON dict

        logger.info("INFERENCE DAG : Loading Z_transformed from %s", z_transformed_gcs_path)
        Z_transformed = load_data_gcs(z_transformed_gcs_path)  # loads CSV

        # ----------------------------
        # Run inference
        # ----------------------------
        Z_pred = run_inference(
            model=model,
            best_params=best_params,
            Z_transformed=Z_transformed,
            threshold=threshold,
            add_proba=True
        )

        # ----------------------------
        # Dump results
        # ----------------------------
        dump_data_gcs(
            data=Z_pred,
            path=output_gcs_path,
            filename=output_filename
        )

        return f"INFERENCE DAG : Predictions completed ({len(Z_pred)} rows)"

    run_model_inference()

# Instantiate the DAG
inference_dag()
