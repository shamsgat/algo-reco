# scripts/ingestion.py

from airflow.utils.log.logging_mixin import LoggingMixin
from scripts.bootstrap import init_gcp_credentials
from scripts.utils import (
    load_data_gcs,
    dump_data_gcs,
    load_data_bq,
    dump_table_into_bq,
)

logger = LoggingMixin().log

class Ingestion:
    def __init__(self):
        # Initialize GCP credentials
        self.project_id, self.dataset_id = init_gcp_credentials()

        # Log the retrieved values
        logger.info("Ingestion initialized with PROJECT_ID=%s and DATASET_ID=%s", 
                    self.project_id, self.dataset_id)

        if not self.project_id or not self.dataset_id:
            raise EnvironmentError("PROJECT_ID or DATASET_ID not set")

    def gcs_to_bq(self, gcs_path: str, table_name: str):
        logger.info("Starting gcs_to_bq for table: %s", table_name)
        df = load_data_gcs(gcs_path)
        dump_table_into_bq(
            df,
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            table_name=table_name,
        )
        logger.info("Completed gcs_to_bq for table: %s", table_name)

    def bq_to_gcs(self, table_name: str, gcs_path: str, filename: str):
        logger.info("Starting bq_to_gcs for table: %s", table_name)
        df = load_data_bq(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            table_name=table_name,
        )
        dump_data_gcs(df, gcs_path, filename=filename)
        logger.info("Completed bq_to_gcs for table: %s", table_name)
