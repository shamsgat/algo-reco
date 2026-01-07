# dags/processing_dag.py
"""
ML data processing data DAG.

XXXX ????
"""
import logging
from airflow.sdk import dag, task
from pendulum import datetime
from scripts.ingestion import Ingestion
from scripts.utils import load_data_bq, dump_table_into_bq, dump_data_gcs,load_data_gcs
from scripts.features import *
from scripts.processing import *

logger = logging.getLogger(__name__)

@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    doc_md=__doc__,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "processing"]
)

def processing_dag():
    ingestion = Ingestion()
    @task
    def train_test_split(features_table_name : str, project_id : str, dataset_id : str) :

        features_num, features_cat = get_feature_lists()
        features_dict = {"numerical" : features_num, "categorical" : features_cat}
        dump_data_gcs(features_dict,"gs://algo_reco/features/train","features")
    
        features_dataset = load_data_bq(project_id, dataset_id, features_table_name)
        target = define_target(features_dataset)
        features = select_features(features_dataset, features_num, features_cat)
        X_train, X_test, y_train, y_test = temporal_train_test_split(features, target)
        dump_data_gcs(X_train,"gs://algo_reco/features/train","x_train")
        dump_data_gcs(X_test, "gs://algo_reco/features/train","x_test")
        dump_data_gcs(pd.DataFrame(y_train), "gs://algo_reco/features/train","y_train")
        dump_data_gcs(pd.DataFrame(y_test), "gs://algo_reco/features/train","y_test")
        return f'PROCESSING : Train/Test split dumped into GCS'
    
    @task
    def preprocessing():
        features_dict = load_data_gcs("gs://algo_reco/features/train/features.json")
        logger.info(f'PROCESSING : features_dict = {features_dict}')
        X_train = load_data_gcs("gs://algo_reco/features/train/x_train.csv")
        X_test = load_data_gcs("gs://algo_reco/features/train/x_test.csv")
        preprocessor = build_preprocessor(features_dict["numerical"],features_dict["categorical"])
        X_train_transformed, X_test_transformed, preprocessor = fit_transform_preprocessor(preprocessor,X_train,X_test)
        dump_data_gcs(X_train_transformed,"gs://algo_reco/features/train","x_train_preprocessed")
        dump_data_gcs(X_test_transformed,"gs://algo_reco/features/train","x_test_preprocessed")
        dump_data_gcs(preprocessor,"gs://algo_reco/features/train","preprocessor")       
        return f'PROCESSING : Preprocessed data (X_train_transformed, X_test_transformed, preprocessor) dumped into GCS'
    
    # BigQuery features table to be loaded
    project_id = ingestion.project_id
    dataset_id = ingestion.dataset_id
    features_table_name = "features_dataset"
    # Task instantiation
    split_task = train_test_split(features_table_name, project_id, dataset_id)
    preprocess_task = preprocessing()

    # Dependency
    split_task >> preprocess_task
    
# Instantiate the DAG
processing_dag()