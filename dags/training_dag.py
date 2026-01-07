from airflow.sdk import dag, task
from pendulum import datetime
from scripts.train_model import run_hyperopt
from scripts.utils import convert_for_json, load_data_gcs, dump_data_gcs

@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    doc_md=__doc__,
    default_args={"owner": "ML", "retries": 3},
    tags=["ml", "processing"]
)

def training_dag():
    @task
    def train_model_task():
        X_train = load_data_gcs("gs://algo_reco/features/train/x_train_preprocessed.csv")
        y_train = load_data_gcs("gs://algo_reco/features/train/y_train.csv")
        X_test = load_data_gcs("gs://algo_reco/features/train/x_test_preprocessed.csv")
        y_test = load_data_gcs("gs://algo_reco/features/train/y_test.csv")

        best_params, trials, best_model = run_hyperopt(X_train, y_train, X_test, y_test)
        best_params_python = convert_for_json(best_params)

        trials_serializable = convert_for_json(trials.trials)

        dump_data_gcs(best_params_python, "gs://algo_reco/models", "best_params")
        dump_data_gcs(trials_serializable, "gs://algo_reco/models", "trials")
        dump_data_gcs(best_model, "gs://algo_reco/models", "best_model")
        
        return f'TRAINING : Best model and parameters dumped into GCS'
    
    train_model_task()

training_dag_instance = training_dag()