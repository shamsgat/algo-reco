import json
import logging
import pandas as pd
import gcsfs
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import sklearn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
import joblib

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def load_data_gcs(gcs_path: str) -> pd.DataFrame:
    """
    Load a CSV | JSON  or Joblib file from Google Cloud Storage into a pandas DataFrame.

    Parameters
    ----------
    gcs_path : str
        Path to the file in GCS (e.g., gs://my-bucket/path/file.csv)

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    logger.info("LOAD DATA : Starting GCS load: %s", gcs_path)

    fs = gcsfs.GCSFileSystem()

    # Check if the file exists
    if not fs.exists(gcs_path):
        logger.error("LOAD DATA : GCS path does not exist: %s", gcs_path)
        raise FileNotFoundError(f"LOAD DATA : GCS path does not exist: {gcs_path}")
    # CSV
    if gcs_path.endswith(".csv"):
        logger.info("LOAD DATA : Detected CSV file")
        data = pd.read_csv(gcs_path)
        logger.info("LOAD DATA : Load CSV completed successfully (%d rows)", len(data))
    # JSON
    elif gcs_path.endswith(".json"):
        logger.info("LOAD DATA : Detected JSON file")
        with fs.open(gcs_path, "r") as f:
            content = json.load(f)

        # Dict → config
        if isinstance(content, dict):
            logger.info("LOAD DATA : JSON parsed as dict (config file)")
            data = content
            logger.info("LOAD DATA : Load JSON completed successfully (dict with %d keys)", len(data))

        # List → dataset
        elif isinstance(content, list):
            logger.info("JSON parsed as list (dataset)")
            data = pd.DataFrame(content)
            logger.info("LOAD DATA : Load JSON completed successfully (%d rows)", len(data))

        else:
            logger.error("LOAD DATA : Unsupported JSON structure in %s", gcs_path)
            raise ValueError("LOAD DATA : Unsupported JSON structure")
    
    # Joblib (for sklearn objects)
    elif gcs_path.endswith(".joblib"):
        logger.info("LOAD DATA : Detected Joblib file")
        with fs.open(gcs_path, "rb") as f:
            data = joblib.load(f)
        logger.info("LOAD DATA : Load Joblib completed successfully: %s", type(data).__name__)
    
    else:
        logger.error("LOAD DATA : Unsupported file format for path: %s", gcs_path)
        raise ValueError("LOAD DATA : Unsupported file format. Only CSV, JSON & Joblib are supported.")

    return data


def dump_data_gcs(data: pd.DataFrame | dict | list | BaseEstimator, path: str, filename: str) -> None:
    """
    Dump data to GCS.
    - DataFrame -> CSV
    - dict/list -> JSON TXT file

    Parameters
    ----------
    data : pd.DataFrame | dict | list | sklearn object
    path : str
        GCS directory path (e.g., gs://bucket/folder/subfolder)
    filename : str
        File name without extension
    """
    logger.info("DUMP DATA : Starting dump to GCS: %s/%s", path, filename)

    fs = gcsfs.GCSFileSystem()

    if not fs.exists(path):
        logger.info("DUMP DATA : Path does not exist. Creating path: %s", path)
        fs.mkdirs(path)
    else:
        logger.info("DUMP DATA : Path already exists: %s", path)

    # Dump DataFrame as CSV
    if isinstance(data, pd.DataFrame):
        output_path = f"{path}/{filename}.csv"
        logger.info("DUMP DATA : Detected DataFrame with %d rows. Writing CSV to %s", len(data), output_path)
        data.to_csv(output_path, index=False)
    
    # Dump dict or list as JSON
    elif isinstance(data, (dict, list)):
        output_path = f"{path}/{filename}.json"
        logger.info("DUMP DATA : Detected JSON-like data (%s). Writing JSON to %s", type(data).__name__, output_path)
        with fs.open(output_path, "w") as f:
            import json
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    # Sklearn objects (ColumnTransformer, Pipeline, Model)
    elif isinstance(data, (BaseEstimator, ColumnTransformer)):
        output_path = f"{path}/{filename}.joblib"
        logger.info("DUMP DATA : Sklearn object detected (%s). Writing joblib to %s",
            type(data).__name__, output_path
        )
        with fs.open(output_path, "wb") as f:
            joblib.dump(data, f)
    else:
        logger.error("DUMP DATA : Unsupported data type: %s", type(data))
        raise TypeError("DUMP DATA : Unsupported data type. Expected pandas DataFrame, dict, list or sklearn object.")

    logger.info("DUMP DATA : Dump to GCS completed successfully")


def load_data_bq(project_id: str, dataset_id: str, table_name: str) -> pd.DataFrame:
    """
    Load a BigQuery table into a pandas DataFrame.

    Parameters
    ----------
    project_id : str
        GCP project ID
    dataset_id : str
        BigQuery dataset ID
    table_name : str
        Table name in the dataset

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    table_ref = f"{project_id}.{dataset_id}.{table_name}"
    logger.info("LOAD DATA : Starting BigQuery load: %s", table_ref)

    client = bigquery.Client(project=project_id)

    try:
        table = client.get_table(table_ref)
        logger.info("LOAD DATA : Table found (%d rows, %d columns)", table.num_rows, len(table.schema))
    except NotFound:
        logger.error("LOAD DATA : Table does not exist: %s", table_ref)
        raise FileNotFoundError(f"LOAD DATA : BigQuery table does not exist: {table_ref}")

    df = client.list_rows(table).to_dataframe()
    logger.info("LOAD DATA : BigQuery load completed successfully (%d rows)", len(df))
    return df


def dump_table_into_bq(df: pd.DataFrame, project_id: str, dataset_id: str, table_name: str) -> None:
    """
    Dump a pandas DataFrame into BigQuery.
    Creates the dataset if it does not exist.

    Parameters
    ----------
    df : pd.DataFrame
        Data to load into BigQuery
    project_id : str
        GCP project ID
    dataset_id : str
        BigQuery dataset ID
    table_name : str
        Table name in the dataset
    """
    table_ref = f"{project_id}.{dataset_id}.{table_name}"
    dataset_ref = f"{project_id}.{dataset_id}"

    logger.info("DUMP DATA : Starting dump to BigQuery: %s", table_ref)

    client = bigquery.Client(project=project_id)

    # Check or create dataset
    try:
        client.get_dataset(dataset_ref)
        logger.info("DUMP DATA : Dataset %s already exists", dataset_ref)
    except NotFound:
        logger.warning("DUMP DATA : Dataset %s not found. Creating it...", dataset_ref)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"  # Change to "US" if needed
        client.create_dataset(dataset)
        logger.info("DUMP DATA : Dataset %s created", dataset_ref)

    # Load the table
    logger.info("DUMP DATA : Droping table if exists and loading data into BigQuery table %s (%d rows)", table_ref, len(df))
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    load_job = client.load_table_from_dataframe(
        df,
        table_ref,
        job_config=job_config,
    )
    load_job.result()
    logger.info("DUMP DATA : Dump to BigQuery completed successfully for table %s", table_ref)
