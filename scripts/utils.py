import logging
import pandas as pd
import gcsfs
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def load_data_gcs(gcs_path: str) -> pd.DataFrame:
    """
    Load a CSV or JSON file from Google Cloud Storage into a pandas DataFrame.

    Parameters
    ----------
    gcs_path : str
        Path to the file in GCS (e.g., gs://my-bucket/path/file.csv)

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    logger.info("Starting GCS load: %s", gcs_path)

    fs = gcsfs.GCSFileSystem()

    if not fs.exists(gcs_path):
        logger.error("GCS path does not exist: %s", gcs_path)
        raise FileNotFoundError(f"GCS path does not exist: {gcs_path}")

    if gcs_path.endswith(".csv"):
        logger.info("Detected CSV file")
        data = pd.read_csv(gcs_path)
    elif gcs_path.endswith(".json"):
        logger.info("Detected JSON file")
        data = pd.read_json(gcs_path, lines=True)
    else:
        logger.error("Unsupported file format for path: %s", gcs_path)
        raise ValueError("Unsupported file format. Only CSV and JSON are supported.")

    logger.info("Load completed successfully (%d rows)", len(data))
    return data


def dump_data_gcs(data: pd.DataFrame | dict | list, path: str, filename: str) -> None:
    """
    Dump data to GCS.
    - DataFrame -> CSV
    - dict/list -> JSON TXT file

    Parameters
    ----------
    data : pd.DataFrame | dict | list
        Data to dump
    path : str
        GCS directory path (e.g., gs://bucket/folder/subfolder)
    filename : str
        File name without extension
    """
    logger.info("Starting dump to GCS: %s/%s", path, filename)

    fs = gcsfs.GCSFileSystem()

    if not fs.exists(path):
        logger.info("Path does not exist. Creating path: %s", path)
        fs.mkdirs(path)
    else:
        logger.info("Path already exists: %s", path)

    if isinstance(data, pd.DataFrame):
        output_path = f"{path}/{filename}.csv"
        logger.info("Detected DataFrame with %d rows. Writing CSV to %s", len(data), output_path)
        data.to_csv(output_path, index=False)
    elif isinstance(data, (dict, list)):
        output_path = f"{path}/{filename}.txt"
        logger.info("Detected JSON-like data (%s). Writing TXT to %s", type(data).__name__, output_path)
        with fs.open(output_path, "w") as f:
            import json
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        logger.error("Unsupported data type: %s", type(data))
        raise TypeError("Unsupported data type. Expected pandas DataFrame, dict, or list.")

    logger.info("Dump to GCS completed successfully")


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
    logger.info("Starting BigQuery load: %s", table_ref)

    client = bigquery.Client(project=project_id)

    try:
        table = client.get_table(table_ref)
        logger.info("Table found (%d rows, %d columns)", table.num_rows, len(table.schema))
    except NotFound:
        logger.error("Table does not exist: %s", table_ref)
        raise FileNotFoundError(f"BigQuery table does not exist: {table_ref}")

    df = client.list_rows(table).to_dataframe()
    logger.info("BigQuery load completed successfully (%d rows)", len(df))
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

    logger.info("Starting dump to BigQuery: %s", table_ref)

    client = bigquery.Client(project=project_id)

    # Check or create dataset
    try:
        client.get_dataset(dataset_ref)
        logger.info("Dataset %s already exists", dataset_ref)
    except NotFound:
        logger.warning("Dataset %s not found. Creating it...", dataset_ref)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"  # Change to "US" if needed
        client.create_dataset(dataset)
        logger.info("Dataset %s created", dataset_ref)

    # Load the table
    logger.info("Loading data into BigQuery table %s (%d rows)", table_ref, len(df))
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()
    logger.info("Dump to BigQuery completed successfully for table %s", table_ref)
