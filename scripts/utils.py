import logging
import pandas as pd
import json
from scripts.bootstrap import init_gcp_credentials
from typing import Union
import gcsfs
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

# Initialize GCP credentials and get project ID
PROJECT_ID = init_gcp_credentials()[0]
DATASET_ID = init_gcp_credentials()[1]

def load_data_gcs(gcs_path: str) -> pd.DataFrame:
    """
    Load a CSV or JSON file from Google Cloud Storage into a pandas DataFrame.
    Parameters: gcs_path : str, Path to the file in GCS (e.g. gs://my-bucket/path/file.csv)
    Returns:pd.DataFrame
    """
    logger.info("LOAD DATA : Starting load from GCS: %s", gcs_path)

    fs = gcsfs.GCSFileSystem()

    # Check path exists
    if not fs.exists(gcs_path):
        logger.error("LOAD DATA : GCS path does not exist: %s", gcs_path)
        raise FileNotFoundError(f"LOAD DATA : GCS path does not exist: {gcs_path}")

    if gcs_path.endswith(".csv"):
        logger.info("LOAD DATA : Detected CSV file")
        data = pd.read_csv(gcs_path)

    elif gcs_path.endswith(".json"):
        logger.info("LOAD DATA : Detected JSON file")
        data = pd.read_json(gcs_path, lines=True)

    else:
        logger.error("LOAD DATA : Unsupported file format for path: %s", gcs_path)
        raise ValueError("LOAD DATA : Unsupported file format. Only CSV and JSON are supported.")

    logger.info("LOAD DATA : Load completed successfully (%d rows)", len(data))
    return data


def dump_data_gcs(data: Union[pd.DataFrame, dict, list], path: str, filename: str) -> None:
    """
    Dump data to GCS.
    - DataFrame -> CSV
    - dict/list -> TXT (JSON format)

    Parameters
    ----------
    data : pd.DataFrame | dict | list
        Data to dump
    path : str
        GCS directory path (e.g. gs://bucket/folder/subfolder)
    filename : str
        File name without extension
    """
    logger.info("DUMP DATA : Starting dump to GCS")

    fs = gcsfs.GCSFileSystem()

    logger.info("DUMP DATA : Checking if path exists: %s", path)
    if not fs.exists(path):
        logger.info("DUMP DATA : Path does not exist. Creating path: %s", path)
        fs.mkdirs(path)
    else:
        logger.info("DUMP DATA : Path already exists: %s", path)

    if isinstance(data, pd.DataFrame):
        output_path = f"{path}/{filename}.csv"
        logger.info("DUMP DATA : Detected DataFrame with %d rows", len(data))
        logger.info("DUMP DATA : Writing CSV to %s", output_path)
        data.to_csv(output_path, index=False)

    elif isinstance(data, (dict, list)):
        output_path = f"{path}/{filename}.txt"
        logger.info("DUMP DATA : Detected JSON-like data (%s)", type(data).__name__)
        logger.info("DUMP DATA : Writing TXT to %s", output_path)
        with fs.open(output_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    else:
        logger.error("DUMP DATA : Unsupported data type: %s", type(data))
        raise TypeError(
            "DUMP DATA : Unsupported data type. Expected pandas DataFrame, dict, or list."
        )

    logger.info("DUMP DATA : Dump to GCS completed successfully")

def load_data_bq(project_id: str, dataset_id: str, table_name: str) -> pd.DataFrame:

    logger.info("LOAD DATA : Starting load from BigQuery table: %s", f"{project_id}.{dataset_id}.{table_name}")

    client = bigquery.Client(project=project_id)

    # Check table exists
    try:
        table = client.get_table(f"{project_id}.{dataset_id}.{table_name}")
        logger.info(
            "LOAD DATA : Table BQ found (%d rows, %d columns)",
            table.num_rows,
            len(table.schema),
        )
    except NotFound:
        logger.error("LOAD DATA : Table BQ does not exist: %s", f"{project_id}.{dataset_id}.{table_name}")
        raise FileNotFoundError(f"LOAD DATA BQ : Table does not exist: {project_id}.{dataset_id}.{table_name}")

    # Load data
    df = client.list_rows(table).to_dataframe()

    logger.info(
        "LOAD DATA : Load BQ completed successfully (%d rows)",
        len(df),
    )
    return df

def dump_table_into_bq(df: pd.DataFrame, project_id: str, dataset_id: str, table_name: str) -> None:
    """
    Dump un DataFrame dans BigQuery.
    Crée le dataset s'il n'existe pas.
    """

    logger.info("DUMP DATA : Starting dump from BigQuery table: %s", f"{project_id}.{dataset_id}.{table_name}")


    client = bigquery.Client(project=project_id)

    dataset_ref = f"{project_id}.{dataset_id}"
    table_ref = f"{dataset_ref}.{table_name}"

    # 1️⃣ Vérifier / créer le dataset (mkdir logique)
    try:
        client.get_dataset(dataset_ref)
        logger.info("DUMP DATA : Dataset %s in BQ already exists", dataset_ref)
    except NotFound:
        logger.warning("DUMP DATA : Dataset %s in BQ not found. Creating it...", dataset_ref)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"  # ou "US" selon ton projet
        client.create_dataset(dataset)
        logger.info("DUMP DATA : Dataset %s in BQ created", dataset_ref)

    # 2️⃣ Charger la table
    logger.info(
        "DUMP DATA : Starting load into BigQuery table %s (%d rows)",
        table_ref,
        len(df),
    )

    job = client.load_table_from_dataframe(df, table_ref)
    job.result()

    logger.info("DUMP DATA : Dump BQ completed successfully for table %s", table_ref)


