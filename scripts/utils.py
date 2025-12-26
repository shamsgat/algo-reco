import logging
import pandas as pd
import json
from typing import Union
import gcsfs
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

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
        
def load_data_bq(table_id: str) -> pd.DataFrame:
    """
    Load a BigQuery table into a pandas DataFrame.

    Parameters
    ----------
    table_id : str
        Full table ID (project.dataset.table)

    Returns
    -------
    pd.DataFrame
    """
    logger.info("LOAD DATA BQ : Starting load from BigQuery table: %s", table_id)

    client = bigquery.Client()

    # Check table exists
    try:
        table = client.get_table(table_id)
        logger.info(
            "LOAD DATA BQ : Table found (%d rows, %d columns)",
            table.num_rows,
            len(table.schema),
        )
    except NotFound:
        logger.error("LOAD DATA BQ : Table does not exist: %s", table_id)
        raise FileNotFoundError(f"LOAD DATA BQ : Table does not exist: {table_id}")

    # Load data
    df = client.list_rows(table).to_dataframe()

    logger.info(
        "LOAD DATA BQ : Load completed successfully (%d rows)",
        len(df),
    )
    return df

# Test Load & Dump GCS
# produits_path_gcs = "gs://algo_reco/raw/produits/produits.csv"
# produits = load_data_gcs(produits_path_gcs)
# path = "gs://algo_reco/features/train"
# dump_data_gcs(produits[:3],path, "pro_test2")

# Test load data BigQuery
df = load_data_bq("smart-quasar-478510-r3.algo_reco_raw.produits_raw")
print(df.head())


