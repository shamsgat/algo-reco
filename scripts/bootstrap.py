#scripts/bootstrap.py

from pathlib import Path
import os
from typing import Optional, Tuple
from dotenv import load_dotenv


def init_gcp_credentials() -> Tuple[Optional[str], Optional[str]]:
    # Charge le .env
    load_dotenv()

    # Racine du projet
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Résout le chemin absolu des credentials depuis la racine
    cred_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    cred_absolute_path = (PROJECT_ROOT / cred_path).resolve()

    if not cred_absolute_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {cred_absolute_path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_absolute_path)

    project_id = os.environ.get("PROJECT_ID")
    dataset_id = os.environ.get("DATASET_ID")

    # Retourne le project_id si défini dans .env, sinon None
    return project_id, dataset_id

