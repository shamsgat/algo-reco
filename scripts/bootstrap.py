#scripts/bootstrap.py

from pathlib import Path
import os
from dotenv import load_dotenv

def init_gcp_credentials():
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

    # Retourne le project_id si défini dans .env, sinon None
    return os.environ.get("GOOGLE_CLOUD_PROJECT")
