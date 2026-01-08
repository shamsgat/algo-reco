# scripts/inference.py
from airflow.utils.log.logging_mixin import LoggingMixin
import numpy as np
import pandas as pd

logger = LoggingMixin().log

def run_inference(model,best_params: dict,Z_transformed: pd.DataFrame,threshold: float = 0.5,add_proba: bool = True) -> pd.DataFrame:
    """
    Run inference on already-loaded artifacts.

    Parameters
    ----------
    model : trained model object
        Loaded in the DAG (joblib.load).
    best_params : dict
        Loaded in the DAG (best_params.json).
    Z_transformed : pd.DataFrame
        Transformed features.
    threshold : float
        Classification threshold.
    add_proba : bool
        Whether to add prediction probability column.

    Returns
    -------
    pd.DataFrame
        Z_transformed enriched with predictions.
    """

    logger.info("INFERENCE : Starting inference")

    # ----------------------------
    # Safety checks
    # ----------------------------
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "INFERENCE : Model does not expose predict_proba()"
        )

    if not isinstance(Z_transformed, pd.DataFrame):
        raise TypeError(
            "INFERENCE : Z_transformed must be a pandas DataFrame"
        )

    model_type = best_params.get("type", "unknown")
    logger.info("INFERENCE : Model type = %s", model_type)
    logger.info("INFERENCE : Z_transformed shape = %s", Z_transformed.shape)
    logger.info("INFERENCE : Using threshold = %.3f", threshold)

    # ----------------------------
    # Prediction
    # ----------------------------
    logger.info("INFERENCE  : Computing prediction probabilities")
    proba = model.predict_proba(Z_transformed)[:, 1]

    if add_proba:
        Z_transformed["prediction_proba"] = proba

    Z_transformed["prediction"] = (proba >= threshold).astype(int)

    logger.info("INFERENCE : Inference completed")

    return Z_transformed
