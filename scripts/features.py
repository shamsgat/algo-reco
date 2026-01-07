import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    precision_score, recall_score, f1_score
)

import mlflow
import mlflow.lightgbm
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from lightgbm import LGBMClassifier, LGBMRanker, early_stopping, log_evaluation

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
MLFLOW_TRACKING_URI = "http://localhost:5555"
EXP_NAME = "stockout_substitution_hyperopt_classifier_ranker_5"

def find_project_root(marker=".git"):
    path = Path().resolve()
    while path != path.parent:
        if (path / marker).exists():
            return path
        path = path.parent
    raise FileNotFoundError(f"Project root with {marker} not found")

ROOT_DIR = find_project_root()

def merge_and_add_suffix(df_add_suffix, df_keep, suffix, column_to_merge):
    df_add = df_add_suffix.copy()
    df_add = df_add.rename(columns={c: c + suffix for c in df_add.columns if c != column_to_merge})
    return pd.merge(df_keep, df_add, left_on=column_to_merge, right_on=column_to_merge, how='left')

def build_dataset(transactions, substitutions, produits):
    subs_prod_orig = merge_and_add_suffix(produits, substitutions, 'Original', 'idProduitOriginal')
    subs_prod_orig_subst = merge_and_add_suffix(produits, subs_prod_orig, 'Substitution', 'idProduitSubstitution')
    df = pd.merge(transactions, subs_prod_orig_subst,
                  left_on=['idProduit','idTransaction'],
                  right_on=['idProduitOriginal','idTransaction'], how='inner')
    df['estAcceptee_bin'] = (~df['estAcceptee']).astype(int)
    df['date'] = pd.to_datetime(df['dateHeureTransaction'])
    df['Month'] = df['date'].dt.month
    df['Day_of_week_name'] = df['date'].dt.day_name()
    df["MemeMarque"] = (df["marqueOriginal"] == df["marqueSubstitution"]).astype(int)
    df["MemeNutriscore"] = (df["nutriscoreOriginal"] == df["nutriscoreSubstitution"]).astype(int)
    df["MemeConditionnement"] = (df["conditionnementOriginal"] == df["conditionnementSubstitution"]).astype(int)
    df["MemeTypeMarque"] = (df["typeMarqueOriginal"] == df["typeMarqueSubstitution"]).astype(int)
    df["DiffPrix"] = df["prixSubstitution"] - df["prixOriginal"]
    df["MemeBio"] = ((df["estBioOriginal"] == True) & (df["estBioSubstitution"] == True)).astype(int)
    return df


#TODO connecter directement via utils de GCP
substitutions = pd.read_csv(DATA_RAW_DIR / "substitutions" / "raw_substitutions_substitutions.csv")
produits = pd.read_csv(DATA_RAW_DIR / "produits" / "raw_produits_produits.csv")
transactions = pd.read_csv(DATA_RAW_DIR / "transactions" / "raw_transactions_transactions.csv")

df = build_dataset(transactions, substitutions, produits)



