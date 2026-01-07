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

def compute_ranking_metrics(y_true, y_score, group, ks=(1,3,5)):
    metrics = {}
    idx = 0
    per_k = {k: {"ndcg": [], "hit": []} for k in ks}
    for g in group:
        y_g = np.asarray(y_true[idx: idx + g])
        s_g = np.asarray(y_score[idx: idx + g])
        idx += g
        order = np.argsort(-s_g)
        rels_sorted = y_g[order]
        for k in ks:
            rels_k = rels_sorted[:k]
            per_k[k]["ndcg"].append(np.sum(rels_k / np.log2(np.arange(2, len(rels_k)+2))))
            per_k[k]["hit"].append(float(np.any(rels_k > 0)))
    for k in ks:
        metrics[f"ndcg_at_{k}"] = float(np.mean(per_k[k]["ndcg"]))
        metrics[f"hit_rate_at_{k}"] = float(np.mean(per_k[k]["hit"]))

    return metrics

def objective_ranker(params, X_train, y_train, X_val, y_val, group_train, group_val):
    params = {k:int(v) if k in ["num_leaves","n_estimators","min_child_samples"] else v for k,v in params.items()}
    with mlflow.start_run(nested=True):
        model = LGBMRanker(**params)
        model.fit(
            X_train, y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            eval_metric="ndcg",
            callbacks=[early_stopping(stopping_rounds=50, verbose=False),
                       log_evaluation(period=0)]
        )
        best_iter = model.best_iteration_
        mlflow.log_metric("best_iteration", best_iter)
        scores = model.predict(X_val, num_iteration=best_iter)
        metrics = compute_ranking_metrics(y_val, scores, group_val)
        mlflow.log_params(params)
        safe_log_metrics(metrics)
        mlflow.lightgbm.log_model(model, "model")
        return {"loss": -metrics["ndcg_at_3"], "status": STATUS_OK}


def compute_classification_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "logloss": float(log_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def objective_classifier(params, model_class, X_train, y_train, X_val, y_val):
    params = {k:int(v) if isinstance(v,float) and v.is_integer() else v for k,v in params.items()}
    with mlflow.start_run(nested=True):
        model = model_class(**params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:,1]
        metrics = compute_classification_metrics(y_val, y_proba)
        mlflow.log_params(params)
        safe_log_metrics(metrics)
        if isinstance(model,LGBMClassifier):
            mlflow.lightgbm.log_model(model,"model")
        return {"loss": -metrics["auc"], "status": STATUS_OK}

def safe_log_metrics(metrics_dict):
    safe_metrics = {}
    for k, v in metrics_dict.items():
        try:
            safe_metrics[k] = float(v)
        except (TypeError, ValueError):
            safe_metrics[k] = np.nan
    mlflow.log_metrics(safe_metrics)



features_num = ["DiffPrix", "MemeMarque", "MemeNutriscore", "MemeBio",
                "prixOriginal", "MemeConditionnement", "MemeTypeMarque", "estBioOriginal", "Month"]
features_cat = ["categorieOriginal", "marqueOriginal", "typeMarqueOriginal", "nutriscoreOriginal",
                "origineOriginal", "conditionnementOriginal", "categorieSubstitution",
                "typeMarqueSubstitution", "origineSubstitution", "Day_of_week_name"]

X = df[features_num + features_cat]
y = df["estAcceptee_bin"]

# Split temporel
cutoff_idx = int(len(df) * 0.8)
X_train_raw, X_val_raw = X.iloc[:cutoff_idx], X.iloc[cutoff_idx:]
y_train, y_val = y.iloc[:cutoff_idx], y.iloc[cutoff_idx:]

# Pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, features_num),
    ("cat", categorical_transformer, features_cat)
])

X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)

#TODO recupérer df du script features.py
group_train = df.iloc[:cutoff_idx].groupby('idTransaction').size().to_numpy()
group_val = df.iloc[cutoff_idx:].groupby('idTransaction').size().to_numpy()

space_lgbm_ranker = {
    "num_leaves": hp.quniform("num_leaves", 31, 127, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
    "n_estimators": hp.quniform("n_estimators", 300, 1200, 50),
    "min_child_samples": hp.quniform("min_child_samples", 20, 100, 5),
    "subsample": hp.uniform("subsample", 0.7, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 1.0),
}

models = {
    "LogReg": {"model_class": LogisticRegression, "param_grid": {"C": [0.1,1.0,10.0], "penalty":["l2"]}, "fixed_params": {"solver":"lbfgs","max_iter":2000,"n_jobs":-1,"random_state":42}, "type":"classification"},
    "XGBClassifier": {"model_class": XGBClassifier, "param_grid": {"n_estimators":[500,1000],"max_depth":[4,6,8],"learning_rate":[0.03,0.05,0.1],"subsample":[0.7,0.9,1.0],"colsample_bytree":[0.7,0.9,1.0],"min_child_weight":[1,5,10],"reg_alpha":[0.0,0.1,1.0],"reg_lambda":[1.0,2.0,5.0]}, "fixed_params":{"objective":"binary:logistic","eval_metric":"auc","tree_method":"hist","random_state":42,"n_jobs":-1}, "type":"classification"},
    "LGBMClassifier": {"model_class": LGBMClassifier, "param_grid":{"num_leaves":[31,63,127],"learning_rate":[0.03,0.05,0.1],"n_estimators":[500,1000],"min_child_samples":[20,50,100],"subsample":[0.7,0.9,1.0],"colsample_bytree":[0.7,0.9,1.0],"reg_alpha":[0.0,0.1,1.0],"reg_lambda":[0.0,0.1,1.0]}, "fixed_params":{"objective":"binary","metric":"auc","random_state":42,"n_jobs":-1}, "type":"classification"},
    "CatBoostClassifier": {"model_class": CatBoostClassifier, "param_grid":{"depth":[6,8,10],"learning_rate":[0.03,0.05,0.1],"iterations":[500,1000],"l2_leaf_reg":[1,3,5,9],"subsample":[0.7,0.9,1.0],"rsm":[0.7,0.9,1.0]}, "fixed_params":{"loss_function":"Logloss","eval_metric":"AUC","random_seed":42,"verbose":0}, "type":"classification"},
    "LGBMRanker": {"model_class": LGBMRanker, "param_grid":{"num_leaves":[31,63,127],"learning_rate":[0.03,0.05,0.1],"n_estimators":[500,1000],"min_child_samples":[20,50,100],"subsample":[0.7,0.9,1.0],"colsample_bytree":[0.7,0.9,1.0]}, "fixed_params":{"objective":"lambdarank","metric":"ndcg","random_state":42,"n_jobs":-1}, "type":"ranking"}
}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # ou localhost si accessible
try:
    mlflow.create_experiment(EXP_NAME)
except mlflow.exceptions.MlflowException:
    pass
mlflow.set_experiment(EXP_NAME)


for name, cfg in models.items():
    print(f"=== Optimisation {name} ===")
    
    # Création de l'espace Hyperopt
    space = {k: hp.choice(k, v) if isinstance(v, list) else v for k, v in cfg["param_grid"].items()}
    trials = Trials()
    
    if cfg["type"] == "classification":
        fmin(
            fn=lambda p: objective_classifier(
                p,
                cfg["model_class"],
                X_train,
                y_train,
                X_val,
                y_val
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
            rstate=np.random.default_rng(42)
        )
    else:
        fmin(
            fn=lambda p: objective_ranker(
                p,
                X_train,
                y_train,
                X_val,
                y_val,
                group_train,
                group_val
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
            rstate=np.random.default_rng(42)
        )

