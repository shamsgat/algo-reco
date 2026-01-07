# Espace d'hyperparamètres pour Hyperopt
space = hp.choice("model_type", [
    {
        "type": "xgb",
        "eta": hp.uniform("eta", 0.01, 0.3),
        "max_depth": hp.choice("max_depth_x", [4, 6, 8, 10]),
        "subsample": hp.uniform("subsample_x", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_x", 0.5, 1.0),
        "min_child_weight": hp.uniform("min_child_x", 1, 10),
    },
    {
        "type": "lgb",
        "learning_rate": hp.uniform("learning_rate_l", 0.01, 0.3),
        "max_depth": hp.choice("max_depth_l", [-1, 4, 6, 8]),
        "num_leaves": hp.choice("num_leaves_l", [31, 63, 127]),
        "feature_fraction": hp.uniform("feature_fraction_l", 0.6, 1.0),
        "bagging_fraction": hp.uniform("bagging_fraction_l", 0.6, 1.0),
        "bagging_freq": hp.choice("bagging_freq_l", [0, 1, 5]),
    },
    {
        "type": "cat",
        "learning_rate": hp.uniform("learning_rate_c", 0.01, 0.3),
        "depth": hp.choice("depth_c", [4, 6, 8, 10]),
        "l2_leaf_reg": hp.uniform("l2_leaf_reg_c", 1, 10),
    }
])

# decode helpers (pour reconstruire les meilleurs modèles depuis trials)
_xgb_opts = {"max_depth_x": [4,6,8,10]}
_lgb_opts = {"max_depth_l": [-1,4,6,8], "num_leaves_l": [31,63,127], "bagging_freq_l": [0,1,5]}
_cat_opts = {"depth_c": [4,6,8,10]}

def scalarize(vals):
    return {k: (v[0] if isinstance(v, list) and len(v)==1 else v) for k,v in vals.items()}

def decode_xgb(vals):
    p = scalarize(vals)
    return {
        "eta": p.get("eta"),
        "max_depth": _xgb_opts["max_depth_x"][int(p.get("max_depth_x",0))] if "max_depth_x" in p else None,
        "subsample": p.get("subsample_x"),
        "colsample_bytree": p.get("colsample_x"),
        "min_child_weight": p.get("min_child_x"),
    }

def decode_lgb(vals):
    p = scalarize(vals)
    return {
        "learning_rate": p.get("learning_rate_l"),
        "max_depth": _lgb_opts["max_depth_l"][int(p.get("max_depth_l",0))] if "max_depth_l" in p else None,
        "num_leaves": _lgb_opts["num_leaves_l"][int(p.get("num_leaves_l",0))] if "num_leaves_l" in p else None,
        "feature_fraction": p.get("feature_fraction_l"),
        "bagging_fraction": p.get("bagging_fraction_l"),
        "bagging_freq": _lgb_opts["bagging_freq_l"][int(p.get("bagging_freq_l",0))] if "bagging_freq_l" in p else None,
    }

def decode_cat(vals):
    p = scalarize(vals)
    return {
        "learning_rate": p.get("learning_rate_c"),
        "depth": _cat_opts["depth_c"][int(p.get("depth_c",0))] if "depth_c" in p else None,
        "l2_leaf_reg": p.get("l2_leaf_reg_c"),
    }

# construction centralisée de modèles (utilise les decode_*)
def build_model_from_params(model_type, vals, random_state=42):
    if model_type == "xgb":
        params = decode_xgb(vals)
        return xgb.XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            tree_method="hist",
            use_label_encoder=False,
            eta=params.get("eta"),
            max_depth=int(params.get("max_depth")) if params.get("max_depth") is not None else 6,
            subsample=params.get("subsample"),
            colsample_bytree=params.get("colsample_bytree"),
            min_child_weight=params.get("min_child_weight"),
            random_state=random_state
        )
    if model_type == "lgb":
        params = decode_lgb(vals)
        return lgb.LGBMClassifier(
            n_estimators=100,
            verbose=-1,
            learning_rate=params.get("learning_rate"),
            max_depth=int(params.get("max_depth")) if params.get("max_depth") is not None else -1,
            num_leaves=int(params.get("num_leaves")) if params.get("num_leaves") is not None else 31,
            feature_fraction=params.get("feature_fraction"),
            bagging_fraction=params.get("bagging_fraction"),
            bagging_freq=int(params.get("bagging_freq")) if params.get("bagging_freq") is not None else 0,
            random_state=random_state
        )
    # catboost
    params = decode_cat(vals)
    return CatBoostClassifier(
        iterations=100,
        verbose=False,
        learning_rate=params.get("learning_rate"),
        depth=int(params.get("depth")) if params.get("depth") is not None else 6,
        l2_leaf_reg=params.get("l2_leaf_reg"),
        random_state=random_state
    )

# objective : utilise build_model_from_params
def objective(params):
    model_type = params["type"]
    params = dict(params)
    del params["type"]
    model = build_model_from_params(model_type, params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)
    return {"loss": -score, "status": STATUS_OK}

# Run Hyperopt
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials)

# extra maps
idx_to_type = {0: "xgb", 1: "lgb", 2: "cat"}
idx_to_label = {0: "XGBOOST", 1: "LIGHTGBM", 2: "CATBOOST"}
