# # scripts/processing.py

# # 1) Définir la cible
# y = transactions_avec_substitution["estAcceptee_bin"]

# # 2) Définir les features
# features_num = [
#     "DiffPrix",
#     "MemeMarque",
#     "MemeNutriscore",
#     "MemeBio",
#     "prixOriginal",
#  #   "prixSubstitution",
#     "MemeConditionnement",
#     "MemeTypeMarque",
#     "estBioOriginal",
#     "Month",
# ]

# features_cat = [
#     "categorieOriginal",
#     "marqueOriginal",
#     "typeMarqueOriginal",
#     "nutriscoreOriginal",
#     "origineOriginal",
#     "conditionnementOriginal",
#     "categorieSubstitution",
#  #  "marqueSubstitution",
#     "typeMarqueSubstitution",
#  #  "nutriscoreSubstitution",
#     "origineSubstitution",
#  #  "estBioSubstitution",
# #  "conditionnementSubstitution",
#     "Day_of_week_name",
# ]

# X = transactions_avec_substitution[features_num + features_cat]

# # 3) Split temporel AVANT fit du préprocessing
# cutoff_idx = int(len(transactions_avec_substitution) * 0.8)
# X_train_raw, X_test_raw = X.iloc[:cutoff_idx], X.iloc[cutoff_idx:]
# y_train, y_test = y.iloc[:cutoff_idx], y.iloc[cutoff_idx:]
                                              
# # 4) Imputation des valeurs nulles et encodage des variables catégorielles
# numeric_transformer = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#     ]
# )
# categorical_transformer = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("onehot", OneHotEncoder(handle_unknown='ignore')),
#     ]
# )
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, features_num),
#         ("cat", categorical_transformer, features_cat),
#     ]
# )

# # Fit du preprocessing sur le train et transform le test
# X_train = preprocessor.fit_transform(X_train_raw)
# X_test = preprocessor.transform(X_test_raw)

