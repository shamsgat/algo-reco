import pandas as pd

def get_X_y(df, target_column, feature_columns=None):
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    return X, y
