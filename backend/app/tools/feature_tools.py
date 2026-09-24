# Feature Engineering Tools - Conservative feature engineering
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Conservative feature engineering transformer."""
    
    def __init__(
        self,
        add_datetime_features: bool = True,
        add_interaction_features: bool = False,
        add_polynomial_features: bool = False,
        log_transform_skewed: bool = True,
        skew_threshold: float = 1.0,
        drop_high_cardinality: bool = True,
        cardinality_threshold: int = 50,
        random_state: int = 42
    ):
        self.add_datetime_features = add_datetime_features
        self.add_interaction_features = add_interaction_features
        self.add_polynomial_features = add_polynomial_features
        self.log_transform_skewed = log_transform_skewed
        self.skew_threshold = skew_threshold
        self.drop_high_cardinality = drop_high_cardinality
        self.cardinality_threshold = cardinality_threshold
        self.random_state = random_state
        
        self.datetime_columns_ = []
        self.skewed_columns_ = []
        self.high_cardinality_columns_ = []
        self.created_features_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        # Detect datetime columns
        self.datetime_columns_ = X.select_dtypes(include=["datetime64"]).columns.tolist()
        
        # Detect skewed numerical columns
        numerical = X.select_dtypes(include=[np.number])
        if len(numerical.columns) > 0:
            skewness = numerical.skew()
            self.skewed_columns_ = skewness[abs(skewness) > self.skew_threshold].index.tolist()
        
        # Detect high cardinality categorical
        categorical = X.select_dtypes(include=["object", "category"])
        if len(categorical.columns) > 0:
            cardinalities = categorical.nunique()
            self.high_cardinality_columns_ = cardinalities[cardinalities > self.cardinality_threshold].index.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.created_features_ = []
        
        # Log transform skewed features
        if self.log_transform_skewed and self.skewed_columns_:
            for col in self.skewed_columns_:
                if col in X.columns:
                    # Add 1 to handle zeros
                    min_val = X[col].min()
                    if min_val <= 0:
                        X[f"{col}_log"] = np.log1p(X[col] - min_val + 1)
                    else:
                        X[f"{col}_log"] = np.log1p(X[col])
                    self.created_features_.append(f"{col}_log")
        
        # Datetime features
        if self.add_datetime_features and self.datetime_columns_:
            for col in self.datetime_columns_:
                if col in X.columns:
                    X[f"{col}_year"] = X[col].dt.year
                    X[f"{col}_month"] = X[col].dt.month
                    X[f"{col}_day"] = X[col].dt.day
                    X[f"{col}_dayofweek"] = X[col].dt.dayofweek
                    X[f"{col}_quarter"] = X[col].dt.quarter
                    X[f"{col}_is_weekend"] = (X[col].dt.dayofweek >= 5).astype(int)
                    self.created_features_.extend([
                        f"{col}_year", f"{col}_month", f"{col}_day",
                        f"{col}_dayofweek", f"{col}_quarter", f"{col}_is_weekend"
                    ])
        
        # Interaction features (top 5 numerical)
        if self.add_interaction_features:
            numerical = X.select_dtypes(include=[np.number]).columns.tolist()
            for i, col1 in enumerate(numerical[:5]):
                for col2 in numerical[i+1:6]:
                    if col1 in X.columns and col2 in X.columns:
                        X[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                        X[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-8)
                        self.created_features_.extend([f"{col1}_x_{col2}", f"{col1}_div_{col2}"])
        
        # Drop high cardinality
        if self.drop_high_cardinality and self.high_cardinality_columns_:
            X = X.drop(columns=[c for c in self.high_cardinality_columns_ if c in X.columns])
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.created_features_


def extract_datetime_features(df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
    """Extract features from datetime columns."""
    df = df.copy()
    for col in datetime_columns:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
    return df


def create_ratio_features(df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
    """Create ratio features between numerical columns."""
    df = df.copy()
    for i, col1 in enumerate(numerical_columns):
        for col2 in numerical_columns[i+1:]:
            if col1 in df.columns and col2 in df.columns:
                df[f"{col1}_ratio_{col2}"] = df[col1] / (df[col2] + 1e-8)
    return df


def create_aggregation_features(df: pd.DataFrame, group_column: str, agg_columns: List[str]) -> pd.DataFrame:
    """Create aggregation features grouped by categorical column."""
    if group_column not in df.columns:
        return df
    
    df = df.copy()
    agg_functions = ["mean", "std", "min", "max", "count"]
    
    for col in agg_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            for func in agg_functions:
                grouped = df.groupby(group_column)[col].transform(func)
                df[f"{col}_{func}_by_{group_column}"] = grouped
    
    return df


def drop_redundant_features(df: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """Drop highly correlated features."""
    numerical = df.select_dtypes(include=[np.number])
    if len(numerical.columns) < 2:
        return df, []
    
    corr_matrix = numerical.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return df.drop(columns=to_drop), to_drop


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    method: str = "onehot",
    max_categories: int = 20
) -> pd.DataFrame:
    """Encode categorical features."""
    df = df.copy()
    
    for col in categorical_columns:
        if col not in df.columns:
            continue
        
        n_unique = df[col].nunique()
        
        if method == "onehot" and n_unique <= max_categories:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        elif method == "label":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        elif method == "frequency":
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
        elif method == "target":
            # Requires target - handled in pipeline
            pass
    
    return df