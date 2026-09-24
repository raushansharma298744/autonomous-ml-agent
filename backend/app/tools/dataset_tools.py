# Dataset Tools - Deterministic data analysis functions
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path)


def profile_dataset(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Generate comprehensive dataset profile."""
    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=["datetime64"]).columns.tolist(),
    }
    
    # Target analysis
    if target_column in df.columns:
        target = df[target_column]
        profile["target_analysis"] = analyze_target(target)
    
    # Basic statistics for numerical columns
    numerical_cols = profile["numerical_columns"]
    if numerical_cols:
        profile["numerical_stats"] = df[numerical_cols].describe().to_dict()
    
    # Unique value counts for categorical
    categorical_cols = profile["categorical_columns"]
    if categorical_cols:
        profile["categorical_unique_counts"] = {
            col: int(df[col].nunique()) for col in categorical_cols
        }
    
    # Outlier detection (IQR method)
    profile["outliers"] = detect_outliers(df[numerical_cols]) if numerical_cols else {}
    
    return profile


def analyze_target(target: pd.Series) -> Dict[str, Any]:
    """Analyze target variable."""
    analysis = {
        "dtype": str(target.dtype),
        "missing": int(target.isnull().sum()),
        "unique_values": int(target.nunique()),
    }
    
    # Classification if: object/category dtype OR integer with few unique values
    is_categorical = target.dtype in ["object", "category"]
    is_integer_categorical = (
        pd.api.types.is_integer_dtype(target) and 
        target.nunique() <= 20 and 
        target.nunique() / len(target) < 0.5
    )
    
    if is_categorical or is_integer_categorical:
        # Classification
        analysis["problem_type"] = "classification"
        analysis["class_distribution"] = target.value_counts().to_dict()
        analysis["class_balance"] = target.value_counts(normalize=True).to_dict()
        analysis["is_imbalanced"] = check_class_imbalance(target)
    else:
        # Regression
        analysis["problem_type"] = "regression"
        analysis["target_stats"] = {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
            "median": float(target.median()),
            "skewness": float(target.skew()),
            "kurtosis": float(target.kurtosis()),
        }
    
    return analysis


def check_class_imbalance(target: pd.Series, threshold: float = 0.1) -> bool:
    """Check if classification target is imbalanced."""
    class_counts = target.value_counts(normalize=True)
    min_class_ratio = class_counts.min()
    return min_class_ratio < threshold


def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> Dict[str, Any]:
    """Detect outliers in numerical columns."""
    outliers = {}
    
    for col in df.columns:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            outliers[col] = {
                "count": int(outlier_mask.sum()),
                "percentage": float(outlier_mask.mean() * 100),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
    
    return outliers


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> tuple:
    """Split dataset into train/test with optional stratification."""
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    stratify_param = y if stratify and y.dtype in ["object", "category"] or y.nunique() < 20 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test


def save_dataset(df: pd.DataFrame, path: str):
    """Save dataset to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)