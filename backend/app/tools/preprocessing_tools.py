# Preprocessing Tools - Deterministic preprocessing functions
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from typing import Dict, Any, List, Tuple, Optional


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    numerical_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    scale_numerical: bool = True,
    encode_categorical: str = "onehot"  # onehot, label
) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for preprocessing."""
    
    transformers = []
    
    # Numerical pipeline
    if numerical_columns:
        num_steps = [
            ("imputer", SimpleImputer(strategy=numerical_strategy))
        ]
        if scale_numerical:
            num_steps.append(("scaler", StandardScaler()))
        
        numerical_pipeline = Pipeline(num_steps)
        transformers.append(("num", numerical_pipeline, numerical_columns))
    
    # Categorical pipeline
    if categorical_columns:
        cat_steps = [
            ("imputer", SimpleImputer(strategy=categorical_strategy))
        ]
        
        if encode_categorical == "onehot":
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif encode_categorical == "label":
            # Label encoding is handled separately per column
            pass
        
        categorical_pipeline = Pipeline(cat_steps)
        transformers.append(("cat", categorical_pipeline, categorical_columns))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    
    return preprocessor


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame):
    """Fit preprocessor on training data only (no data leakage)."""
    preprocessor.fit(X_train)
    return preprocessor


def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transform data using fitted preprocessor."""
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> List[str]:
    """Get feature names after preprocessing."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except:
        # Fallback for older sklearn versions
        return [f"feature_{i}" for i in range(preprocessor.transform(X).shape[1])]


def encode_target(y: pd.Series, problem_type: str) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
    """Encode target variable for classification."""
    if problem_type == "classification":
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        return y_encoded, encoder
    return y.values, None


def decode_target(y_encoded: np.ndarray, encoder: LabelEncoder) -> np.ndarray:
    """Decode target variable."""
    return encoder.inverse_transform(y_encoded)


def apply_class_weight(y: np.ndarray, method: str = "balanced") -> Dict[int, float]:
    """Calculate class weights for imbalanced classification."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight(class_weight=method, classes=classes, y=y)
    return dict(zip(classes, weights))


def apply_resampling(X: np.ndarray, y: np.ndarray, method: str = "smote", random_state: int = 42):
    """Apply resampling to handle class imbalance."""
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=random_state)
    elif method == "adasyn":
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN(random_state=random_state)
    elif method == "random_over":
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=random_state)
    elif method == "random_under":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled