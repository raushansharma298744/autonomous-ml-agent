# Model Tools - Deterministic model training functions
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings("ignore")

# Optional imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


def get_classification_models(dataset_size: int = 10000) -> Dict[str, Dict[str, Any]]:
    """Get candidate classification models with default parameters."""
    models = {
        "LogisticRegression": {
            "model": "sklearn.linear_model.LogisticRegression",
            "params": {
                "max_iter": 1000,
                "random_state": 42,
                "n_jobs": -1
            },
            "suitable_for_large": True,
            "supports_class_weight": True
        },
        "RandomForestClassifier": {
            "model": "sklearn.ensemble.RandomForestClassifier",
            "params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1
            },
            "suitable_for_large": True,
            "supports_class_weight": True
        },
        "ExtraTreesClassifier": {
            "model": "sklearn.ensemble.ExtraTreesClassifier",
            "params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1
            },
            "suitable_for_large": True,
            "supports_class_weight": True
        },
        "GradientBoostingClassifier": {
            "model": "sklearn.ensemble.GradientBoostingClassifier",
            "params": {
                "n_estimators": 100,
                "random_state": 42
            },
            "suitable_for_large": False,
            "supports_class_weight": False
        },
        "AdaBoostClassifier": {
            "model": "sklearn.ensemble.AdaBoostClassifier",
            "params": {
                "n_estimators": 100,
                "random_state": 42
            },
            "suitable_for_large": False,
            "supports_class_weight": False
        },
        "BaggingClassifier": {
            "model": "sklearn.ensemble.BaggingClassifier",
            "params": {
                "n_estimators": 100,
                "random_state": 42,
                "n_jobs": -1
            },
            "suitable_for_large": True,
            "supports_class_weight": False
        },
        "DecisionTreeClassifier": {
            "model": "sklearn.tree.DecisionTreeClassifier",
            "params": {
                "random_state": 42,
                "class_weight": "balanced"
            },
            "suitable_for_large": True,
            "supports_class_weight": True
        },
        "KNeighborsClassifier": {
            "model": "sklearn.neighbors.KNeighborsClassifier",
            "params": {
                "n_neighbors": 5,
                "n_jobs": -1
            },
            "suitable_for_large": False,
            "supports_class_weight": False
        },
        "SVC": {
            "model": "sklearn.svm.SVC",
            "params": {
                "probability": True,
                "random_state": 42,
                "class_weight": "balanced"
            },
            "suitable_for_large": False,
            "supports_class_weight": True
        },
        "GaussianNB": {
            "model": "sklearn.naive_bayes.GaussianNB",
            "params": {},
            "suitable_for_large": True,
            "supports_class_weight": False
        },
        "XGBClassifier": {
            "model": "xgboost.XGBClassifier",
            "params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
                "use_label_encoder": False,
                "eval_metric": "logloss"
            },
            "suitable_for_large": True,
            "supports_class_weight": True
        },
    }
    
    # Add HistGradientBoosting for large datasets
    if dataset_size > 10000:
        models["HistGradientBoostingClassifier"] = {
            "model": "sklearn.ensemble.HistGradientBoostingClassifier",
            "params": {
                "max_iter": 200,
                "random_state": 42,
                "class_weight": "balanced"
            },
            "suitable_for_large": True,
            "supports_class_weight": True
        }
    
    # LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models["LGBMClassifier"] = {
            "model": "lightgbm.LGBMClassifier",
            "params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
                "class_weight": "balanced"
            },
            "suitable_for_large": True,
            "supports_class_weight": True
        }
    
    # CatBoost if available
    if CATBOOST_AVAILABLE:
        models["CatBoostClassifier"] = {
            "model": "catboost.CatBoostClassifier",
            "params": {
                "iterations": 200,
                "random_state": 42,
                "verbose": False,
                "thread_count": -1,
                "auto_class_weights": "Balanced"
            },
            "suitable_for_large": True,
            "supports_class_weight": True
        }
    
    return models


def get_regression_models(dataset_size: int = 10000) -> Dict[str, Dict[str, Any]]:
    """Get candidate regression models with default parameters."""
    models = {
        "LinearRegression": {
            "model": "sklearn.linear_model.LinearRegression",
            "params": {
                "n_jobs": -1
            },
            "suitable_for_large": True
        },
        "Ridge": {
            "model": "sklearn.linear_model.Ridge",
            "params": {
                "alpha": 1.0,
                "random_state": 42
            },
            "suitable_for_large": True
        },
        "Lasso": {
            "model": "sklearn.linear_model.Lasso",
            "params": {
                "alpha": 1.0,
                "random_state": 42,
                "max_iter": 2000
            },
            "suitable_for_large": True
        },
        "ElasticNet": {
            "model": "sklearn.linear_model.ElasticNet",
            "params": {
                "alpha": 1.0,
                "l1_ratio": 0.5,
                "random_state": 42,
                "max_iter": 2000
            },
            "suitable_for_large": True
        },
        "RandomForestRegressor": {
            "model": "sklearn.ensemble.RandomForestRegressor",
            "params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1
            },
            "suitable_for_large": True
        },
        "ExtraTreesRegressor": {
            "model": "sklearn.ensemble.ExtraTreesRegressor",
            "params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1
            },
            "suitable_for_large": True
        },
        "GradientBoostingRegressor": {
            "model": "sklearn.ensemble.GradientBoostingRegressor",
            "params": {
                "n_estimators": 100,
                "random_state": 42
            },
            "suitable_for_large": False
        },
        "AdaBoostRegressor": {
            "model": "sklearn.ensemble.AdaBoostRegressor",
            "params": {
                "n_estimators": 100,
                "random_state": 42
            },
            "suitable_for_large": False
        },
        "BaggingRegressor": {
            "model": "sklearn.ensemble.BaggingRegressor",
            "params": {
                "n_estimators": 100,
                "random_state": 42,
                "n_jobs": -1
            },
            "suitable_for_large": True
        },
        "DecisionTreeRegressor": {
            "model": "sklearn.tree.DecisionTreeRegressor",
            "params": {
                "random_state": 42
            },
            "suitable_for_large": True
        },
        "KNeighborsRegressor": {
            "model": "sklearn.neighbors.KNeighborsRegressor",
            "params": {
                "n_neighbors": 5,
                "n_jobs": -1
            },
            "suitable_for_large": False
        },
        "SVR": {
            "model": "sklearn.svm.SVR",
            "params": {
                "kernel": "rbf",
                "C": 1.0
            },
            "suitable_for_large": False
        },
        "XGBRegressor": {
            "model": "xgboost.XGBRegressor",
            "params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0
            },
            "suitable_for_large": True
        },
    }
    
    if dataset_size > 10000:
        models["HistGradientBoostingRegressor"] = {
            "model": "sklearn.ensemble.HistGradientBoostingRegressor",
            "params": {
                "max_iter": 200,
                "random_state": 42
            },
            "suitable_for_large": True
        }
    
    if LIGHTGBM_AVAILABLE:
        models["LGBMRegressor"] = {
            "model": "lightgbm.LGBMRegressor",
            "params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1
            },
            "suitable_for_large": True
        }
    
    if CATBOOST_AVAILABLE:
        models["CatBoostRegressor"] = {
            "model": "catboost.CatBoostRegressor",
            "params": {
                "iterations": 200,
                "random_state": 42,
                "verbose": False,
                "thread_count": -1
            },
            "suitable_for_large": True
        }
    
    return models


def create_model(model_config: Dict[str, Any]) -> BaseEstimator:
    """Create model instance from config."""
    model_path = model_config["model"]
    params = model_config.get("params", {})
    
    # Dynamic import
    module_path, class_name = model_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    
    return model_class(**params)


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    **fit_params
) -> BaseEstimator:
    """Train a model."""
    if sample_weight is not None:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight, **fit_params)
        except TypeError:
            # Model doesn't support sample_weight
            model.fit(X_train, y_train, **fit_params)
    else:
        model.fit(X_train, y_train, **fit_params)
    return model


def train_multiple_models(
    models_config: Dict[str, Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, BaseEstimator]:
    """Train multiple models."""
    trained_models = {}
    
    for name, config in models_config.items():
        try:
            model = create_model(config)
            model = train_model(model, X_train, y_train, sample_weight)
            trained_models[name] = model
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            trained_models[name] = None
    
    return trained_models


def predict_model(
    model: BaseEstimator,
    X: np.ndarray,
    return_proba: bool = False
) -> np.ndarray:
    """Make predictions with a model."""
    if return_proba and hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X)


def get_model_params(model: BaseEstimator) -> Dict[str, Any]:
    """Get model parameters."""
    return model.get_params()


def set_model_params(model: BaseEstimator, params: Dict[str, Any]) -> BaseEstimator:
    """Set model parameters."""
    model.set_params(**params)
    return model


def recommend_best_algorithm(dataset_profile: Dict[str, Any], problem_type: str) -> List[str]:
    """Recommend top algorithms based on dataset characteristics."""
    n_rows = dataset_profile.get("n_rows", 0)
    n_features = dataset_profile.get("n_columns", 0)
    n_categorical = len(dataset_profile.get("categorical_columns", []))
    n_numerical = len(dataset_profile.get("numerical_columns", []))
    class_imbalance = dataset_profile.get("class_imbalance", False)
    high_cardinality = dataset_profile.get("high_cardinality_columns", [])
    
    recommendations = []
    
    # Large data: gradient boosting variants excel
    if n_rows > 50000:
        if problem_type == "classification":
            recommendations.extend(["HistGradientBoostingClassifier", "LGBMClassifier", "CatBoostClassifier", "XGBClassifier"])
        else:
            recommendations.extend(["HistGradientBoostingRegressor", "LGBMRegressor", "CatBoostRegressor", "XGBRegressor"])
    # Medium data: RandomForest / ExtraTrees strong
    elif n_rows > 5000:
        if problem_type == "classification":
            recommendations.extend(["RandomForestClassifier", "ExtraTreesClassifier", "XGBClassifier"])
        else:
            recommendations.extend(["RandomForestRegressor", "ExtraTreesRegressor", "XGBRegressor"])
    # Small data: simpler models, regularized linear
    else:
        if problem_type == "classification":
            recommendations.extend(["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"])
        else:
            recommendations.extend(["Ridge", "Lasso", "ElasticNet", "RandomForestRegressor"])
    
    # High cardinality categorical -> tree models handle better
    if high_cardinality:
        if problem_type == "classification":
            recommendations = ["CatBoostClassifier", "LGBMClassifier", "XGBClassifier"] + [m for m in recommendations if m not in ["CatBoostClassifier","LGBMClassifier","XGBClassifier"]]
        else:
            recommendations = ["CatBoostRegressor", "LGBMRegressor", "XGBRegressor"] + [m for m in recommendations if m not in ["CatBoostRegressor","LGBMRegressor","XGBRegressor"]]
    
    # Class imbalance -> models with built-in weighting
    if class_imbalance and problem_type == "classification":
        imbalance_models = ["CatBoostClassifier", "LGBMClassifier", "XGBClassifier", "RandomForestClassifier", "HistGradientBoostingClassifier"]
        recommendations = [m for m in imbalance_models if m in recommendations] + [m for m in recommendations if m not in imbalance_models]
    
    # High dimensional (many features) -> regularized linear / tree with feature importance
    if n_features > 100:
        if problem_type == "classification":
            recommendations = ["LogisticRegression", "LGBMClassifier", "XGBClassifier"] + [m for m in recommendations if m not in ["LogisticRegression","LGBMClassifier","XGBClassifier"]]
        else:
            recommendations = ["Ridge", "Lasso", "ElasticNet", "LGBMRegressor", "XGBRegressor"] + [m for m in recommendations if m not in ["Ridge","Lasso","ElasticNet","LGBMRegressor","XGBRegressor"]]
    
    # Ensure unique and limit
    seen = set()
    final = []
    for m in recommendations:
        if m not in seen:
            seen.add(m)
            final.append(m)
        if len(final) >= 5:
            break
    
    return final