# Tuning Tools - Optuna hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Any, Callable, Optional, List
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings("ignore")


def get_search_space(model_name: str) -> Dict[str, Any]:
    """Get Optuna search space for a model."""
    spaces = {
        "LogisticRegression": {
            "C": ("loguniform", 1e-4, 1e2),
            "penalty": ("categorical", ["l1", "l2", "elasticnet"]),
            "solver": ("categorical", ["liblinear", "saga"]),
            "max_iter": ("categorical", [1000, 2000]),
        },
        "RandomForestClassifier": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 30),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("categorical", ["sqrt", "log2", None]),
            "bootstrap": ("categorical", [True, False]),
        },
        "RandomForestRegressor": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 30),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("categorical", ["sqrt", "log2", None]),
            "bootstrap": ("categorical", [True, False]),
        },
        "GradientBoostingClassifier": {
            "n_estimators": ("int", 50, 300),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_depth": ("int", 3, 10),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "subsample": ("float", 0.6, 1.0),
        },
        "GradientBoostingRegressor": {
            "n_estimators": ("int", 50, 300),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_depth": ("int", 3, 10),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "subsample": ("float", 0.6, 1.0),
        },
        "XGBClassifier": {
            "n_estimators": ("int", 50, 500),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_depth": ("int", 3, 12),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "gamma": ("float", 0, 5),
            "reg_alpha": ("loguniform", 1e-8, 1.0),
            "reg_lambda": ("loguniform", 1e-8, 1.0),
        },
        "XGBRegressor": {
            "n_estimators": ("int", 50, 500),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_depth": ("int", 3, 12),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "gamma": ("float", 0, 5),
            "reg_alpha": ("loguniform", 1e-8, 1.0),
            "reg_lambda": ("loguniform", 1e-8, 1.0),
        },
        "HistGradientBoostingClassifier": {
            "max_iter": ("int", 50, 300),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_depth": ("int", 3, 20),
            "min_samples_leaf": ("int", 1, 50),
            "l2_regularization": ("loguniform", 1e-8, 1.0),
        },
        "HistGradientBoostingRegressor": {
            "max_iter": ("int", 50, 300),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_depth": ("int", 3, 20),
            "min_samples_leaf": ("int", 1, 50),
            "l2_regularization": ("loguniform", 1e-8, 1.0),
        },
    }
    
    return spaces.get(model_name, {})


def suggest_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest hyperparameters from search space."""
    params = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == "int":
            params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
        elif param_type == "float":
            params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
        elif param_type == "loguniform":
            params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_config[1])
    
    return params


def create_objective(
    model_factory: Callable[[Dict[str, Any]], BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    cv: int = 5,
    scoring: str = None,
    sample_weight: Optional[np.ndarray] = None
) -> Callable:
    """Create Optuna objective function."""
    
    if scoring is None:
        if problem_type == "classification":
            scoring = "f1_weighted"
        else:
            scoring = "neg_mean_squared_error"
    
    # CV strategy
    if problem_type == "classification":
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    def objective(trial: optuna.Trial) -> float:
        # Get search space from model factory
        model_name = model_factory.__name__ if hasattr(model_factory, '__name__') else 'model'
        
        # Use trial user attrs to pass search space
        search_space = trial.study.user_attrs.get("search_space", {})
        if not search_space:
            # Fallback
            search_space = {}
        
        params = suggest_params(trial, search_space)
        
        try:
            model = model_factory(params)
            
            # Cross-validation
            if sample_weight is not None:
                # Can't easily use sample_weight with cross_val_score
                scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
            else:
                scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
            
            return float(np.mean(scores))
        
        except Exception as e:
            # Return poor score for failed trials
            if problem_type == "classification":
                return 0.0
            else:
                return float('inf')
    
    return objective


def optimize_hyperparameters(
    model_factory: Callable[[Dict[str, Any]], BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    problem_type: str,
    n_trials: int = 50,
    cv: int = 5,
    scoring: str = None,
    timeout: int = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization."""
    
    search_space = get_search_space(model_name)
    if not search_space:
        return {}
    
    # Create study
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize" if problem_type == "classification" else "minimize",
        sampler=sampler
    )
    study.set_user_attr("search_space", search_space)
    
    # Create objective
    objective = create_objective(
        model_factory, X, y, problem_type, cv, scoring
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study
    }


def train_best_model(
    model_factory: Callable[[Dict[str, Any]], BaseEstimator],
    best_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray
) -> BaseEstimator:
    """Train model with best hyperparameters on full training set."""
    model = model_factory(best_params)
    model.fit(X_train, y_train)
    return model