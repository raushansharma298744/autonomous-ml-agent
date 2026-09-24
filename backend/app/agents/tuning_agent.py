# Hyperparameter Tuning Agent - Optuna-based optimization
import logging
import time
from typing import Dict, Any, Callable, Optional
import numpy as np
from app.tools.tuning_tools import (
    optimize_hyperparameters, train_best_model, get_search_space
)
from app.tools.model_tools import create_model
from app.tools.evaluation_tools import evaluate_model

logger = logging.getLogger(__name__)


def should_tune_model(
    best_metrics: Dict[str, Any],
    problem_type: str,
    iteration: int,
    max_iterations: int
) -> bool:
    """Determine if hyperparameter tuning should be applied."""
    # Always tune on iteration 0 (first run after baseline)
    if iteration == 0:
        return True
    # Tune if performance is promising but not optimal
    if problem_type == "classification":
        f1 = best_metrics.get("f1", 0)
        return 0.5 < f1 < 0.95  # Promising but room for improvement
    else:
        r2 = best_metrics.get("r2", 0)
        return 0.3 < r2 < 0.95
    
    return False


def create_model_factory(model_name: str, base_params: Dict[str, Any] = None) -> Callable:
    """Create a model factory function for Optuna."""
    def factory(params: Dict[str, Any]):
        config = {
            "model": get_model_path(model_name),
            "params": {**(base_params or {}), **params}
        }
        return create_model(config)
    return factory


def get_model_path(model_name: str) -> str:
    """Get full module path for model."""
    paths = {
        "LogisticRegression": "sklearn.linear_model.LogisticRegression",
        "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
        "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
        "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier",
        "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
        "XGBClassifier": "xgboost.XGBClassifier",
        "XGBRegressor": "xgboost.XGBRegressor",
        "HistGradientBoostingClassifier": "sklearn.ensemble.HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor": "sklearn.ensemble.HistGradientBoostingRegressor",
    }
    return paths.get(model_name, f"sklearn.ensemble.{model_name}")


def run_hyperparameter_tuning(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    base_params: Dict[str, Any] = None,
    n_trials: int = 30,
    cv: int = 3
) -> Dict[str, Any]:
    """Run hyperparameter optimization for a model."""
    logger.info(f"[TuningAgent] Starting tuning for {model_name} ({n_trials} trials)")
    
    # Check if search space exists
    search_space = get_search_space(model_name)
    if not search_space:
        logger.warning(f"[TuningAgent] No search space for {model_name}")
        return {}
    
    # Create model factory
    model_factory = create_model_factory(model_name, base_params)
    
    # Run optimization
    start_time = time.time()
    try:
        result = optimize_hyperparameters(
            model_factory=model_factory,
            X=X_train,
            y=y_train,
            model_name=model_name,
            problem_type=problem_type,
            n_trials=n_trials,
            cv=cv,
            timeout=300,  # 5 minute timeout
        )
    except Exception as e:
        logger.error(f"[TuningAgent] Tuning failed: {e}")
        return {}
    
    duration = time.time() - start_time
    
    if not result or "best_params" not in result:
        logger.warning(f"[TuningAgent] No valid results from tuning")
        return {}
    
    # Train best model on full training set
    best_model = train_best_model(
        model_factory, result["best_params"], X_train, y_train
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(best_model, X_test, y_test, problem_type, return_proba=True)
    
    tuning_result = {
        "model_name": model_name,
        "best_params": result["best_params"],
        "best_cv_score": result["best_value"],
        "test_metrics": test_metrics,
        "n_trials": result["n_trials"],
        "tuning_duration": duration,
        "improvement_action": "hyperparameter_tuning",
    }
    
    logger.info(f"[TuningAgent] Tuning complete: CV score={result['best_value']:.4f}, "
                f"Test F1/R2={test_metrics.get('f1', test_metrics.get('r2', 0)):.4f}")
    
    return tuning_result


def create_tuning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for hyperparameter tuning."""
    X_train = state.get("X_train")
    y_train = state.get("y_train")
    X_test = state.get("X_test")
    y_test = state.get("y_test")
    problem_type = state.get("problem_type", "classification")
    best_experiment = state.get("best_experiment", {})
    current_model = state.get("current_model")
    iteration = state.get("iteration_number", 0)
    max_iterations = state.get("max_iterations", 5)
    
    # Check if we should tune
    best_metrics = best_experiment.get("metrics", {})
    
    if not should_tune_model(best_metrics, problem_type, iteration, max_iterations):
        logger.info("[TuningAgent] Skipping tuning based on current performance")
        return {
            "tuning_results": {},
            "current_step": "tuning",
        }
    
    if not current_model:
        logger.warning("[TuningAgent] No current model to tune")
        return {
            "tuning_results": {},
            "current_step": "tuning",
        }
    
    # Run tuning
    tuning_result = run_hyperparameter_tuning(
        model_name=current_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        problem_type=problem_type,
        base_params=best_experiment.get("hyperparameters"),
        n_trials=30,
        cv=3,
    )
    
    if not tuning_result:
        return {
            "tuning_results": {},
            "current_step": "tuning",
        }
    
    # Create experiment record for tuned model
    tuned_experiment = {
        "name": f"{current_model} (tuned, iteration {iteration})",
        "model_name": current_model,
        "model": None,  # Model object not serializable
        "metrics": tuning_result["test_metrics"],
        "hyperparameters": tuning_result["best_params"],
        "training_duration": tuning_result["tuning_duration"],
        "iteration": iteration,
        "improvement_action": "hyperparameter_tuning",
        "cv_score": tuning_result["best_cv_score"],
    }
    
    # Update experiments list
    experiments = state.get("experiments", [])
    experiments.append(tuned_experiment)
    
    # Check if tuned model is better
    primary_metric = "f1" if problem_type == "classification" else "r2"
    tuned_score = tuning_result["test_metrics"].get(primary_metric, 0)
    best_score = best_metrics.get(primary_metric, 0)
    
    if tuned_score > best_score:
        logger.info(f"[TuningAgent] Tuned model improved: {best_score:.4f} → {tuned_score:.4f}")
        return {
            "experiments": experiments,
            "current_experiment": tuned_experiment,
            "best_experiment": tuned_experiment,
            "best_metrics": tuning_result["test_metrics"],
            "tuning_results": tuning_result,
            "current_step": "tuning",
        }
    else:
        logger.info(f"[TuningAgent] Tuned model did not improve: {tuned_score:.4f} <= {best_score:.4f}")
        return {
            "experiments": experiments,
            "tuning_results": tuning_result,
            "current_step": "tuning",
        }