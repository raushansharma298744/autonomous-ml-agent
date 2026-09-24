# Model Selection Agent - Selects and trains candidate models
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from app.tools.model_tools import (
    get_classification_models, get_regression_models,
    create_model, train_model, train_multiple_models
)
from app.tools.evaluation_tools import evaluate_model, compare_models, get_primary_metric

logger = logging.getLogger(__name__)


def get_candidate_models(problem_type: str, dataset_size: int) -> Dict[str, Dict[str, Any]]:
    """Get list of candidate models for problem type."""
    if problem_type == "classification":
        return get_classification_models(dataset_size)
    else:
        return get_regression_models(dataset_size)


def select_best_model(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    class_imbalanced: bool = False
) -> Tuple[str, Dict[str, Any], Any]:
    """Select best model based on evaluation."""
    comparison = compare_models(trained_models, X_test, y_test, problem_type)
    
    if comparison.empty:
        return None, {}, None
    
    # Get primary metric
    primary_metric, _ = get_primary_metric(
        comparison.iloc[0].to_dict(), problem_type, class_imbalanced
    )
    
    best_model_name = comparison.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_metrics = comparison.iloc[0].to_dict()
    
    logger.info(f"[ModelAgent] Best model: {best_model_name} "
                f"({primary_metric}={best_metrics.get(primary_metric, 0):.4f})")
    
    return best_model_name, best_metrics, best_model


def train_baseline_models(
    models_config: Dict[str, Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: Dict[int, float] = None,
    sample_weight: np.ndarray = None
) -> Dict[str, Any]:
    """Train multiple baseline models."""
    trained = {}
    
    for name, config in models_config.items():
        try:
            model = create_model(config)
            
            # Apply class weights if supported
            if class_weights and config.get("supports_class_weight", False):
                model.set_params(class_weight=class_weights)
            
            start_time = time.time()
            model = train_model(model, X_train, y_train, sample_weight=sample_weight)
            duration = time.time() - start_time
            
            trained[name] = {
                "model": model,
                "config": config,
                "training_duration": duration,
            }
            logger.info(f"[ModelAgent] Trained {name} in {duration:.2f}s")
        except Exception as e:
            logger.error(f"[ModelAgent] Failed to train {name}: {e}")
            trained[name] = None
    
    return trained


def create_model_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for model selection and training."""
    X_train = state.get("X_train")
    y_train = state.get("y_train")
    X_test = state.get("X_test")
    y_test = state.get("y_test")
    problem_type = state.get("problem_type", "classification")
    dataset_profile = state.get("dataset_profile", {})
    class_weights = state.get("class_weights")
    candidate_models = state.get("candidate_models", [])
    iteration = state.get("iteration_number", 0)
    # Preserve previous experiments
    existing_experiments = list(state.get("experiments") or [])
    previous_best_metrics = state.get("best_metrics") or {}
    
    logger.info(f"[ModelAgent] Starting model selection (iteration {iteration})")
    
    # Get candidate models
    dataset_size = dataset_profile.get("n_rows", len(X_train))
    all_models = get_candidate_models(problem_type, dataset_size)
    
    # Filter to candidate models if specified
    if candidate_models:
        models_config = {k: v for k, v in all_models.items() if k in candidate_models}
    else:
        models_config = all_models
    
    # Train baseline models
    trained = train_baseline_models(models_config, X_train, y_train, class_weights)
    
    # Filter successful models
    successful = {k: v for k, v in trained.items() if v is not None}
    
    if not successful:
        logger.error("[ModelAgent] No models trained successfully")
        return {
            "experiments": existing_experiments,
            "current_experiment": None,
            "error": "No models trained successfully",
            "current_step": "model_selection",
        }
    
    # Evaluate all models
    model_objects = {k: v["model"] for k, v in successful.items()}
    comparison = compare_models(model_objects, X_test, y_test, problem_type)
    
    # Select best model
    class_imbalanced = dataset_profile.get("class_imbalance", False)
    best_model_name, best_metrics, best_model = select_best_model(
        model_objects, X_test, y_test, problem_type, class_imbalanced
    )
    
    # Create experiment records for this iteration
    new_experiments = []
    for name, info in successful.items():
        metrics = evaluate_model(info["model"], X_test, y_test, problem_type, return_proba=True)
        new_experiments.append({
            "name": f"{name} (iteration {iteration})",
            "model_name": name,
            "model": info["model"],
            "metrics": metrics,
            "hyperparameters": info["model"].get_params(),
            "training_duration": info["training_duration"],
            "iteration": iteration,
            "improvement_action": "baseline" if iteration == 0 else "improvement",
        })
    
    # Append to existing experiments
    all_experiments = existing_experiments + new_experiments
    
    # Best experiment of this iteration
    best_exp = next((e for e in new_experiments if e["model_name"] == best_model_name), new_experiments[0])
    
    # Determine global best across all experiments
    primary = "f1" if problem_type == "classification" else "r2"
    global_best = best_exp
    global_best_metric = best_metrics.get(primary, 0)
    prev_best_metric = previous_best_metrics.get(primary, 0) if previous_best_metrics else -1
    if prev_best_metric >= global_best_metric:
        # Keep previous global best
        global_best = state.get("best_experiment")
        global_best_metric = prev_best_metric
        best_metrics = previous_best_metrics
    else:
        # New global best
        global_best = best_exp
        best_metrics = best_metrics
    
    logger.info(f"[ModelAgent] Completed: {len(new_experiments)} models trained this iteration, "
                f"best this iter: {best_model_name} ({primary}={global_best_metric:.4f}), "
                f"global best {primary}={global_best_metric:.4f}")
    
    return {
        "experiments": all_experiments,
        "current_experiment": best_exp,
        "best_experiment": global_best,
        "best_metrics": best_metrics,
        "current_model": best_model_name,
        "current_step": "model_selection",
    }