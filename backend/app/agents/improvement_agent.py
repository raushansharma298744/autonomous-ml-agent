# Improvement Router - Routes to appropriate improvement action
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def select_next_action(critic_feedback: Dict[str, Any]) -> str:
    """Select next action based on critic feedback."""
    primary_action = critic_feedback.get("primary_action", "none")
    recommended = critic_feedback.get("recommended_actions", [])
    
    action_map = {
        "class_weight": "apply_class_weight",
        "resample": "apply_resampling", 
        "smote": "apply_smote",
        "threshold_adjustment": "adjust_threshold",
        "feature_engineering": "engineer_features",
        "different_model": "select_different_model",
        "hyperparameter_tuning": "run_hyperparameter_tuning",
        "none": "stop",
    }
    
    # Use primary action if available
    if primary_action in action_map:
        return action_map[primary_action]
    
    # Fall back to first recommended action
    for action in recommended:
        if action in action_map:
            return action_map[action]
    
    return "stop"


def create_improvement_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for improvement routing."""
    critic_feedback = state.get("critic_feedback", {})
    should_continue = state.get("should_continue", False)
    
    if not should_continue:
        logger.info("[ImprovementRouter] Stopping - no continuation needed")
        return {
            "next_action": "stop",
            "current_step": "improvement_router",
        }
    
    next_action = select_next_action(critic_feedback)
    
    # Increment iteration
    iteration = state.get("iteration_number", 0) + 1
    
    # Track improvement action
    improvement_history = state.get("improvement_history", [])
    improvement_history.append({
        "iteration": iteration,
        "action": next_action,
        "reason": critic_feedback.get("summary", ""),
        "critic_status": critic_feedback.get("status", ""),
    })
    
    logger.info(f"[ImprovementRouter] Iteration {iteration}: selected action = {next_action}")
    
    return {
        "next_action": next_action,
        "iteration_number": iteration,
        "improvement_history": improvement_history,
        "current_step": "improvement_router",
    }


def apply_class_weight_improvement(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply class weight improvement - retrain with balanced weights."""
    logger.info("[Improvement] Applying class_weight='balanced'")
    
    # The model agent will handle this via class_weights in state
    return {
        "class_weights": "balanced",
        "improvement_applied": "class_weight",
        "current_step": "apply_class_weight",
    }


def apply_resampling_improvement(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply resampling improvement."""
    logger.info("[Improvement] Applying resampling")
    
    # Resampling would be applied during preprocessing
    # For now, mark it for next preprocessing run
    return {
        "resampling_method": "smote",
        "improvement_applied": "resample",
        "current_step": "apply_resampling",
    }


def apply_smote_improvement(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply SMOTE improvement."""
    return apply_resampling_improvement(state)


def adjust_threshold_improvement(state: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust decision threshold."""
    logger.info("[Improvement] Adjusting threshold")
    
    # Threshold adjustment happens at prediction time
    return {
        "threshold_adjustment": True,
        "improvement_applied": "threshold_adjustment",
        "current_step": "adjust_threshold",
    }


def engineer_features_improvement(state: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger feature engineering."""
    logger.info("[Improvement] Triggering feature engineering")
    return {
        "trigger_feature_engineering": True,
        "improvement_applied": "feature_engineering",
        "current_step": "engineer_features",
    }


def select_different_model_improvement(state: Dict[str, Any]) -> Dict[str, Any]:
    """Select a different model."""
    logger.info("[Improvement] Selecting different model")
    
    current_model = state.get("current_model")
    candidate_models = state.get("candidate_models", [])
    
    # Pick next best model not yet tried
    tried_models = set()
    for exp in state.get("experiments", []):
        tried_models.add(exp.get("model_name"))
    
    available = [m for m in candidate_models if m not in tried_models]
    if not available and candidate_models:
        available = [candidate_models[0]]  # Fallback to first
    
    next_model = available[0] if available else candidate_models[0] if candidate_models else "RandomForestClassifier"
    
    return {
        "current_model": next_model,
        "candidate_models": [next_model],
        "improvement_applied": "different_model",
        "current_step": "select_different_model",
    }


def run_hyperparameter_tuning_improvement(state: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger hyperparameter tuning."""
    logger.info("[Improvement] Triggering hyperparameter tuning")
    return {
        "trigger_tuning": True,
        "improvement_applied": "hyperparameter_tuning",
        "current_step": "run_hyperparameter_tuning",
    }