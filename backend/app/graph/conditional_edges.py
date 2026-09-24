# Conditional Edges for LangGraph
from typing import Dict, Any
from app.graph.state import AgentState


def should_continue_improvement(state: AgentState) -> str:
    """
    Decide whether to continue the improvement loop.
    
    Returns:
        "continue" - Run another iteration
        "stop" - Generate final report
    """
    # Check max iterations
    if state.get("iteration_number", 0) >= state.get("max_iterations", 5):
        return "stop"
    
    # Check if critic says stop
    if not state.get("should_continue", True):
        return "stop"
    
    # Check minimum improvement threshold
    improvement = state.get("improvement")
    min_threshold = state.get("min_improvement_threshold", 0.005)
    if improvement is not None and improvement < min_threshold:
        return "stop"
    
    return "continue"


def select_next_action(state: AgentState) -> str:
    """
    Select the next action based on critic feedback.
    
    Returns action name that maps to executable tool.
    """
    # Use next_action from improvement_router if already set
    next_action = state.get("next_action")
    if next_action and next_action != "stop":
        return next_action
    
    # Fallback: compute from critic feedback
    critic_feedback = state.get("critic_feedback", {})
    primary_action = critic_feedback.get("primary_action", "none")
    
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
    
    return action_map.get(primary_action, "stop")


def evaluate_stopping_conditions(state: AgentState) -> Dict[str, Any]:
    """
    Evaluate all stopping conditions.
    
    Returns dict with should_stop boolean and reason.
    """
    reasons = []
    should_stop = False
    
    # Max iterations
    if state.get("iteration_number", 0) >= state.get("max_iterations", 5):
        reasons.append("Maximum iterations reached")
        should_stop = True
    
    # Performance target reached
    best_metrics = state.get("best_metrics") or {}
    problem_type = state.get("problem_type", "classification")
    target_metric = "f1" if problem_type == "classification" else "r2"
    target_value = 0.95 if problem_type == "classification" else 0.9
    if best_metrics.get(target_metric, 0) >= target_value:
        reasons.append(f"Target {target_metric} score ({target_value}) reached")
        should_stop = True
    
    # No meaningful improvement
    improvement = state.get("improvement", 1.0)
    min_threshold = state.get("min_improvement_threshold", 0.005)
    if improvement < min_threshold:
        reasons.append(f"Improvement ({improvement:.4f}) below threshold ({min_threshold})")
        should_stop = True
    
    # Critic says stop
    if not state.get("should_continue", True):
        reasons.append("Critic recommends stopping")
        should_stop = True
    
    # No more improvement actions available
    critic_feedback = state.get("critic_feedback", {})
    if critic_feedback.get("status") == "satisfactory":
        reasons.append("Critic: performance satisfactory")
        should_stop = True
    
    return {
        "should_stop": should_stop,
        "reasons": reasons,
        "primary_reason": reasons[0] if reasons else "Unknown"
    }