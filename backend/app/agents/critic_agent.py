# Critic Agent - Analyzes results and recommends improvements
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


def analyze_experiment(
    experiment_metrics: Dict[str, Any],
    problem_type: str,
    target_distribution: Dict[str, Any],
    previous_metrics: Dict[str, Any] = None,
    train_metrics: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Analyze experiment results and identify weaknesses."""
    llm = get_llm_service()
    
    issues = []
    recommended_actions = []
    
    # Analyze based on problem type
    if problem_type == "classification":
        issues_cls, actions_cls = _analyze_classification(
            experiment_metrics, target_distribution, previous_metrics, train_metrics
        )
        issues.extend(issues_cls)
        recommended_actions.extend(actions_cls)
    else:
        issues_reg, actions_reg = _analyze_regression(
            experiment_metrics, previous_metrics, train_metrics
        )
        issues.extend(issues_reg)
        recommended_actions.extend(actions_reg)
    
    # General issues
    if previous_metrics:
        primary = "f1" if problem_type == "classification" else "r2"
        improvement = experiment_metrics.get(primary, 0) - previous_metrics.get(primary, 0)
        if improvement < 0.001:
            issues.append("Performance plateau - minimal improvement from previous iteration")
            recommended_actions.append("different_model")
    
    # Use LLM for reasoning if available
    if llm.is_available and (issues or recommended_actions):
        llm_analysis = _get_llm_critic_analysis(
            experiment_metrics, problem_type, issues, recommended_actions
        )
        if llm_analysis:
            issues.extend(llm_analysis.get("additional_issues", []))
            recommended_actions.extend(llm_analysis.get("additional_actions", []))
    
    # Deduplicate
    issues = list(dict.fromkeys(issues))
    recommended_actions = list(dict.fromkeys(recommended_actions))
    
    # Determine status
    if not issues:
        status = "satisfactory"
    elif len(issues) > 3:
        status = "needs_significant_improvement"
    else:
        status = "needs_improvement"
    
    # Select primary action
    primary_action = recommended_actions[0] if recommended_actions else "none"
    
    critic_feedback = {
        "status": status,
        "issues": issues,
        "recommended_actions": recommended_actions,
        "primary_action": primary_action,
        "summary": _generate_summary(status, issues, primary_action),
    }
    
    logger.info(f"[CriticAgent] Status: {status}, Issues: {len(issues)}, "
                f"Primary action: {primary_action}")
    
    return critic_feedback


def _analyze_classification(
    metrics: Dict[str, Any],
    target_dist: Dict[str, Any],
    prev_metrics: Dict[str, Any] = None,
    train_metrics: Dict[str, Any] = None
) -> Tuple[List[str], List[str]]:
    """Analyze classification metrics."""
    issues = []
    actions = []
    
    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1 = metrics.get("f1", 0)
    roc_auc = metrics.get("roc_auc")
    
    # Low recall
    if recall < 0.7:
        issues.append(f"Low recall ({recall:.3f}) - model missing positive cases")
        actions.extend(["class_weight", "threshold_adjustment", "resample"])
    
    # Low precision
    if precision < 0.7:
        issues.append(f"Low precision ({precision:.3f}) - too many false positives")
        actions.append("threshold_adjustment")
    
    # Low F1
    if f1 < 0.7:
        issues.append(f"Low F1 score ({f1:.3f}) - poor balance of precision/recall")
        actions.extend(["class_weight", "feature_engineering", "hyperparameter_tuning"])
    
    # Class imbalance
    if target_dist.get("is_imbalanced"):
        issues.append("Class imbalance detected - minority class underrepresented")
        if "class_weight" not in actions:
            actions.append("class_weight")
        actions.append("resample")
    
    # Overfitting check
    if train_metrics:
        train_f1 = train_metrics.get("f1", 0)
        if train_f1 - f1 > 0.1:
            issues.append(f"Overfitting detected (train F1: {train_f1:.3f}, test F1: {f1:.3f})")
            actions.append("hyperparameter_tuning")
    
    # Low ROC-AUC
    if roc_auc is not None and roc_auc < 0.7:
        issues.append(f"Low ROC-AUC ({roc_auc:.3f}) - poor discriminative ability")
        actions.extend(["feature_engineering", "different_model"])
    
    # Accuracy vs F1 mismatch (imbalance indicator)
    if accuracy > 0.9 and f1 < 0.7:
        issues.append("High accuracy but low F1 - likely class imbalance masking poor performance")
        actions.append("class_weight")
    
    return issues, actions


def _analyze_regression(
    metrics: Dict[str, Any],
    prev_metrics: Dict[str, Any] = None,
    train_metrics: Dict[str, Any] = None
) -> Tuple[List[str], List[str]]:
    """Analyze regression metrics."""
    issues = []
    actions = []
    
    r2 = metrics.get("r2", 0)
    rmse = metrics.get("rmse", 0)
    mae = metrics.get("mae", 0)
    
    # Low R2
    if r2 < 0.5:
        issues.append(f"Low R² ({r2:.3f}) - model explains little variance")
        actions.extend(["feature_engineering", "different_model", "hyperparameter_tuning"])
    elif r2 < 0.7:
        issues.append(f"Moderate R² ({r2:.3f}) - room for improvement")
        actions.extend(["hyperparameter_tuning", "feature_engineering"])
    
    # Overfitting check
    if train_metrics:
        train_r2 = train_metrics.get("r2", 0)
        if train_r2 - r2 > 0.1:
            issues.append(f"Overfitting detected (train R²: {train_r2:.3f}, test R²: {r2:.3f})")
            actions.append("hyperparameter_tuning")
    
    # High residuals
    residuals = metrics.get("residuals", {})
    if residuals.get("std", 0) > residuals.get("mean", 0) * 2:
        issues.append("High residual variance - predictions inconsistent")
        actions.append("feature_engineering")
    
    return issues, actions


def _get_llm_critic_analysis(
    metrics: Dict[str, Any],
    problem_type: str,
    issues: List[str],
    actions: List[str]
) -> Optional[Dict[str, Any]]:
    """Get additional analysis from LLM."""
    llm = get_llm_service()
    
    prompt = f"""Analyze these ML model metrics and provide additional insights.

Problem Type: {problem_type}
Metrics: {metrics}
Current Issues: {issues}
Current Recommended Actions: {actions}

Provide brief JSON response:
{{
  "additional_issues": ["issue1", "issue2"],
  "additional_actions": ["action1", "action2"]
}}"""
    
    try:
        response = llm.invoke(prompt, "You are an expert ML critic. Output only valid JSON.")
        import json
        return json.loads(response)
    except Exception:
        return None


def _generate_summary(status: str, issues: List[str], primary_action: str) -> str:
    """Generate human-readable summary."""
    if status == "satisfactory":
        return "Model performance is satisfactory. No critical issues detected."
    
    summary = f"Model needs improvement. Key issues: {'; '.join(issues[:3])}. "
    summary += f"Recommended: {primary_action}."
    return summary


def should_continue(
    improvement,
    iteration: int,
    max_iterations: int,
    min_threshold: float = 0.005
) -> Tuple[bool, str]:
    """Determine if autonomous loop should continue."""
    if iteration >= max_iterations:
        return False, f"Maximum iterations ({max_iterations}) reached"
    
    if improvement is not None and improvement < min_threshold:
        return False, f"Improvement ({improvement:.4f}) below threshold ({min_threshold})"
    
    return True, ""


def create_critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for critic analysis."""
    best_experiment = state.get("best_experiment", {})
    best_metrics = best_experiment.get("metrics", {})
    problem_type = state.get("problem_type", "classification")
    dataset_profile = state.get("dataset_profile", {})
    target_analysis = dataset_profile.get("target_analysis", {})
    iteration = state.get("iteration_number", 0)
    max_iterations = state.get("max_iterations", 5)
    min_improvement = state.get("min_improvement_threshold", 0.005)
    experiments = state.get("experiments", [])
    
    # Get previous best metrics (before current iteration)
    prev_metrics = None
    if len(experiments) > 1:
        # Find best from previous iterations
        prev_exps = [e for e in experiments if e.get("iteration", 0) < iteration]
        if prev_exps:
            prev_best = max(prev_exps, key=lambda e: e.get("metrics", {}).get("f1" if problem_type == "classification" else "r2", 0))
            prev_metrics = prev_best.get("metrics", {})
    
    # Get train metrics if available (from cross-validation)
    train_metrics = best_experiment.get("cv_metrics")
    
    # Analyze
    critic_feedback = analyze_experiment(
        best_metrics, problem_type, target_analysis, prev_metrics, train_metrics
    )
    
    # Check stopping conditions
    improvement = None
    if prev_metrics:
        primary = "f1" if problem_type == "classification" else "r2"
        improvement = best_metrics.get(primary, 0) - prev_metrics.get(primary, 0)
    
    should_continue_loop, stopping_reason = should_continue(
        improvement, iteration + 1, max_iterations, min_improvement
    )
    
    # Override if critic says satisfactory
    if critic_feedback["status"] == "satisfactory":
        should_continue_loop = False
        stopping_reason = "Critic: performance satisfactory"
    
    logger.info(f"[CriticAgent] Iteration {iteration+1}/{max_iterations}, "
                f"continue: {should_continue_loop}, reason: {stopping_reason}")
    
    return {
        "critic_feedback": critic_feedback,
        "should_continue": should_continue_loop,
        "stopping_reason": stopping_reason,
        "improvement": improvement,
        "current_step": "critic",
    }