# Planner Agent - Creates ML workflow plan based on dataset analysis
import logging
import json
from typing import Dict, Any, List
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# Allowed node names in the LangGraph workflow (used for validation)
ALLOWED_STEPS = {
    "preprocessing",
    "eda",
    "feature_engineering",
    "model_selection",
    "tuning",
    "critic",
    "improvement_router",
    "apply_class_weight",
    "apply_resampling",
    "apply_smote",
    "adjust_threshold",
    "engineer_features",
    "select_different_model",
    "run_hyperparameter_tuning",
    "generate_final_report",
}


def create_workflow_plan(dataset_profile: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
    """Create ML workflow plan from dataset profile."""
    llm = get_llm_service()

    # Build structured prompt for LLM
    prompt = f"""Given this dataset profile, create a concise ML workflow plan.

Dataset Profile:
- Rows: {dataset_profile.get('n_rows', 'unknown')}
- Columns: {dataset_profile.get('n_columns', 'unknown')}
- Problem Type: {problem_type}
- Numerical Features: {len(dataset_profile.get('numerical_columns', []))}
- Categorical Features: {len(dataset_profile.get('categorical_columns', []))}
- Missing Values: {sum(v for v in dataset_profile.get('missing_values', {}).values() if v > 0)} columns affected
- Class Imbalance: {dataset_profile.get('class_imbalance', False)}
- ID Columns to Drop: {dataset_profile.get('id_columns', [])}
- High Cardinality Columns: {dataset_profile.get('high_cardinality_columns', [])}
- Target Analysis: {dataset_profile.get('target_analysis', {})}

Allowed workflow step names (choose subset, order matters):
{sorted(ALLOWED_STEPS)}

Return a JSON object with:
{{
  "workflow_steps": ["step1", "step2", ...],
  "reasoning": "brief explanation",
  "priority_actions": ["action1", "action2"],
  "estimated_iterations": 3
}}"""

    system_prompt = """You are an expert ML engineer planning an autonomous workflow.
Output only valid JSON. Keep reasoning concise (max 2 sentences)."""

    try:
        response = llm.invoke(prompt, system_prompt)
        plan = json.loads(response)
        # Validate workflow_steps against allowed set
        steps = plan.get("workflow_steps", [])
        if not isinstance(steps, list):
            raise ValueError("workflow_steps must be a list")
        # Filter out unknown steps
        valid_steps = [s for s in steps if s in ALLOWED_STEPS]
        if len(valid_steps) != len(steps):
            logger.warning(f"Planner LLM returned unknown steps, filtering: {set(steps) - ALLOWED_STEPS}")
        plan["workflow_steps"] = valid_steps
    except Exception as e:
        logger.warning(f"LLM planning failed, using fallback: {e}")
        plan = _fallback_plan(dataset_profile, problem_type)

    # Validate and ensure required fields
    plan.setdefault("workflow_steps", _default_steps(problem_type))
    plan.setdefault("reasoning", "Standard ML workflow based on dataset characteristics")
    plan.setdefault("priority_actions", [])
    plan.setdefault("estimated_iterations", 3)

    logger.info(f"[PlannerAgent] Plan created: {len(plan['workflow_steps'])} steps, "
                f"est. iterations: {plan['estimated_iterations']}")

    return plan


def _fallback_plan(dataset_profile: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
    """Deterministic fallback plan."""
    steps = _default_steps(problem_type)
    priority = []
    
    if dataset_profile.get("missing_values"):
        priority.append("handle_missing_values")
    if dataset_profile.get("class_imbalance"):
        priority.append("handle_class_imbalance")
    if dataset_profile.get("id_columns"):
        priority.append("drop_id_columns")
    if dataset_profile.get("high_cardinality_columns"):
        priority.append("handle_high_cardinality")
    
    return {
        "workflow_steps": steps,
        "reasoning": f"Standard {problem_type} workflow with {len(priority)} priority preprocessing actions",
        "priority_actions": priority,
        "estimated_iterations": 3
    }


def _default_steps(problem_type: str) -> List[str]:
    """Default workflow steps matching LangGraph node names."""
    base = [
        "preprocessing",
        "eda",
        "feature_engineering",
        "model_selection",
        "tuning",
        "critic"
    ]
    if problem_type == "classification":
        base.insert(2, "handle_class_imbalance")
    return base


def select_models(problem_type: str, dataset_size: int, feature_types: Dict[str, Any]) -> List[str]:
    """Select appropriate models based on data characteristics."""
    from app.tools.model_tools import get_classification_models, get_regression_models, recommend_best_algorithm
    
    # Build a minimal profile for recommender
    profile = {
        "n_rows": dataset_size,
        "n_columns": feature_types.get("numerical", 0) + feature_types.get("categorical", 0),
        "numerical_columns": ["num"] * feature_types.get("numerical", 0),
        "categorical_columns": ["cat"] * feature_types.get("categorical", 0),
        "class_imbalance": False,
        "high_cardinality_columns": [],
    }
    # Get recommender ordering
    recommended = recommend_best_algorithm(profile, problem_type)
    
    if problem_type == "classification":
        models = get_classification_models(dataset_size)
    else:
        models = get_regression_models(dataset_size)
    
    # Filter based on dataset size
    suitable = [name for name, config in models.items() 
                if config.get("suitable_for_large", True) or dataset_size < 10000]
    
    # Prioritize recommended models that are suitable
    prioritized = [m for m in recommended if m in suitable]
    # Add remaining suitable models
    for m in suitable:
        if m not in prioritized:
            prioritized.append(m)
    
    # Always include at least 3 models
    if len(prioritized) < 3:
        for m in models.keys():
            if m not in prioritized:
                prioritized.append(m)
            if len(prioritized) >= 3:
                break
    
    return prioritized[:5]  # Max 5 models


def create_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for workflow planning."""
    dataset_profile = state.get("dataset_profile", {})
    problem_type = state.get("problem_type", "classification")
    
    plan = create_workflow_plan(dataset_profile, problem_type)
    candidate_models = select_models(
        problem_type, 
        dataset_profile.get("n_rows", 1000),
        {
            "numerical": len(dataset_profile.get("numerical_columns", [])),
            "categorical": len(dataset_profile.get("categorical_columns", []))
        }
    )
    
    return {
        "workflow_plan": plan,
        "candidate_models": candidate_models,
        "planner_summary": plan.get("reasoning", ""),
        "current_step": "planner",
        "iteration_number": 0,
    }