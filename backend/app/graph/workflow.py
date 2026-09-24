# LangGraph Workflow - Autonomous ML Engineer Agent
import logging
from langgraph.graph import StateGraph, END
from app.graph.state import AgentState, create_initial_state
from app.agents.dataset_agent import create_dataset_node
from app.agents.planner_agent import create_planner_node
from app.agents.preprocessing_agent import create_preprocessing_node
from app.agents.eda_agent import create_eda_node
from app.agents.feature_agent import create_feature_node
from app.agents.model_agent import create_model_node
from app.agents.tuning_agent import create_tuning_node
from app.agents.critic_agent import create_critic_node
from app.agents.improvement_agent import (
    create_improvement_node,
    apply_class_weight_improvement,
    apply_resampling_improvement,
    apply_smote_improvement,
    adjust_threshold_improvement,
    engineer_features_improvement,
    select_different_model_improvement,
    run_hyperparameter_tuning_improvement,
)
from app.agents.report_agent import create_report_node
from app.graph.conditional_edges import (
    should_continue_improvement,
    select_next_action,
    evaluate_stopping_conditions,
)

logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow for autonomous ML."""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("dataset_analysis", create_dataset_node)
    workflow.add_node("planner", create_planner_node)
    workflow.add_node("preprocessing", create_preprocessing_node)
    workflow.add_node("eda", create_eda_node)
    workflow.add_node("feature_engineering", create_feature_node)
    workflow.add_node("model_selection", create_model_node)
    workflow.add_node("tuning", create_tuning_node)
    workflow.add_node("critic", create_critic_node)
    workflow.add_node("improvement_router", create_improvement_node)
    workflow.add_node("generate_final_report", create_report_node)
    
    # Improvement action nodes
    workflow.add_node("apply_class_weight", apply_class_weight_improvement)
    workflow.add_node("apply_resampling", apply_resampling_improvement)
    workflow.add_node("apply_smote", apply_smote_improvement)
    workflow.add_node("adjust_threshold", adjust_threshold_improvement)
    workflow.add_node("engineer_features", engineer_features_improvement)
    workflow.add_node("select_different_model", select_different_model_improvement)
    workflow.add_node("run_hyperparameter_tuning", run_hyperparameter_tuning_improvement)
    
    # Entry point
    workflow.set_entry_point("dataset_analysis")
    
    # Linear chain for initial setup
    workflow.add_edge("dataset_analysis", "planner")
    workflow.add_edge("planner", "preprocessing")
    workflow.add_edge("preprocessing", "eda")
    workflow.add_edge("eda", "feature_engineering")
    workflow.add_edge("feature_engineering", "model_selection")
    workflow.add_edge("model_selection", "tuning")
    workflow.add_edge("tuning", "critic")
    
    # Conditional edge from critic to improvement router or final report
    workflow.add_conditional_edges(
        "critic",
        should_continue_improvement,
        {
            "continue": "improvement_router",
            "stop": "generate_final_report",
        }
    )
    
    # Conditional edge from improvement router to specific improvement action
    workflow.add_conditional_edges(
        "improvement_router",
        select_next_action,
        {
            "apply_class_weight": "apply_class_weight",
            "apply_resampling": "apply_resampling",
            "apply_smote": "apply_smote",
            "adjust_threshold": "adjust_threshold",
            "engineer_features": "engineer_features",
            "select_different_model": "select_different_model",
            "run_hyperparameter_tuning": "run_hyperparameter_tuning",
            "stop": "generate_final_report",
        }
    )
    
    # All improvement actions loop back to model_selection for retraining
    improvement_actions = [
        "apply_class_weight",
        "apply_resampling", 
        "apply_smote",
        "adjust_threshold",
        "engineer_features",
        "select_different_model",
        "run_hyperparameter_tuning",
    ]
    
    for action in improvement_actions:
        workflow.add_edge(action, "model_selection")
    
    # Final report ends the workflow
    workflow.add_edge("generate_final_report", END)
    
    return workflow


def compile_workflow():
    """Compile the workflow for execution."""
    workflow = create_workflow()
    return workflow.compile()


# Export for API use
def create_initial_agent_state(
    job_id: str,
    dataset_path: str,
    dataset_id: str,
    target_column: str,
    problem_type: str = "auto",
    max_iterations: int = 5
) -> AgentState:
    """Create initial state for agent execution."""
    return create_initial_state(
        job_id=job_id,
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        target_column=target_column,
        problem_type=problem_type,
        max_iterations=max_iterations,
    )