# LangGraph State Definition
from typing import TypedDict, Optional, Dict, Any, List
from datetime import datetime, timezone


class AgentState(TypedDict):
    # Dataset info
    dataset_path: str
    dataset_id: str
    dataset_profile: Optional[Dict[str, Any]]
    target_column: str
    problem_type: str  # classification, regression, auto
    
    # Planning
    workflow_plan: Optional[Dict[str, Any]]
    planner_summary: Optional[str]
    preprocessing_plan: Optional[Dict[str, Any]]
    feature_plan: Optional[Dict[str, Any]]
    candidate_models: Optional[List[str]]
    
    # Execution state
    current_step: str
    iteration_number: int
    max_iterations: int
    min_improvement_threshold: float
    
    # Data splits
    X_train: Optional[Any]
    X_test: Optional[Any]
    y_train: Optional[Any]
    y_test: Optional[Any]
    preprocessor: Optional[Any]
    target_encoder: Optional[Any]
    feature_names: Optional[List[str]]
    class_weights: Optional[Dict[int, float]]
    
    # Preprocessing
    preprocessing_summary: Optional[Dict[str, Any]]
    preprocessing_actions: Optional[List[str]]
    
    # EDA
    eda_results: Optional[Dict[str, Any]]
    eda_summary: Optional[str]
    
    # Feature Engineering
    feature_engineering_summary: Optional[Dict[str, Any]]
    feature_actions: Optional[List[str]]
    
    # Models & Experiments
    current_model: Optional[str]
    current_metrics: Optional[Dict[str, Any]]
    experiments: List[Dict[str, Any]]
    current_experiment: Optional[Dict[str, Any]]
    best_experiment: Optional[Dict[str, Any]]
    best_metrics: Optional[Dict[str, float]]
    tuning_results: Optional[Dict[str, Any]]
    
    # Critic feedback
    critic_feedback: Optional[Dict[str, Any]]
    improvement_plan: Optional[Dict[str, Any]]
    
    # Improvement loop
    next_action: Optional[str]
    improvement_history: List[Dict[str, Any]]
    improvement: float
    
    # Control
    should_continue: bool
    stopping_reason: Optional[str]
    error: Optional[str]
    
    # Final output
    final_report: Optional[Dict[str, Any]]
    
    # Metadata
    job_id: str
    created_at: datetime
    updated_at: datetime


# Initial state factory
def create_initial_state(
    job_id: str,
    dataset_path: str,
    dataset_id: str,
    target_column: str,
    problem_type: str = "auto",
    max_iterations: int = 5,
    min_improvement_threshold: float = 0.005
) -> AgentState:
    return AgentState(
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        dataset_profile=None,
        target_column=target_column,
        problem_type=problem_type,
        workflow_plan=None,
        planner_summary=None,
        preprocessing_plan=None,
        feature_plan=None,
        candidate_models=None,
        current_step="dataset_analysis",
        iteration_number=0,
        max_iterations=max_iterations,
        min_improvement_threshold=min_improvement_threshold,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        preprocessor=None,
        target_encoder=None,
        feature_names=None,
        class_weights=None,
        preprocessing_summary=None,
        preprocessing_actions=None,
        eda_results=None,
        eda_summary=None,
        feature_engineering_summary=None,
        feature_actions=None,
        current_model=None,
        current_metrics=None,
        experiments=[],
        current_experiment=None,
        best_experiment=None,
        best_metrics=None,
        tuning_results=None,
        critic_feedback=None,
        improvement_plan=None,
        next_action=None,
        improvement_history=[],
        improvement=0.0,
        should_continue=True,
        stopping_reason=None,
        error=None,
        final_report=None,
        job_id=job_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )