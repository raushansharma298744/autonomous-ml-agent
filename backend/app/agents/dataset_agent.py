# Dataset Agent - Analyzes and profiles datasets
import logging
from typing import Dict, Any, Optional
from app.tools.dataset_tools import load_dataset, profile_dataset, analyze_target, check_class_imbalance

logger = logging.getLogger(__name__)


def analyze_dataset(dataset_path: str, target_column: str) -> Dict[str, Any]:
    """Analyze dataset and return structured profile."""
    logger.info(f"[DatasetAgent] Loading dataset from {dataset_path}")
    df = load_dataset(dataset_path)
    
    logger.info(f"[DatasetAgent] Dataset loaded: {len(df)} rows x {len(df.columns)} columns")
    
    profile = profile_dataset(df, target_column)
    
    # Add target-specific analysis
    if target_column in df.columns:
        target_analysis = analyze_target(df[target_column])
        profile["target_analysis"] = target_analysis
        profile["problem_type"] = target_analysis.get("problem_type", "classification")
        profile["class_imbalance"] = target_analysis.get("is_imbalanced", False)
    
    # Detect potential ID columns (high cardinality, unique)
    id_candidates = []
    for col in df.columns:
        if col != target_column:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95 and df[col].nunique() > 10:
                id_candidates.append(col)
    profile["id_columns"] = id_candidates
    
    # Detect high cardinality categorical columns
    cat_cols = profile.get("categorical_columns", [])
    high_cardinality = [col for col in cat_cols if df[col].nunique() > 50]
    profile["high_cardinality_columns"] = high_cardinality
    
    logger.info(f"[DatasetAgent] Profile complete: problem_type={profile.get('problem_type')}, "
                f"imbalance={profile.get('class_imbalance')}, "
                f"id_columns={id_candidates}")
    
    return profile


def detect_problem_type(target_series) -> str:
    """Detect if classification or regression."""
    from app.tools.dataset_tools import analyze_target
    analysis = analyze_target(target_series)
    return analysis.get("problem_type", "classification")


def calculate_profile(df, target_column: str) -> Dict[str, Any]:
    """Calculate comprehensive dataset statistics."""
    return profile_dataset(df, target_column)


def create_dataset_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for dataset analysis."""
    dataset_path = state["dataset_path"]
    target_column = state["target_column"]
    problem_type = state.get("problem_type", "auto")
    
    profile = analyze_dataset(dataset_path, target_column)
    
    # Override problem type if specified
    if problem_type != "auto":
        profile["problem_type"] = problem_type
    
    return {
        "dataset_profile": profile,
        "problem_type": profile["problem_type"],
        "current_step": "dataset_analysis",
        "iteration_number": 0,
    }