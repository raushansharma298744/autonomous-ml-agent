# Feature Engineering Agent - Conservative feature engineering with validation
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from app.tools.feature_tools import (
    FeatureEngineer, extract_datetime_features, create_ratio_features,
    drop_redundant_features, encode_categorical_features
)
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


def engineer_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    eda_results: Dict[str, Any],
    problem_type: str,
    critic_feedback: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Apply conservative feature engineering based on EDA and critic feedback."""
    logger.info(f"[FeatureAgent] Starting feature engineering with {len(feature_names)} features")
    
    # Convert back to DataFrame for feature engineering
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Determine feature engineering strategy
    strategy = determine_fe_strategy(eda_results, critic_feedback, problem_type)
    
    # Initialize feature engineer with strategy
    fe = FeatureEngineer(
        add_datetime_features=strategy.get("add_datetime", False),
        add_interaction_features=strategy.get("add_interactions", False),
        add_polynomial_features=strategy.get("add_polynomial", False),
        log_transform_skewed=strategy.get("log_transform", True),
        skew_threshold=strategy.get("skew_threshold", 1.0),
        drop_high_cardinality=strategy.get("drop_high_cardinality", True),
        cardinality_threshold=strategy.get("cardinality_threshold", 50),
    )
    
    # Fit on training data only
    fe.fit(X_train_df)
    
    # Transform both train and test
    X_train_fe = fe.transform(X_train_df)
    X_test_fe = fe.transform(X_test_df)
    
    # Get new feature names
    new_feature_names = X_train_fe.columns.tolist()
    created_features = fe.created_features_
    
    # Drop redundant features (high correlation)
    X_train_fe, dropped = drop_redundant_features(X_train_fe, threshold=0.95)
    X_test_fe = X_test_fe[X_train_fe.columns]
    
    if dropped:
        logger.info(f"[FeatureAgent] Dropped {len(dropped)} redundant features: {dropped}")
    
    # Final feature names
    final_feature_names = X_train_fe.columns.tolist()
    
    # Convert back to numpy
    X_train_final = X_train_fe.values
    X_test_final = X_test_fe.values
    
    # Feature engineering summary
    fe_summary = {
        "original_features": len(feature_names),
        "created_features": created_features,
        "dropped_features": dropped,
        "final_features": len(final_feature_names),
        "strategy": strategy,
    }
    
    logger.info(f"[FeatureAgent] Complete: {fe_summary['original_features']} → "
                f"{fe_summary['final_features']} features "
                f"(created: {len(created_features)}, dropped: {len(dropped)})")
    
    return {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "feature_names": final_feature_names,
        "feature_engineering_summary": fe_summary,
        "feature_actions": [
            f"Log-transformed {len([c for c in created_features if '_log' in c])} skewed features",
            f"Created {len([c for c in created_features if '_x_' in c or '_div_' in c])} interaction features",
            f"Extracted {len([c for c in created_features if any(dt in c for dt in ['_year', '_month', '_day'])])} datetime features",
            f"Dropped {len(dropped)} redundant features",
        ],
    }


def determine_fe_strategy(
    eda_results: Dict[str, Any],
    critic_feedback: Dict[str, Any],
    problem_type: str
) -> Dict[str, Any]:
    """Determine feature engineering strategy based on EDA and critic feedback."""
    strategy = {
        "add_datetime": False,
        "add_interactions": False,
        "add_polynomial": False,
        "log_transform": True,
        "skew_threshold": 1.0,
        "drop_high_cardinality": True,
        "cardinality_threshold": 50,
    }
    
    # Check for datetime features in EDA
    feature_types = eda_results.get("feature_types", {})
    # Note: datetime detection would need original dataframe
    
    # Check critic feedback for feature engineering recommendation
    if critic_feedback:
        recommended = critic_feedback.get("recommended_actions", [])
        if "feature_engineering" in recommended:
            strategy["add_interactions"] = True
            strategy["add_polynomial"] = True
    
    # For large feature sets, be conservative
    n_features = len(feature_types.get("numerical", [])) + len(feature_types.get("categorical", []))
    if n_features > 50:
        strategy["add_interactions"] = False
        strategy["add_polynomial"] = False
    
    # For regression, polynomial features can help
    if problem_type == "regression":
        strategy["add_polynomial"] = True
    
    return strategy


def create_feature_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for feature engineering."""
    X_train = state.get("X_train")
    X_test = state.get("X_test")
    feature_names = state.get("feature_names", [])
    eda_results = state.get("eda_results", {})
    problem_type = state.get("problem_type", "classification")
    critic_feedback = state.get("critic_feedback", {})
    
    if X_train is None or X_test is None:
        logger.warning("[FeatureAgent] No train/test data, skipping feature engineering")
        return {
            "feature_names": feature_names,
            "feature_engineering_summary": {"original_features": len(feature_names), "final_features": len(feature_names)},
            "feature_actions": ["Skipped - no data"],
            "current_step": "feature_engineering",
        }
    
    result = engineer_features(X_train, X_test, feature_names, eda_results, problem_type, critic_feedback)
    
    return {
        "X_train": result["X_train"],
        "X_test": result["X_test"],
        "feature_names": result["feature_names"],
        "feature_engineering_summary": result["feature_engineering_summary"],
        "feature_actions": result["feature_actions"],
        "current_step": "feature_engineering",
    }