# Preprocessing Agent - Builds and applies preprocessing pipelines
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from app.tools.preprocessing_tools import (
    build_preprocessing_pipeline, fit_preprocessor, transform_data,
    get_feature_names, encode_target, apply_class_weight
)
from app.tools.dataset_tools import split_dataset

logger = logging.getLogger(__name__)


def build_preprocessing_plan(dataset_profile: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    """Build preprocessing configuration from profile and plan."""
    priority_actions = plan.get("priority_actions", [])
    
    preprocessing_config = {
        "drop_columns": dataset_profile.get("id_columns", []),
        "handle_missing": "handle_missing_values" in priority_actions or True,
        "encode_categorical": True,
        "scale_numerical": True,
        "handle_class_imbalance": "handle_class_imbalance" in priority_actions 
                                   or dataset_profile.get("class_imbalance", False),
        "handle_high_cardinality": "handle_high_cardinality" in priority_actions,
        "high_cardinality_threshold": 50,
    }
    
    return preprocessing_config


def apply_preprocessing(
    df: pd.DataFrame,
    target_column: str,
    preprocessing_config: Dict[str, Any],
    problem_type: str,
    random_state: int = 42
) -> Dict[str, Any]:
    """Apply preprocessing pipeline to dataset."""
    logger.info(f"[PreprocessingAgent] Starting preprocessing for {len(df)} rows")
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Drop ID columns
    drop_cols = preprocessing_config.get("drop_columns", [])
    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        logger.info(f"[PreprocessingAgent] Dropped ID columns: {drop_cols}")
    
    # Identify column types
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Handle high cardinality categorical
    if preprocessing_config.get("handle_high_cardinality"):
        high_card = [c for c in categorical_columns if X[c].nunique() > 
                     preprocessing_config.get("high_cardinality_threshold", 50)]
        if high_card:
            X = X.drop(columns=high_card)
            categorical_columns = [c for c in categorical_columns if c not in high_card]
            logger.info(f"[PreprocessingAgent] Dropped high cardinality: {high_card}")
    
    # Train/test split (before any fitting to avoid leakage)
    X_train, X_test, y_train, y_test = split_dataset(
        pd.concat([X, y], axis=1), target_column, 
        test_size=0.2, random_state=random_state,
        stratify=problem_type == "classification"
    )
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        X_train,
        numerical_columns=[c for c in numerical_columns if c in X_train.columns],
        categorical_columns=[c for c in categorical_columns if c in X_train.columns],
        scale_numerical=preprocessing_config.get("scale_numerical", True),
        encode_categorical="onehot"
    )
    
    # Fit on training data only
    preprocessor = fit_preprocessor(preprocessor, X_train)
    
    # Transform both train and test
    X_train_transformed = transform_data(preprocessor, X_train)
    X_test_transformed = transform_data(preprocessor, X_test)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor, X_train)
    
    # Encode target for classification
    y_train_encoded, target_encoder = encode_target(y_train, problem_type)
    y_test_encoded, _ = encode_target(y_test, problem_type)
    
    # Calculate class weights if needed
    class_weights = None
    if preprocessing_config.get("handle_class_imbalance") and problem_type == "classification":
        class_weights = apply_class_weight(y_train_encoded, "balanced")
        logger.info(f"[PreprocessingAgent] Class weights calculated: {class_weights}")
    
    # Preprocessing summary
    preprocessing_summary = {
        "original_features": len(df.columns) - 1,
        "dropped_columns": drop_cols,
        "numerical_features": len(numerical_columns),
        "categorical_features": len(categorical_columns),
        "final_features": len(feature_names),
        "class_weights": class_weights,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    
    logger.info(f"[PreprocessingAgent] Complete: {preprocessing_summary['final_features']} features, "
                f"train={preprocessing_summary['train_size']}, test={preprocessing_summary['test_size']}")
    
    return {
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train_encoded,
        "y_test": y_test_encoded,
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
        "feature_names": feature_names,
        "class_weights": class_weights,
        "preprocessing_summary": preprocessing_summary,
        "preprocessing_actions": [
            f"Dropped {len(drop_cols)} ID columns" if drop_cols else "No ID columns dropped",
            f"Encoded {len(categorical_columns)} categorical features",
            f"Scaled {len(numerical_columns)} numerical features",
            "Applied class weights" if class_weights else "No class weights",
        ],
    }


def create_preprocessing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for preprocessing."""
    dataset_path = state["dataset_path"]
    target_column = state["target_column"]
    problem_type = state.get("problem_type", "classification")
    dataset_profile = state.get("dataset_profile", {})
    workflow_plan = state.get("workflow_plan", {})
    
    # Load dataset
    import pandas as pd
    df = pd.read_csv(dataset_path)
    
    # Build preprocessing config
    preprocessing_config = build_preprocessing_plan(dataset_profile, workflow_plan)
    
    # Apply preprocessing
    result = apply_preprocessing(df, target_column, preprocessing_config, problem_type)
    
    return {
        "X_train": result["X_train"],
        "X_test": result["X_test"],
        "y_train": result["y_train"],
        "y_test": result["y_test"],
        "preprocessor": result["preprocessor"],
        "target_encoder": result["target_encoder"],
        "feature_names": result["feature_names"],
        "class_weights": result["class_weights"],
        "preprocessing_summary": result["preprocessing_summary"],
        "preprocessing_actions": result["preprocessing_actions"],
        "current_step": "preprocessing",
    }