import pytest
import numpy as np
import pandas as pd
from app.tools.preprocessing_tools import (
    build_preprocessing_pipeline,
    fit_preprocessor,
    transform_data,
    apply_class_weight
)
from app.tools.evaluation_tools import (
    evaluate_classification,
    evaluate_regression,
    get_primary_metric
)


def test_build_preprocessing_pipeline():
    """Test preprocessing pipeline creation."""
    numerical_cols = ['age', 'income']
    categorical_cols = ['gender', 'city']
    
    preprocessor = build_preprocessing_pipeline(
        X=pd.DataFrame(),
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols
    )
    
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2


def test_fit_and_transform():
    """Test fitting and transforming with preprocessor."""
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA']
    })
    
    preprocessor = build_preprocessing_pipeline(
        X=df,
        numerical_columns=['age', 'income'],
        categorical_columns=['gender', 'city']
    )
    
    fitted = fit_preprocessor(preprocessor, df)
    transformed = transform_data(fitted, df)
    
    assert transformed.shape[0] == 5
    assert transformed.shape[1] > 2  # numerical + one-hot encoded


def test_apply_class_weight():
    """Test class weight calculation."""
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 9:1 imbalance
    
    weights = apply_class_weight(y, method='balanced')
    
    assert 0 in weights
    assert 1 in weights
    assert weights[1] > weights[0]  # Minority class gets higher weight


def test_evaluate_classification():
    """Test classification evaluation."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    y_proba = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7],
        [0.7, 0.3], [0.6, 0.4], [0.9, 0.1], [0.4, 0.6]
    ])
    
    metrics = evaluate_classification(y_true, y_pred, y_proba)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 'confusion_matrix' in metrics
    # 7 correct out of 8 = 0.875
    assert abs(metrics['accuracy'] - 0.875) < 0.01


def test_evaluate_regression():
    """Test regression evaluation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    
    metrics = evaluate_regression(y_true, y_pred)
    
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert metrics['r2'] > 0.9  # Should be very high


def test_get_primary_metric():
    """Test primary metric selection."""
    clf_metrics = {'accuracy': 0.9, 'f1': 0.85, 'precision': 0.88}
    reg_metrics = {'r2': 0.9, 'rmse': 0.5, 'mae': 0.4}
    
    # Balanced classification
    name, value = get_primary_metric(clf_metrics, 'classification', class_imbalanced=False)
    assert name == 'accuracy'
    assert value == 0.9
    
    # Imbalanced classification
    name, value = get_primary_metric(clf_metrics, 'classification', class_imbalanced=True)
    assert name == 'f1'
    assert value == 0.85
    
    # Regression
    name, value = get_primary_metric(reg_metrics, 'regression')
    assert name == 'r2'
    assert value == 0.9