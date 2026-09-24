import pytest
import pandas as pd
import numpy as np
from app.tools.dataset_tools import (
    profile_dataset,
    analyze_target,
    check_class_imbalance,
    detect_outliers,
    split_dataset
)


def test_profile_dataset():
    """Test dataset profiling."""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [10, 20, 30, 40, 50],
        'cat1': ['a', 'b', 'a', 'b', 'a'],
        'target': [0, 1, 0, 1, 0]
    })
    
    profile = profile_dataset(df, 'target')
    
    assert profile['n_rows'] == 5
    assert profile['n_columns'] == 4
    assert 'num1' in profile['numerical_columns']
    assert 'cat1' in profile['categorical_columns']
    assert 'target_analysis' in profile
    assert profile['target_analysis']['problem_type'] == 'classification'


def test_analyze_target_classification():
    """Test target analysis for classification."""
    target = pd.Series([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])  # 7:3 ratio
    
    analysis = analyze_target(target)
    
    assert analysis['problem_type'] == 'classification'
    # With 3/10 = 0.3 > 0.1 threshold, not imbalanced by default
    assert analysis['is_imbalanced'] == False
    
    # Test with custom threshold
    assert check_class_imbalance(target, threshold=0.25) == True
    assert 'class_distribution' in analysis


def test_analyze_target_regression():
    """Test target analysis for regression."""
    target = pd.Series([1.5, 2.3, 3.1, 4.2, 5.0, 2.8, 3.9])
    
    analysis = analyze_target(target)
    
    assert analysis['problem_type'] == 'regression'
    assert 'target_stats' in analysis
    assert 'mean' in analysis['target_stats']


def test_check_class_imbalance():
    """Test class imbalance detection."""
    balanced = pd.Series([0, 1, 0, 1, 0, 1])
    imbalanced = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    assert check_class_imbalance(balanced) == False
    # Default threshold is 0.1, so 1/10 = 0.1 is NOT imbalanced
    # Need even more imbalance to trigger
    very_imbalanced = pd.Series([0] * 95 + [1] * 5)
    assert check_class_imbalance(very_imbalanced) == True
    
    # Test with custom threshold
    assert check_class_imbalance(imbalanced, threshold=0.2) == True


def test_detect_outliers():
    """Test outlier detection."""
    df = pd.DataFrame({
        'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'with_outlier': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    })
    
    outliers = detect_outliers(df)
    
    assert 'normal' in outliers
    assert 'with_outlier' in outliers
    assert outliers['with_outlier']['count'] > 0


def test_split_dataset():
    """Test dataset splitting."""
    df = pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100, 200),
        'target': [0, 1] * 50
    })
    
    X_train, X_test, y_train, y_test = split_dataset(df, 'target', test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20