# Evaluation Tools - Comprehensive model evaluation
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted"
) -> Dict[str, Any]:
    """Evaluate classification model."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    
    # Per-class metrics
    per_class = {}
    classes = np.unique(y_true)
    for cls in classes:
        cls_mask = y_true == cls
        per_class[str(cls)] = {
            "precision": float(precision_score(cls_mask, y_pred == cls, zero_division=0)),
            "recall": float(recall_score(cls_mask, y_pred == cls, zero_division=0)),
            "f1": float(f1_score(cls_mask, y_pred == cls, zero_division=0)),
            "support": int(cls_mask.sum())
        }
    metrics["per_class"] = per_class
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = classes.tolist()
    
    # ROC-AUC if probabilities available
    if y_proba is not None:
        try:
            if len(classes) == 2:
                # Binary classification
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                # Multi-class
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average=average))
        except:
            metrics["roc_auc"] = None
    
    # Classification report as dict
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
    except:
        pass
    
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """Evaluate regression model."""
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    
    # Additional metrics
    residuals = y_true - y_pred
    metrics["residuals"] = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "max": float(np.max(np.abs(residuals))),
    }
    
    # MAPE (if no zeros in y_true)
    if np.all(y_true != 0):
        metrics["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    
    return metrics


def evaluate_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    problem_type: str,
    return_proba: bool = False
) -> Dict[str, Any]:
    """Evaluate model on data."""
    y_pred = model.predict(X)
    
    y_proba = None
    if return_proba and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    
    if problem_type == "classification":
        return evaluate_classification(y_true, y_pred, y_proba)
    else:
        return evaluate_regression(y_true, y_pred)


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str
) -> pd.DataFrame:
    """Compare multiple models."""
    results = []
    
    for name, model in models.items():
        if model is None:
            continue
        
        try:
            metrics = evaluate_model(model, X_test, y_test, problem_type, return_proba=True)
            result = {"model": name}
            result.update(metrics)
            results.append(result)
        except Exception as e:
            results.append({"model": name, "error": str(e)})
    
    df = pd.DataFrame(results)
    
    # Sort by primary metric
    if problem_type == "classification" and "f1" in df.columns:
        df = df.sort_values("f1", ascending=False)
    elif problem_type == "regression" and "r2" in df.columns:
        df = df.sort_values("r2", ascending=False)
    
    return df


def get_primary_metric(metrics: Dict[str, Any], problem_type: str, class_imbalanced: bool = False) -> Tuple[str, float]:
    """Get primary metric for model selection."""
    if problem_type == "classification":
        if class_imbalanced:
            # For imbalanced, prefer F1 or balanced accuracy
            return ("f1", metrics.get("f1", 0.0))
        else:
            return ("accuracy", metrics.get("accuracy", 0.0))
    else:
        # For regression, prefer R2
        return ("r2", metrics.get("r2", 0.0))


def calculate_improvement(baseline_metrics: Dict[str, Any], new_metrics: Dict[str, Any], primary_metric: str) -> float:
    """Calculate improvement between two metric sets."""
    baseline = baseline_metrics.get(primary_metric, 0.0)
    new = new_metrics.get(primary_metric, 0.0)
    
    if baseline == 0:
        return 0.0
    
    return (new - baseline) / abs(baseline)


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    cv: int = 5,
    scoring: str = None,
    return_train_score: bool = False
) -> Dict[str, Any]:
    """Cross-validate a model."""
    from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
    
    if scoring is None:
        if problem_type == "classification":
            scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc_ovr_weighted"]
        else:
            scoring = ["neg_mean_absolute_error", "neg_mean_squared_error", "r2"]
    
    if problem_type == "classification":
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=return_train_score,
        n_jobs=-1
    )
    
    results = {}
    for key, values in cv_results.items():
        if key.startswith("test_") or key.startswith("train_"):
            metric_name = key.replace("test_", "").replace("train_", "")
            results[f"{key}_mean"] = float(np.mean(values))
            results[f"{key}_std"] = float(np.std(values))
    
    return results


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None):
    """Generate confusion matrix plot data."""
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = np.unique(y_true).tolist()
    
    return {
        "matrix": cm.tolist(),
        "labels": labels,
        "normalized": (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
    }


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray):
    """Generate ROC curve data."""
    if len(np.unique(y_true)) == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(roc_auc)
        }
    return None


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray):
    """Generate Precision-Recall curve data."""
    if len(np.unique(y_true)) == 2:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba[:, 1])
        pr_auc = auc(recall, precision)
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(pr_auc)
        }
    return None