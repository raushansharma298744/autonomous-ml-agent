# MLflow Service - Experiment tracking with MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from typing import Dict, Any, Optional, List
import os
from app.config import settings


def init_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    # Set artifact location
    os.makedirs(settings.MLRUNS_DIR, exist_ok=True)


def start_run(run_name: str = None, experiment_name: str = "autonomous_ml") -> str:
    """Start MLflow run."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=settings.MLRUNS_DIR
        )
    else:
        experiment_id = experiment.experiment_id
    
    run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    return run.info.run_id


def end_run():
    """End MLflow run."""
    mlflow.end_run()


def log_params(params: Dict[str, Any]):
    """Log parameters."""
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: int = None):
    """Log metrics."""
    mlflow.log_metrics(metrics, step=step)


def log_model(model, artifact_path: str = "model", registered_name: str = None):
    """Log model artifact."""
    mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_name)


def log_artifact(local_path: str, artifact_path: str = None):
    """Log artifact file."""
    mlflow.log_artifact(local_path, artifact_path)


def log_figure(fig, artifact_path: str):
    """Log matplotlib figure."""
    mlflow.log_figure(fig, artifact_path)


def log_dict(data: Dict[str, Any], artifact_path: str):
    """Log dictionary as JSON artifact."""
    import json
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f, indent=2, default=str)
        temp_path = f.name
    
    try:
        mlflow.log_artifact(temp_path, artifact_path)
    finally:
        os.unlink(temp_path)


def get_run(run_id: str):
    """Get MLflow run."""
    return mlflow.get_run(run_id)


def search_runs(experiment_name: str = "autonomous_ml", filter_string: str = "") -> List:
    """Search runs."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    
    return mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_string)


def get_best_run(experiment_name: str = "autonomous_ml", metric: str = "f1", ascending: bool = False):
    """Get best run by metric."""
    runs = search_runs(experiment_name)
    if runs.empty:
        return None
    
    metric_col = f"metrics.{metric}"
    if metric_col not in runs.columns:
        return None
    
    runs_sorted = runs.sort_values(metric_col, ascending=ascending)
    best_run_id = runs_sorted.iloc[0]["run_id"]
    
    return get_run(best_run_id)