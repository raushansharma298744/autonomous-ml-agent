# Report Service - Generate final ML reports
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import os

from app.config import PROJECT_TITLE


def generate_final_report(
    job_data: Dict[str, Any],
    dataset_profile: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    best_experiment: Dict[str, Any],
    critic_feedback: List[Dict[str, Any]],
    improvements: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """Generate comprehensive final report."""
    
    report = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "job_id": job_data.get("id"),
            "dataset_id": job_data.get("dataset_id"),
            "dataset_name": dataset_profile.get("filename", "unknown"),
            "target_column": job_data.get("target_column"),
            "problem_type": job_data.get("problem_type"),
        },
        "dataset_summary": {
            "rows": dataset_profile.get("n_rows"),
            "columns": dataset_profile.get("n_columns"),
            "numerical_features": len(dataset_profile.get("numerical_columns", [])),
            "categorical_features": len(dataset_profile.get("categorical_columns", [])),
            "missing_values": dataset_profile.get("missing_values", {}),
            "duplicate_rows": dataset_profile.get("duplicate_rows", 0),
            "target_analysis": dataset_profile.get("target_analysis", {}),
        },
        "preprocessing_performed": job_data.get("preprocessing_steps", []),
        "eda_findings": extract_eda_findings(dataset_profile),
        "features_used": best_experiment.get("feature_names", []),
        "feature_engineering": job_data.get("feature_engineering_steps", []),
        "models_tested": summarize_models(experiments),
        "hyperparameter_tuning": summarize_tuning(experiments),
        "experiment_comparison": compare_experiments(experiments),
        "best_experiment": {
            "id": best_experiment.get("id"),
            "model": best_experiment.get("model_name"),
            "metrics": best_experiment.get("metrics"),
            "hyperparameters": best_experiment.get("hyperparameters"),
            "training_duration": best_experiment.get("training_duration_seconds"),
        },
        "critic_findings": critic_feedback,
        "improvements_performed": improvements,
        "stopping_reason": job_data.get("stopping_reason", "Max iterations reached"),
        "limitations": generate_limitations(job_data, dataset_profile, best_experiment),
        "recommended_next_steps": generate_next_steps(job_data, best_experiment, critic_feedback),
    }
    
    return report


def extract_eda_findings(dataset_profile: Dict[str, Any]) -> List[str]:
    """Extract key EDA findings from profile."""
    findings = []
    
    # Target analysis
    target = dataset_profile.get("target_analysis", {})
    if target.get("problem_type") == "classification":
        dist = target.get("class_distribution", {})
        if dist:
            total = sum(dist.values())
            for cls, count in dist.items():
                pct = count / total * 100
                findings.append(f"Target class '{cls}': {count} samples ({pct:.1f}%)")
        
        if target.get("is_imbalanced"):
            findings.append("Class imbalance detected - minority class underrepresented")
    
    # Missing values
    missing = dataset_profile.get("missing_values", {})
    high_missing = {k: v for k, v in missing.items() if v > 0}
    if high_missing:
        findings.append(f"Missing values in {len(high_missing)} columns")
        for col, count in sorted(high_missing.items(), key=lambda x: -x[1])[:3]:
            pct = dataset_profile.get("missing_percentage", {}).get(col, 0)
            findings.append(f"  - {col}: {count} ({pct:.1f}%)")
    
    # Outliers
    outliers = dataset_profile.get("outliers", {})
    if outliers:
        total_outliers = sum(o.get("count", 0) for o in outliers.values())
        if total_outliers > 0:
            findings.append(f"Outliers detected in {len(outliers)} numerical columns (total: {total_outliers})")
    
    # Correlations
    numerical = dataset_profile.get("numerical_columns", [])
    if len(numerical) > 1:
        findings.append(f"Correlation analysis performed on {len(numerical)} numerical features")
    
    return findings


def summarize_models(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize all models tested."""
    summary = []
    seen_models = set()
    
    for exp in experiments:
        model_name = exp.get("model_name")
        if model_name not in seen_models:
            seen_models.add(model_name)
            summary.append({
                "model": model_name,
                "baseline_metrics": exp.get("metrics") if exp.get("iteration", 0) == 0 else None,
                "tuned_metrics": exp.get("metrics") if exp.get("iteration", 0) > 0 else None,
                "best_iteration": exp.get("iteration", 0),
            })
    
    return summary


def summarize_tuning(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize hyperparameter tuning."""
    tuning = []
    
    for exp in experiments:
        if exp.get("hyperparameters") and exp.get("iteration", 0) > 0:
            tuning.append({
                "model": exp.get("model_name"),
                "iteration": exp.get("iteration"),
                "best_params": exp.get("hyperparameters"),
                "best_score": exp.get("metrics", {}).get("f1") or exp.get("metrics", {}).get("r2"),
            })
    
    return tuning


def compare_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare all experiments."""
    comparison = []
    
    for exp in experiments:
        metrics = exp.get("metrics", {})
        comparison.append({
            "experiment_id": exp.get("id"),
            "model": exp.get("model_name"),
            "iteration": exp.get("iteration"),
            "parent_experiment": exp.get("parent_experiment_id"),
            "metrics": {
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "roc_auc": metrics.get("roc_auc"),
                "r2": metrics.get("r2"),
                "rmse": metrics.get("rmse"),
            },
            "duration": exp.get("training_duration_seconds"),
        })
    
    return comparison


def generate_limitations(
    job_data: Dict[str, Any],
    dataset_profile: Dict[str, Any],
    best_experiment: Dict[str, Any]
) -> List[str]:
    """Generate limitations section."""
    limitations = []
    
    # Dataset size
    n_rows = dataset_profile.get("n_rows", 0)
    if n_rows < 1000:
        limitations.append(f"Small dataset ({n_rows} rows) - results may not generalize")
    elif n_rows < 10000:
        limitations.append(f"Moderate dataset size ({n_rows} rows) - consider more data for production")
    
    # Missing data
    missing_pct = dataset_profile.get("missing_percentage", {})
    high_missing = [k for k, v in missing_pct.items() if v > 20]
    if high_missing:
        limitations.append(f"High missingness in columns: {', '.join(high_missing)} - imputation may introduce bias")
    
    # Single split
    limitations.append("Single train/test split used - cross-validation recommended for production deployment")
    
    # No external validation
    limitations.append("No external validation dataset used - performance on unseen data unknown")
    
    # Feature engineering
    if not job_data.get("feature_engineering_steps"):
        limitations.append("No feature engineering applied - potential for improvement")
    
    # Model complexity
    best_model = best_experiment.get("model_name", "")
    if "XGBoost" in best_model or "GradientBoosting" in best_model:
        limitations.append("Tree-based models may overfit on small datasets - regularization important")
    
    return limitations


def generate_next_steps(
    job_data: Dict[str, Any],
    best_experiment: Dict[str, Any],
    critic_feedback: List[Dict[str, Any]]
) -> List[str]:
    """Generate recommended next steps."""
    steps = []
    
    # Based on critic feedback
    for feedback in critic_feedback:
        if feedback.get("type") == "warning":
            action = feedback.get("recommended_action", "")
            if action == "class_weight":
                steps.append("Deploy with class-weighted predictions and monitor class-specific metrics")
            elif action == "threshold_adjustment":
                steps.append("Optimize decision threshold for business-specific cost matrix")
            elif action == "feature_engineering":
                steps.append("Investigate domain-specific feature engineering opportunities")
    
    # General steps
    steps.append("Implement model monitoring for prediction drift and data drift")
    steps.append("Set up automated retraining pipeline with scheduled frequency")
    steps.append("Add A/B testing framework for safe model updates")
    steps.append("Collect feedback loop data for continuous improvement")
    steps.append("Document model card with intended use cases and limitations")
    
    return steps


def save_report(report: Dict[str, Any], output_path: str):
    """Save report as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)


def generate_html_report(report: Dict[str, Any]) -> str:
    """Generate HTML report from structured data."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{PROJECT_TITLE} - Final Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px 20px; line-height: 1.6; color: #333; }}
        h1 {{ color: #1e3a8a; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }}
        h2 {{ color: #1e40af; margin-top: 40px; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px; }}
        h3 {{ color: #3730a3; }}
        .meta {{ background: #f8fafc; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .metric {{ display: inline-block; background: #eff6ff; padding: 10px 20px; margin: 5px; border-radius: 6px; font-weight: 600; color: #1e40af; }}
        .finding {{ background: #fef3c7; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid #f59e0b; }}
        .limitation {{ background: #fef2f2; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid #ef4444; }}
        .step {{ background: #f0fdf4; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid #22c55e; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f8fafc; font-weight: 600; }}
        .code {{ background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: 'Monaco', 'Consolas', monospace; }}
    </style>
</head>
<body>
    <h1>🤖 {PROJECT_TITLE} - Final Report</h1>
    
    <div class="meta">
        <strong>Job ID:</strong> {report['metadata']['job_id']}<br>
        <strong>Dataset:</strong> {report['metadata']['dataset_name']}<br>
        <strong>Target:</strong> {report['metadata']['target_column']}<br>
        <strong>Problem Type:</strong> {report['metadata']['problem_type']}<br>
        <strong>Generated:</strong> {report['metadata']['generated_at']}
    </div>
    
    <h2>📊 Dataset Summary</h2>
    <ul>
        <li><strong>Rows:</strong> {report['dataset_summary']['rows']:,}</li>
        <li><strong>Columns:</strong> {report['dataset_summary']['columns']}</li>
        <li><strong>Numerical Features:</strong> {report['dataset_summary']['numerical_features']}</li>
        <li><strong>Categorical Features:</strong> {report['dataset_summary']['categorical_features']}</li>
        <li><strong>Duplicate Rows:</strong> {report['dataset_summary']['duplicate_rows']}</li>
    </ul>
    
    <h2>🔧 Preprocessing Performed</h2>
    <ul>
        {''.join(f'<li>{step}</li>' for step in report['preprocessing_performed'])}
    </ul>
    
    <h2>📈 EDA Findings</h2>
    {''.join(f'<div class="finding">{finding}</div>' for finding in report['eda_findings'])}
    
    <h2>🎯 Features Used</h2>
    <ul>
        {''.join(f'<li>{feat}</li>' for feat in report['features_used'])}
    </ul>
    
    <h2>🏗️ Feature Engineering</h2>
    <ul>
        {''.join(f'<li>{step}</li>' for step in report['feature_engineering'])}
    </ul>
    
    <h2>🧪 Models Tested</h2>
    <table>
        <tr><th>Model</th><th>Baseline</th><th>Tuned</th><th>Best Iteration</th></tr>
        {''.join(f"<tr><td>{m['model']}</td><td>{m['baseline_metrics']}</td><td>{m['tuned_metrics']}</td><td>{m['best_iteration']}</td></tr>" for m in report['models_tested'])}
    </table>
    
    <h2>🏆 Best Experiment</h2>
    <div class="metric">Model: {report['best_experiment']['model']}</div>
    <div class="metric">F1 Score: {report['best_experiment']['metrics'].get('f1', 'N/A')}</div>
    <div class="metric">Accuracy: {report['best_experiment']['metrics'].get('accuracy', 'N/A')}</div>
    <div class="metric">Training Time: {f"{report['best_experiment']['training_duration']:.1f}" if report['best_experiment']['training_duration'] is not None else "N/A"}s</div>
    
    <h2>⚖️ Critic Findings</h2>
    {''.join(f'<div class="finding"><strong>{f.get("type", "info").upper()}:</strong> {f.get("message", "")}</div>' for f in report['critic_findings'])}
    
    <h2>🚀 Improvements Performed</h2>
    <ul>
        {''.join(f'<li>{imp}</li>' for imp in report['improvements_performed'])}
    </ul>
    
    <h2>🛑 Stopping Reason</h2>
    <p>{report['stopping_reason']}</p>
    
    <h2>⚠️ Limitations</h2>
    {''.join(f'<div class="limitation">{lim}</div>' for lim in report['limitations'])}
    
    <h2>🎯 Recommended Next Steps</h2>
    {''.join(f'<div class="step">{step}</div>' for step in report['recommended_next_steps'])}
    
</body>
</html>
"""
    return html


def save_html_report(report: Dict[str, Any], output_path: str):
    """Save HTML report."""
    html = generate_html_report(report)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)