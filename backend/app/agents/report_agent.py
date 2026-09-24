# Final Report Agent - Generates comprehensive reports
import logging
from typing import Dict, Any, List, Optional
from app.services.report_service import generate_final_report, save_report, generate_html_report, save_html_report

logger = logging.getLogger(__name__)


def create_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for final report generation."""
    logger.info("[ReportAgent] Generating final report")
    
    # Prepare job data
    job_data = {
        "id": state.get("job_id"),
        "dataset_id": state.get("dataset_id"),
        "target_column": state.get("target_column"),
        "problem_type": state.get("problem_type"),
        "preprocessing_steps": state.get("preprocessing_actions", []),
        "feature_engineering_steps": state.get("feature_actions", []),
        "stopping_reason": state.get("stopping_reason", "Max iterations reached"),
    }
    
    dataset_profile = state.get("dataset_profile", {})
    if dataset_profile:
        dataset_profile["filename"] = state.get("dataset_path", "unknown").split("/")[-1]
    
    experiments = state.get("experiments", [])
    best_experiment = state.get("best_experiment", {})
    critic_feedback = state.get("critic_feedback", {})
    improvements = state.get("improvement_history", [])
    
    # Generate report
    output_dir = "./reports"
    report = generate_final_report(
        job_data=job_data,
        dataset_profile=dataset_profile,
        experiments=experiments,
        best_experiment=best_experiment,
        critic_feedback=[critic_feedback] if critic_feedback else [],
        improvements=improvements,
        output_dir=output_dir,
    )
    
    # Save reports
    report_path = f"{output_dir}/report_{state.get('job_id')}.json"
    html_path = f"{output_dir}/report_{state.get('job_id')}.html"
    
    save_report(report, report_path)
    html_content = generate_html_report(report)
    save_html_report(report, html_path)
    
    logger.info(f"[ReportAgent] Report saved to {report_path} and {html_path}")
    
    return {
        "final_report": report,
        "report_path": report_path,
        "html_path": html_path,
        "current_step": "final_report",
    }