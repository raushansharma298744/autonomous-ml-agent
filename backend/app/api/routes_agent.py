from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import asyncio
import json
from typing import Dict, Any
from datetime import datetime

from app.database import AsyncSessionLocal
from app.models.job import Job
from app.models.dataset import Dataset
from app.graph.workflow import compile_workflow, create_initial_agent_state

router = APIRouter(prefix="", tags=["agent"])

active_connections: Dict[str, WebSocket] = {}

class ConnectionManager:
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        active_connections[client_id] = websocket

    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in active_connections and active_connections[client_id] == websocket:
            del active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in active_connections:
            await active_connections[client_id].send_text(message)

manager = ConnectionManager()

def format_time():
    return datetime.now().strftime("%H:%M:%S")

# Step mapping for frontend progress
STEP_ORDER = [
    "dataset_analysis",
    "planner",
    "preprocessing", 
    "eda",
    "feature_engineering",
    "model_selection",
    "tuning",
    "critic",
    "improvement_router",
    "apply_class_weight",
    "apply_resampling",
    "apply_smote",
    "adjust_threshold",
    "engineer_features",
    "select_different_model",
    "run_hyperparameter_tuning",
    "generate_final_report",
]

def get_step_index(node_name: str) -> int:
    """Get step index for progress tracking."""
    try:
        return STEP_ORDER.index(node_name)
    except ValueError:
        return 0

async def execute_agent_workflow(job_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return
            
        result = await session.execute(select(Dataset).where(Dataset.id == job.dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            return
            
        dataset_path = dataset.path
        target_column = job.target_column
        dataset_id = dataset.id
        problem_type = job.problem_type
        max_iterations = job.max_iterations

    workflow = compile_workflow()
    initial_state = create_initial_agent_state(
        job_id=job_id,
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        target_column=target_column,
        problem_type=problem_type,
        max_iterations=max_iterations,
    )

    await asyncio.sleep(1)
    
    await manager.send_message(json.dumps({
        "type": "log",
        "log": {"time": format_time(), "agent": "System", "message": f"Starting autonomous ML workflow for {dataset.name}", "level": "info"}
    }), job_id)

    # Update job status to running
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        if job:
            job.status = "running"
            job.started_at = datetime.utcnow()
            await session.commit()

    try:
        iteration = 0
        last_iteration = 0
        
        current_state = initial_state.copy()
        
        for output in workflow.stream(initial_state):
            for node_name, state in output.items():
                current_state.update(state)
                step_idx = get_step_index(node_name)
                total_steps = len([s for s in STEP_ORDER if s not in [
                    "apply_class_weight", "apply_resampling", "apply_smote",
                    "adjust_threshold", "engineer_features", "select_different_model",
                    "run_hyperparameter_tuning"
                ]])
                
                # Send step update
                await manager.send_message(json.dumps({
                    "type": "step_update",
                    "currentStep": step_idx + 1,
                    "totalSteps": total_steps,
                    "currentNode": node_name,
                }), job_id)
                
                # Send log
                agent_name = node_name.replace("_", " ").title()
                await manager.send_message(json.dumps({
                    "type": "log",
                    "log": {"time": format_time(), "agent": agent_name, "message": f"Running {agent_name}", "level": "info"}
                }), job_id)
                
                # Node-specific updates
                if node_name == "dataset_analysis":
                    prof = current_state.get("dataset_profile", {})
                    await manager.send_message(json.dumps({
                        "type": "log",
                        "log": {"time": format_time(), "agent": "Dataset Analyst", "message": f"Dataset: {prof.get('n_rows', 0)} rows, {prof.get('n_columns', 0)} cols, Problem: {current_state.get('problem_type')}", "level": "info"}
                    }), job_id)
                
                elif node_name == "planner":
                    plan = current_state.get("workflow_plan", {})
                    steps = plan.get("workflow_steps", [])
                    await manager.send_message(json.dumps({
                        "type": "log",
                        "log": {"time": format_time(), "agent": "Planner", "message": f"Workflow planned: {len(steps)} steps - {plan.get('reasoning', '')}", "level": "info"}
                    }), job_id)
                
                elif node_name == "preprocessing":
                    prep = current_state.get("preprocessing_summary", {})
                    await manager.send_message(json.dumps({
                        "type": "log",
                        "log": {"time": format_time(), "agent": "Preprocessing", "message": f"Features: {prep.get('original_features', 0)} → {prep.get('final_features', 0)}", "level": "success"}
                    }), job_id)
                
                elif node_name == "eda":
                    eda = current_state.get("eda_results", {})
                    insights = eda.get("insights", [])
                    if insights:
                        await manager.send_message(json.dumps({
                            "type": "log",
                            "log": {"time": format_time(), "agent": "EDA", "message": insights[0], "level": "info"}
                        }), job_id)
                
                elif node_name == "feature_engineering":
                    fe = current_state.get("feature_engineering_summary", {})
                    await manager.send_message(json.dumps({
                        "type": "log",
                        "log": {"time": format_time(), "agent": "Feature Engineering", "message": f"Features: {fe.get('original_features', 0)} → {fe.get('final_features', 0)}", "level": "success"}
                    }), job_id)
                
                elif node_name == "model_selection":
                    best_exp = current_state.get("best_experiment", {})
                    metrics = best_exp.get("metrics", {})
                    model_name = best_exp.get("model_name", "Unknown")
                    primary = "f1" if current_state.get("problem_type") == "classification" else "r2"
                    score = metrics.get(primary)
                    safe_score = f"{score:.4f}" if score is not None else "0.0000"
                    
                    await manager.send_message(json.dumps({
                        "type": "log",
                        "log": {"time": format_time(), "agent": "Model Selection", "message": f"Best: {model_name} ({primary.upper()}={safe_score})", "level": "success"}
                    }), job_id)
                    
                    # Send stats update
                    await send_stats_update(current_state, job_id)
                
                elif node_name == "tuning":
                    tuning = current_state.get("tuning_results", {})
                    if tuning:
                        cv_score = tuning.get("best_cv_score", 0)
                        await manager.send_message(json.dumps({
                            "type": "log",
                            "log": {"time": format_time(), "agent": "Tuning", "message": f"Best CV score: {cv_score:.4f} ({tuning.get('n_trials', 0)} trials)", "level": "info"}
                        }), job_id)
                
                elif node_name == "critic":
                    critic = current_state.get("critic_feedback", {})
                    status = critic.get("status", "unknown")
                    issues = critic.get("issues", [])
                    action = critic.get("primary_action", "none")
                    
                    await manager.send_message(json.dumps({
                        "type": "log",
                        "log": {"time": format_time(), "agent": "Critic", "message": f"Status: {status}. Issues: {len(issues)}. Action: {action}", "level": "warning" if status != "satisfactory" else "success"}
                    }), job_id)
                    
                    # Send stats with critic feedback
                    await send_stats_update(current_state, job_id, critic_feedback=critic.get("summary", ""))
                
                elif node_name == "improvement_router":
                    next_action = current_state.get("next_action", "stop")
                    iteration = current_state.get("iteration_number", 0)
                    
                    await manager.send_message(json.dumps({
                        "type": "log",
                        "log": {"time": format_time(), "agent": "Decision", "message": f"Iteration {iteration}: {next_action}", "level": "info"}
                    }), job_id)
                
                elif node_name == "final_report":
                    await manager.send_message(json.dumps({
                        "type": "log",
                        "log": {"time": format_time(), "agent": "Report", "message": "Generating final report...", "level": "success"}
                    }), job_id)
                
                # Update job progress in database
                if node_name in ["model_selection", "critic", "improvement_router"]:
                    async with AsyncSessionLocal() as session:
                        job = await session.get(Job, job_id)
                        if job:
                            job.current_step = node_name
                            job.current_iteration = current_state.get("iteration_number", 0)
                            best_metrics = current_state.get("best_metrics")
                            if best_metrics:
                                job.best_metrics = best_metrics
                            await session.commit()
        
        # Final completion
        await manager.send_message(json.dumps({
            "type": "step_update",
            "currentStep": len(STEP_ORDER),
            "totalSteps": len(STEP_ORDER),
            "currentNode": "completed",
        }), job_id)
        
        await manager.send_message(json.dumps({
            "type": "log",
            "log": {"time": format_time(), "agent": "System", "message": "Autonomous ML workflow completed successfully!", "level": "success"}
        }), job_id)
        
        # Update job status
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if job:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.stopping_reason = initial_state.get("stopping_reason", "Completed")
                await session.commit()
        
    except Exception as e:
        logger_msg = f"Error during execution: {str(e)}"
        await manager.send_message(json.dumps({
            "type": "log",
            "log": {"time": format_time(), "agent": "System", "message": logger_msg, "level": "error"}
        }), job_id)
        
        # Update job status
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                await session.commit()


async def send_stats_update(state: Dict[str, Any], job_id: str, critic_feedback: str = ""):
    """Send stats update to frontend."""
    best_exp = state.get("best_experiment", {})
    metrics = best_exp.get("metrics", {})
    problem_type = state.get("problem_type", "classification")
    
    def safe_format(val):
        return f"{val:.4f}" if val is not None else "0.0000"

    stats = {
        "model": best_exp.get("model_name", "Unknown"),
        "iteration": f"{state.get('iteration_number', 0)} / {state.get('max_iterations', 5)}",
        "best_score": safe_format(metrics.get('f1' if problem_type == 'classification' else 'r2')),
        "accuracy": safe_format(metrics.get('accuracy')) if problem_type == "classification" else "-",
        "precision": safe_format(metrics.get('precision')) if problem_type == "classification" else "-",
        "recall": safe_format(metrics.get('recall')) if problem_type == "classification" else "-",
        "f1": safe_format(metrics.get('f1')) if problem_type == "classification" else "-",
        "r2": safe_format(metrics.get('r2')) if problem_type == "regression" else "-",
        "rmse": safe_format(metrics.get('rmse')) if problem_type == "regression" else "-",
        "critic_feedback": critic_feedback,
        "problem_type": problem_type.capitalize(),
    }
    
    await manager.send_message(json.dumps({
        "type": "stats_update",
        "stats": stats
    }), job_id)


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


@router.post("/jobs/{job_id}/run")
async def run_agent(job_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(execute_agent_workflow, job_id)
    return {"message": "Agent execution started", "job_id": job_id}