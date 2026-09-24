# Experiment Service - Business logic for experiment management
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.models.experiment import Experiment, ExperimentCreate
from app.models.job import Job
import uuid


async def create_experiment(
    db: AsyncSession,
    experiment_data: ExperimentCreate
) -> Experiment:
    """Create a new experiment record."""
    experiment = Experiment(
        id=str(uuid.uuid4()),
        **experiment_data.model_dump()
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment


async def get_experiment(db: AsyncSession, experiment_id: str) -> Optional[Experiment]:
    """Get experiment by ID."""
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    return result.scalar_one_or_none()


async def get_experiments_by_job(db: AsyncSession, job_id: str) -> List[Experiment]:
    """Get all experiments for a job."""
    result = await db.execute(
        select(Experiment)
        .where(Experiment.job_id == job_id)
        .order_by(desc(Experiment.created_at))
    )
    return result.scalars().all()


async def update_experiment(
    db: AsyncSession,
    experiment_id: str,
    updates: Dict[str, Any]
) -> Optional[Experiment]:
    """Update experiment fields."""
    experiment = await get_experiment(db, experiment_id)
    if not experiment:
        return None
    
    for key, value in updates.items():
        if hasattr(experiment, key):
            setattr(experiment, key, value)
    
    await db.commit()
    await db.refresh(experiment)
    return experiment


async def get_best_experiment(db: AsyncSession, job_id: str, metric: str = "f1") -> Optional[Experiment]:
    """Get best experiment for a job based on metric."""
    experiments = await get_experiments_by_job(db, job_id)
    
    best = None
    best_value = -float('inf')
    
    for exp in experiments:
        if exp.metrics and metric in exp.metrics:
            value = exp.metrics[metric]
            if value > best_value:
                best_value = value
                best = exp
    
    return best