from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from sqlalchemy import String, Text, DateTime, JSON, ForeignKey, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import uuid


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"), nullable=False)
    parent_experiment_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiments.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    preprocessing_version: Mapped[int] = mapped_column(Integer, default=1)
    feature_version: Mapped[int] = mapped_column(Integer, default=1)
    hyperparameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    training_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    critic_feedback: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    job: Mapped["Job"] = relationship("Job", back_populates="experiments")
    parent: Mapped[Optional["Experiment"]] = relationship("Experiment", remote_side=[id], back_populates="children")
    children: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="parent")


class ExperimentCreate(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    job_id: str
    parent_experiment_id: Optional[str] = None
    name: str
    model_name: str
    model_type: str
    preprocessing_version: int = 1
    feature_version: int = 1
    hyperparameters: Optional[Dict[str, Any]] = None


class ExperimentResponse(BaseModel):
    model_config = {'protected_namespaces': (), 'from_attributes': True}
    
    id: str
    job_id: str
    parent_experiment_id: Optional[str]
    name: str
    model_name: str
    model_type: str
    preprocessing_version: int
    feature_version: int
    hyperparameters: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    training_duration_seconds: Optional[float]
    mlflow_run_id: Optional[str]
    critic_feedback: Optional[Dict[str, Any]]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]