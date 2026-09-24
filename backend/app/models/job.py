from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from sqlalchemy import String, Text, DateTime, JSON, ForeignKey, Enum as SQLEnum, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import uuid


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    problem_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[JobStatus] = mapped_column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    current_step: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    current_iteration: Mapped[int] = mapped_column(Integer, default=0)
    max_iterations: Mapped[int] = mapped_column(Integer, default=5)
    best_experiment_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    best_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    stopping_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="jobs")
    experiments: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="job")


class JobCreate(BaseModel):
    dataset_id: str
    name: str
    target_column: str
    problem_type: str
    max_iterations: int = 5
    config: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    model_config = {'from_attributes': True}
    
    id: str
    dataset_id: str
    name: str
    target_column: str
    problem_type: str
    status: JobStatus
    current_step: Optional[str]
    current_iteration: int
    max_iterations: int
    best_experiment_id: Optional[str]
    best_metrics: Optional[Dict[str, Any]]
    stopping_reason: Optional[str]
    error_message: Optional[str]
    config: Optional[Dict[str, Any]]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]