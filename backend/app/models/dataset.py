from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from sqlalchemy import String, Text, DateTime, JSON, Enum as SQLEnum, ForeignKey, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import uuid


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    rows: Mapped[int] = mapped_column(Integer, default=0)
    columns: Mapped[int] = mapped_column(Integer, default=0)
    target_column: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    problem_type: Mapped[Optional[ProblemType]] = mapped_column(SQLEnum(ProblemType), nullable=True)
    profile: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    jobs: Mapped[List["Job"]] = relationship("Job", back_populates="dataset")


class DatasetCreate(BaseModel):
    name: str
    filename: str
    path: str
    target_column: Optional[str] = None
    problem_type: Optional[ProblemType] = None


class DatasetResponse(BaseModel):
    model_config = {'from_attributes': True}
    
    id: str
    name: str
    filename: str
    rows: int
    columns: int
    target_column: Optional[str]
    problem_type: Optional[ProblemType]
    created_at: datetime