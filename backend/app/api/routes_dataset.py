import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd

from app.database import get_db
from app.models.dataset import Dataset, ProblemType
from app.models.job import Job, JobStatus
from app.config import settings

router = APIRouter(prefix="", tags=["datasets"])


def _validate_csv(file: UploadFile) -> None:
    # Check file extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed",
        )
    # Optionally check content type
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/csv"):
        # Not strict, just warning
        pass


def _read_csv_metadata(file_path: str, target_column: str) -> dict:
    """Read CSV with pandas and return basic metadata."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid CSV file: {e}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file is empty",
        )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' not found in CSV",
        )

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": df.columns.tolist(),
    }


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    problem_type: str = Form("auto"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a CSV dataset, persist it, create Dataset and Job records.
    Returns the created job_id.
    """
    _validate_csv(file)

    # Ensure dataset directory exists
    os.makedirs(settings.DATASETS_DIR, exist_ok=True)

    # Save file with a unique name to avoid collisions
    file_ext = os.path.splitext(file.filename)[1]
    stored_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.DATASETS_DIR, stored_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read CSV metadata
    meta = _read_csv_metadata(file_path, target_column)

    # Map problem_type string to enum
    pt_enum = None
    if problem_type.lower() == "classification":
        pt_enum = ProblemType.CLASSIFICATION
    elif problem_type.lower() == "regression":
        pt_enum = ProblemType.REGRESSION
    elif problem_type.lower() == "auto":
        pt_enum = ProblemType.AUTO
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="problem_type must be 'classification', 'regression', or 'auto'",
        )

    # Create Dataset record
    dataset = Dataset(
        id=str(uuid.uuid4()),
        name=file.filename,
        filename=stored_filename,
        path=file_path,
        rows=meta["n_rows"],
        columns=meta["n_columns"],
        target_column=target_column,
        problem_type=pt_enum,
        # profile can be filled later by dataset analyst
    )
    db.add(dataset)

    # Create Job record
    job = Job(
        id=str(uuid.uuid4()),
        dataset_id=dataset.id,
        name=f"Job for {file.filename}",
        target_column=target_column,
        problem_type=problem_type.lower(),
        status=JobStatus.PENDING,
        current_step="created",
        current_iteration=0,
        max_iterations=settings.MAX_ITERATIONS,
    )
    db.add(job)

    try:
        await db.commit()
        await db.refresh(dataset)
        await db.refresh(job)
    except Exception as e:
        await db.rollback()
        # Clean up saved file on DB failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e}",
        )

    return {
        "message": "Dataset uploaded successfully",
        "job_id": job.id,
        "dataset_id": dataset.id,
        "filename": file.filename,
        "target_column": target_column,
        "problem_type": problem_type,
        "rows": meta["n_rows"],
        "columns": meta["n_columns"],
    }


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": dataset.id,
        "name": dataset.name,
        "filename": dataset.filename,
        "rows": dataset.rows,
        "columns": dataset.columns,
        "target_column": dataset.target_column,
        "problem_type": dataset.problem_type.value if dataset.problem_type else None,
        "created_at": dataset.created_at,
    }


@router.get("/")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "filename": d.filename,
            "rows": d.rows,
            "columns": d.columns,
            "target_column": d.target_column,
            "problem_type": d.problem_type.value if d.problem_type else None,
            "created_at": d.created_at,
        }
        for d in datasets
    ]