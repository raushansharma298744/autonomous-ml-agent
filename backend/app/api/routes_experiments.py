from fastapi import APIRouter

router = APIRouter(prefix="", tags=["experiments"])


@router.get("/")
async def list_experiments():
    return {"message": "List experiments - to be implemented"}


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    return {"message": f"Get experiment {experiment_id} - to be implemented"}


@router.get("/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    return {"message": f"Get experiment metrics {experiment_id} - to be implemented"}