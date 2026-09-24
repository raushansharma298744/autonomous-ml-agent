import os
import json
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="", tags=["reports"])


@router.get("/{job_id}")
async def get_report(job_id: str):
    report_path = f"./reports/report_{job_id}.json"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/{job_id}/html")
async def get_report_html(job_id: str):
    html_path = f"./reports/report_{job_id}.html"
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="HTML Report not found")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()