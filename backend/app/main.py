from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.config import settings, PROJECT_TITLE
from app.database import init_db, close_db
from app.api import routes_dataset, routes_agent, routes_experiments, routes_reports


structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {PROJECT_TITLE}")
    await init_db()
    logger.info("Database initialized")
    yield
    await close_db()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=PROJECT_TITLE,
    description="Agentic AI system for end-to-end machine learning workflows",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_dataset.router, prefix="/api/v1/datasets", tags=["datasets"])
app.include_router(routes_agent.router, prefix="/api/v1/agent", tags=["agent"])
app.include_router(routes_experiments.router, prefix="/api/v1/experiments", tags=["experiments"])
app.include_router(routes_reports.router, prefix="/api/v1/reports", tags=["reports"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "autonomous-ml-engineer-agent"}


@app.get("/")
async def root():
    return {
        "name": PROJECT_TITLE,
        "version": "0.1.0",
        "description": "Agentic AI system for autonomous ML workflows",
        "docs": "/docs" if settings.DEBUG else "disabled",
    }