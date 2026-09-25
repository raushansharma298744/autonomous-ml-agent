from functools import lru_cache
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project branding
PROJECT_TITLE = "AutoML Agent – Autonomous Machine Learning System"
SHORT_NAME = "automl-agent"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # LLM
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    LLM_MODEL: str = Field(default="llama-3.3-70b-versatile", description="LLM model name")
    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(default=4096, description="LLM max tokens")

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/autonomous_ml",
        description="PostgreSQL connection string",
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str]) -> str:
        if isinstance(v, str):
            if v.startswith("postgres://"):
                return v.replace("postgres://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql://"):
                return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="autonomous_ml")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)

    # MLflow
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    MLFLOW_ARTIFACT_ROOT: str = Field(default="./mlruns")

    # Application
    APP_HOST: str = Field(default="0.0.0.0")
    APP_PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")

    # Agent Configuration
    MAX_ITERATIONS: int = Field(default=5, description="Maximum autonomous iterations")
    MIN_IMPROVEMENT_THRESHOLD: float = Field(
        default=0.01, description="Minimum improvement to continue"
    )
    DEFAULT_TEST_SIZE: float = Field(default=0.2, description="Default test split ratio")
    RANDOM_STATE: int = Field(default=42, description="Random seed for reproducibility")

    # File Paths
    DATASETS_DIR: str = Field(default="./datasets")
    REPORTS_DIR: str = Field(default="./reports")
    MLRUNS_DIR: str = Field(default="./mlruns")

    # Frontend
    FRONTEND_URL: str = Field(default="http://localhost:5173")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()