"""Runtime configuration loaded from the environment and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from apscheduler.triggers.cron import CronTrigger
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.evaluation.metrics import SUPPORTED_PRIMARY_METRICS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_ALGORITHMS = {
    "logistic_regression": "logistic_regression",
    "logistic": "logistic_regression",
    "logreg": "logistic_regression",
    "gradient_boosting": "gradient_boosting",
    "gb": "gradient_boosting",
    "gbrt": "gradient_boosting",
    "random_forest": "random_forest",
    "rf": "random_forest",
}


class Settings(BaseSettings):
    """Paths, model choices, and the retraining schedule."""

    app_name: str = "Model Retraining Pipeline"
    database_url: str = "sqlite:///./db/pipeline.db"
    data_csv_path: str = "./data/breast_cancer.csv"
    data_meta_path: str = "./data/dataset_meta.json"
    incoming_dir: str = "./data/incoming"
    models_dir: str = "./models"
    primary_metric: str = "f1_weighted"
    higher_is_better: bool = True
    algorithm: str = "logistic_regression"
    n_estimators: int = Field(default=120, ge=1, le=2000)
    learning_rate: float = Field(default=0.08, gt=0, le=1)
    max_depth: int = Field(default=3, ge=1, le=50)
    test_modulus: int = Field(default=5, ge=2, le=100)
    random_state: int = Field(default=42, ge=0)
    retrain_cron: str = "0 2 * * *"
    retrain_interval_minutes: int | None = None
    train_on_startup: bool = True
    scheduler_enabled: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("primary_metric")
    @classmethod
    def metric_is_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_PRIMARY_METRICS:
            supported = ", ".join(SUPPORTED_PRIMARY_METRICS)
            raise ValueError(f"PRIMARY_METRIC must be one of: {supported}.")
        return normalized

    @field_validator("algorithm")
    @classmethod
    def algorithm_is_supported(cls, value: str) -> str:
        normalized = _ALGORITHMS.get(value.strip().lower())
        if normalized is None:
            raise ValueError(
                "ALGORITHM must be logistic_regression, gradient_boosting, or random_forest."
            )
        return normalized

    @field_validator("log_level")
    @classmethod
    def log_level_is_known(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
        return normalized

    @field_validator("retrain_cron")
    @classmethod
    def cron_is_valid(cls, value: str) -> str:
        parts = value.split()
        if len(parts) != 5:
            raise ValueError(
                "RETRAIN_CRON must contain 5 fields: minute hour day month day_of_week."
            )
        minute, hour, day, month, day_of_week = parts
        try:
            CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
                timezone="UTC",
            )
        except Exception as exc:
            raise ValueError(f"RETRAIN_CRON is invalid: {exc}") from exc
        return value

    @field_validator("retrain_interval_minutes")
    @classmethod
    def interval_is_positive(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 1:
            raise ValueError("RETRAIN_INTERVAL_MINUTES must be at least 1.")
        return value

    @model_validator(mode="after")
    def resolve_paths(self) -> Settings:
        """Anchor relative paths at the repository root, not the process cwd.

        Mutate this instance and return it. Pydantic 2 ignores a replacement
        instance returned from a model validator during ``__init__``.
        """
        resolved = {
            "data_csv_path": _absolute(self.data_csv_path),
            "data_meta_path": _absolute(self.data_meta_path),
            "incoming_dir": _absolute(self.incoming_dir),
            "models_dir": _absolute(self.models_dir),
            "database_url": _absolute_sqlite_url(self.database_url),
        }
        for name, value in resolved.items():
            if getattr(self, name) != value:
                object.__setattr__(self, name, value)
        return self


def _absolute(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _absolute_sqlite_url(url: str) -> str:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw = url[len(prefix) :]
    if raw in {"", ":memory:"} or raw.startswith("file:"):
        return url
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return prefix + str(path)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings."""
    return Settings()
