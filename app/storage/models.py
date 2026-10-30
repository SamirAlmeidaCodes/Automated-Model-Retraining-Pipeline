"""SQLAlchemy tables for the retraining history."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

from app.clock import utcnow


class Base(DeclarativeBase):
    """Declarative base for pipeline tables."""


class DatasetSchema(Base):
    """Feature order and label names for the active dataset."""

    __tablename__ = "dataset_schema"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_names: Mapped[list] = mapped_column(JSON, nullable=False)
    target_column: Mapped[str] = mapped_column(String(128), nullable=False)
    label_map: Mapped[dict] = mapped_column(JSON, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class Observation(Base):
    """One labeled row used for training or evaluation."""

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    target: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False, default="unknown")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class ModelVersion(Base):
    """A trained model, its metrics, and whether it was promoted."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    hyperparams: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    primary_metric: Mapped[str] = mapped_column(String(64), nullable=False)
    primary_metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_primary_metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    is_production: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    training_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    test_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    data_watermark: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"ModelVersion(version={self.version!r}, status={self.status!r})"


class TrainingRun(Base):
    """One scheduler or manual check, including skips and failures."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    trigger: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    new_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ingested_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
