"""Queries for observations, model versions, and training runs."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.clock import utcnow
from app.exceptions import DataError
from app.storage.models import DatasetSchema, ModelVersion, Observation, TrainingRun


def get_schema(session: Session) -> DatasetSchema | None:
    """Return the dataset schema, if one has been stored."""
    return session.scalars(select(DatasetSchema).limit(1)).first()


def count_observations(session: Session) -> int:
    """Return the number of stored observations."""
    return int(session.scalar(select(func.count(Observation.id))) or 0)


def latest_watermark(session: Session) -> int:
    """Return the highest observation id already covered by a training attempt.

    Rejected candidates advance the watermark too. Otherwise a candidate that
    lost the comparison would be retrained on every schedule tick.
    """
    value = session.scalar(select(func.max(ModelVersion.data_watermark)))
    return int(value or 0)


def count_observations_after(session: Session, watermark: int) -> int:
    """Return how many observations arrived after ``watermark``."""
    value = session.scalar(
        select(func.count(Observation.id)).where(Observation.id > watermark)
    )
    return int(value or 0)


def class_counts(session: Session) -> dict[int, int]:
    """Return observation counts keyed by integer class label."""
    rows = session.execute(
        select(Observation.target, func.count(Observation.id)).group_by(Observation.target)
    ).all()
    return {int(target): int(count) for target, count in rows}


def add_schema(session: Session, schema: DatasetSchema) -> DatasetSchema:
    """Insert the dataset schema."""
    session.add(schema)
    session.flush()
    return schema


def add_observations(session: Session, rows: list[dict], source: str) -> int:
    """Insert labeled rows. Each row has ``features`` and ``target``."""
    observations = [
        Observation(
            features=row["features"],
            target=row["target"],
            source=source,
            ingested_at=utcnow(),
        )
        for row in rows
    ]
    session.add_all(observations)
    session.flush()
    return len(observations)


def load_frame(session: Session, feature_names: list[str], target_column: str) -> pd.DataFrame:
    """Load every observation into a frame ordered by id."""
    observations = session.scalars(select(Observation).order_by(Observation.id.asc())).all()
    records = []
    for observation in observations:
        features = observation.features or {}
        record = {"id": observation.id, target_column: observation.target}
        for name in feature_names:
            if name not in features:
                raise DataError(f"Observation {observation.id} is missing feature {name!r}.")
            record[name] = float(features[name])
        records.append(record)
    if not records:
        raise DataError("No observations are available.")
    return pd.DataFrame.from_records(records)


def get_production(session: Session) -> ModelVersion | None:
    """Return the model that is currently marked as production."""
    return session.scalars(
        select(ModelVersion).where(ModelVersion.is_production.is_(True)).limit(1)
    ).first()


def list_versions(session: Session) -> list[ModelVersion]:
    """Return model versions, newest first."""
    return list(session.scalars(select(ModelVersion).order_by(ModelVersion.id.desc())))


def list_runs(session: Session, limit: int = 100) -> list[TrainingRun]:
    """Return training runs, newest first."""
    return list(
        session.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit))
    )


def add_training_run(session: Session, **fields) -> TrainingRun:
    """Insert a training run and flush it so the id is available."""
    run = TrainingRun(**fields)
    session.add(run)
    session.flush()
    return run
