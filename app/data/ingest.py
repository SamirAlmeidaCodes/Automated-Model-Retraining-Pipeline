"""Load the sample dataset and ingest CSV files dropped into the incoming folder."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import Settings
from app.exceptions import DataError
from app.storage.database import session_scope
from app.storage.models import DatasetSchema
from app.storage.repository import add_observations, add_schema, count_observations, get_schema

logger = logging.getLogger(__name__)


def seed_if_empty(settings: Settings) -> int:
    """Load the sample CSV into SQLite when the observation table is empty.

    Returns the number of rows inserted. An existing database is left as it is.
    """
    meta = _read_meta(Path(settings.data_meta_path))
    csv_path = Path(settings.data_csv_path)
    frame = None
    with session_scope() as session:
        schema = get_schema(session)
        existing = count_observations(session)
        if schema is not None and existing > 0:
            return 0
        if not csv_path.exists():
            raise DataError(f"Sample dataset not found at {csv_path}.")
        frame = read_table(csv_path)
        target_column = meta["target_column"]
        if target_column not in frame.columns:
            raise DataError(f"Sample dataset is missing the target column {target_column!r}.")
        feature_names = [column for column in frame.columns if column != target_column]
        if not feature_names:
            raise DataError("Sample dataset has no feature columns.")
        if schema is None:
            add_schema(
                session,
                DatasetSchema(
                    dataset_name=meta["dataset_name"],
                    feature_names=feature_names,
                    target_column=target_column,
                    label_map=meta["label_map"],
                    description=meta.get("description"),
                ),
            )
        if existing > 0:
            return 0
        rows = rows_from_frame(frame, feature_names, target_column)
        inserted = add_observations(session, rows, source=f"seed:{csv_path.name}")
    logger.info("Seeded %s observations from %s", inserted, csv_path.name)
    return inserted


def ingest_incoming(settings: Settings) -> int:
    """Import every CSV in the incoming directory.

    A file that does not match the stored schema is moved to ``failed`` and
    does not stop the remaining files. Valid files move to ``processed``
    after their rows commit.
    """
    incoming = Path(settings.incoming_dir)
    if not incoming.exists():
        return 0
    files = sorted(path for path in incoming.glob("*.csv") if path.is_file())
    if not files:
        return 0

    with session_scope() as session:
        schema = get_schema(session)
        if schema is None:
            raise DataError("Cannot ingest files before the dataset schema exists.")
        feature_names = list(schema.feature_names)
        target_column = schema.target_column

    total = 0
    for path in files:
        try:
            frame = read_table(path)
            rows = rows_from_frame(frame, feature_names, target_column)
            with session_scope() as session:
                added = add_observations(session, rows, source=path.name)
            _move_unique(path, incoming / "processed")
            total += added
            logger.info("Ingested %s rows from %s", added, path.name)
        except Exception:
            logger.exception("Failed to ingest %s", path.name)
            try:
                if path.exists():
                    _move_unique(path, incoming / "failed")
            except Exception:
                logger.exception("Could not move failed file %s", path)
    return total


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV and drop an unnamed index column if one is present."""
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DataError(f"Could not read {path.name}: {exc}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    unnamed = [column for column in frame.columns if column.startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    if frame.empty:
        raise DataError(f"{path.name} has no rows.")
    return frame


def rows_from_frame(frame: pd.DataFrame, feature_names: list[str], target_column: str) -> list[dict]:
    """Convert a table into feature dictionaries and integer labels."""
    missing = [column for column in (*feature_names, target_column) if column not in frame.columns]
    if missing:
        raise DataError("Missing columns: " + ", ".join(missing))
    rows: list[dict] = []
    selected = frame.loc[:, [*feature_names, target_column]]
    for record in selected.to_dict(orient="records"):
        target = coerce_target(record.pop(target_column))
        features: dict[str, float] = {}
        for name in feature_names:
            raw = record[name]
            if pd.isna(raw):
                raise DataError(f"Feature {name!r} contains a null.")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise DataError(f"Feature {name!r} is not numeric.") from exc
            if not math.isfinite(value):
                raise DataError(f"Feature {name!r} contains a non-finite value.")
            features[name] = value
        rows.append({"features": features, "target": target})
    return rows


def coerce_target(value) -> int:
    """Convert a class label to an integer. Fractional labels are rejected."""
    if pd.isna(value):
        raise DataError("Target value is null.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"Target value {value!r} is not an integer class label.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise DataError(f"Target value {value!r} is not an integer class label.")
    return int(numeric)


def label_map_from_raw(raw: dict) -> dict[int, str]:
    """Convert JSON object keys back into integer class ids."""
    return {int(key): str(label) for key, label in raw.items()}


def _read_meta(path: Path) -> dict:
    if not path.exists():
        raise DataError(f"Dataset metadata not found at {path}.")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Dataset metadata is not valid JSON: {exc}") from exc
    for key in ("dataset_name", "target_column", "label_map"):
        if key not in meta:
            raise DataError(f"Dataset metadata is missing {key!r}.")
    if not isinstance(meta["label_map"], dict) or not meta["label_map"]:
        raise DataError("Dataset metadata label_map must be a non-empty object.")
    # Validate that the keys are integers before the schema is stored.
    label_map_from_raw(meta["label_map"])
    return meta


def _move_unique(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = directory / f"{stamp}_{source.name}"
    source.rename(destination)
    return destination
