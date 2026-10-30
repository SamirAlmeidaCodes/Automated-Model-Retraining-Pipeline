"""Convert stored rows into plain dictionaries for the API and the dashboard."""

from __future__ import annotations

from app.clock import to_iso
from app.data.ingest import label_map_from_raw
from app.storage.models import ModelVersion, TrainingRun


def version_to_dict(row: ModelVersion) -> dict:
    """Serialize one model version, including the metric delta versus baseline."""
    baseline = row.baseline_primary_metric_value
    delta = None if baseline is None else float(row.primary_metric_value) - float(baseline)
    return {
        "id": row.id,
        "version": row.version,
        "created_at": to_iso(row.created_at),
        "algorithm": row.algorithm,
        "hyperparams": dict(row.hyperparams or {}),
        "metrics": {key: float(value) for key, value in (row.metrics or {}).items()},
        "primary_metric": row.primary_metric,
        "primary_metric_value": float(row.primary_metric_value),
        "baseline_primary_metric_value": None if baseline is None else float(baseline),
        "delta": delta,
        "is_production": bool(row.is_production),
        "promoted": bool(row.promoted),
        "status": row.status,
        "training_rows": int(row.training_rows),
        "test_rows": int(row.test_rows),
        "data_watermark": int(row.data_watermark),
        "notes": row.notes,
        "artifact_path": row.artifact_path,
    }


def run_to_dict(row: TrainingRun) -> dict:
    """Serialize one training run."""
    return {
        "id": row.id,
        "started_at": to_iso(row.started_at),
        "finished_at": to_iso(row.finished_at),
        "trigger": row.trigger,
        "status": row.status,
        "message": row.message,
        "model_version_id": row.model_version_id,
        "new_rows": int(row.new_rows),
        "ingested_rows": int(row.ingested_rows),
    }


def current_model_dict(
    row: ModelVersion,
    *,
    dataset_name: str,
    description: str | None,
    feature_names: list[str],
    label_map: dict,
    observation_count: int,
    class_counts: dict[int, int],
    artifact_available: bool,
) -> dict:
    """Serialize the active model plus dataset context for the current-model endpoint."""
    labels = label_map_from_raw(label_map)
    payload = version_to_dict(row)
    payload.update(
        {
            "dataset_name": dataset_name,
            "description": description,
            "feature_names": list(feature_names),
            "label_map": {str(key): value for key, value in labels.items()},
            "observation_count": observation_count,
            "class_counts": [
                {"label": labels.get(key, str(key)), "count": count}
                for key, count in sorted(class_counts.items())
            ],
            "artifact_available": artifact_available,
        }
    )
    return payload
