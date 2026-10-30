"""HTTP routes for prediction, model history, and the dashboard."""

from __future__ import annotations

import logging
from datetime import timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.dashboard import render_dashboard
from app.api.schemas import (
    CurrentModelResponse,
    HealthResponse,
    HistoryResponse,
    PredictRequest,
    PredictResponse,
    RetrainResponse,
)
from app.api.serializers import current_model_dict, run_to_dict, version_to_dict
from app.data.features import vectorize_features
from app.data.ingest import label_map_from_raw
from app.exceptions import DataError, ModelNotReadyError
from app.registry.artifacts import load_production_bundle
from app.scheduler.jobs import schedule_description
from app.storage.database import session_scope
from app.storage.repository import (
    class_counts,
    count_observations,
    get_production,
    get_schema,
    list_runs,
    list_versions,
)
from app.training.cycle import run_retraining_cycle

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report whether a production artifact is registered and present on disk."""
    settings = request.app.state.settings
    with session_scope() as session:
        row = get_production(session)
        version = None if row is None else row.version
    loaded = version is not None and (Path(settings.models_dir) / "production.joblib").exists()
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        version=version,
        scheduler_enabled=bool(settings.scheduler_enabled),
    )


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Score one row with the current production model."""
    settings = request.app.state.settings
    try:
        bundle = load_production_bundle(Path(settings.models_dir))
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        frame = vectorize_features(body.features, list(bundle["feature_names"]))
    except DataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    estimator = bundle["estimator"]
    try:
        prediction = int(estimator.predict(frame)[0])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    label_map = {int(key): str(value) for key, value in bundle["label_map"].items()}
    probabilities = None
    if hasattr(estimator, "predict_proba"):
        try:
            scores = estimator.predict_proba(frame)[0]
            probabilities = {
                label_map.get(int(class_id), str(int(class_id))): float(score)
                for class_id, score in zip(estimator.classes_, scores)
            }
        except Exception:
            logger.warning("predict_proba failed; returning the label only", exc_info=True)
            probabilities = None
    return PredictResponse(
        prediction=prediction,
        label=label_map.get(prediction, str(prediction)),
        probabilities=probabilities,
        model_version=str(bundle["version"]),
    )


@router.get("/model/current", response_model=CurrentModelResponse)
def current_model(request: Request) -> CurrentModelResponse:
    """Return the active production model, its metrics, and the dataset it was trained on."""
    settings = request.app.state.settings
    with session_scope() as session:
        row = get_production(session)
        if row is None:
            raise HTTPException(status_code=404, detail="No production model is available yet.")
        schema = get_schema(session)
        payload = current_model_dict(
            row,
            dataset_name=schema.dataset_name if schema else "Unknown dataset",
            description=schema.description if schema else None,
            feature_names=list(schema.feature_names) if schema else [],
            label_map=dict(schema.label_map) if schema else {},
            observation_count=count_observations(session),
            class_counts=class_counts(session),
            artifact_available=(Path(settings.models_dir) / "production.joblib").exists(),
        )
    return CurrentModelResponse(**payload)


@router.get("/model/history", response_model=HistoryResponse)
def model_history() -> HistoryResponse:
    """Return every stored model version and the recent retraining checks."""
    with session_scope() as session:
        versions = [version_to_dict(row) for row in list_versions(session)]
        runs = [run_to_dict(row) for row in list_runs(session)]
    return HistoryResponse(versions=versions, runs=runs)


@router.post("/model/retrain", response_model=RetrainResponse)
def retrain(request: Request, force: bool = False):
    """Ingest new CSV files and train when new rows exist.

    Pass ``force=true`` to train even when no rows have arrived since the last
    attempt. The candidate is promoted only when its primary metric is strictly
    better than the current production model on the same holdout.
    """
    settings = request.app.state.settings
    result = run_retraining_cycle(settings, trigger="manual", force=force)
    payload = RetrainResponse(**result.to_dict()).model_dump()
    status_code = 200
    if result.status == "failed":
        status_code = 500
    elif result.status == "busy":
        status_code = 409
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=payload)
    return payload


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    """Show the current model, the metric comparison, and retraining history."""
    settings = request.app.state.settings
    with session_scope() as session:
        production = get_production(session)
        schema = get_schema(session)
        versions = [version_to_dict(row) for row in list_versions(session)]
        runs = [run_to_dict(row) for row in list_runs(session)]
        counts = class_counts(session)
        total = count_observations(session)
        labels = label_map_from_raw(schema.label_map) if schema else {}
        class_count_rows = [
            {"label": labels.get(key, str(key)), "count": count}
            for key, count in sorted(counts.items())
        ]
        if production is None or schema is None:
            current = None
        else:
            current = current_model_dict(
                production,
                dataset_name=schema.dataset_name,
                description=schema.description,
                feature_names=list(schema.feature_names),
                label_map=dict(schema.label_map),
                observation_count=total,
                class_counts=counts,
                artifact_available=(Path(settings.models_dir) / "production.joblib").exists(),
            )
        dataset_name = schema.dataset_name if schema else "No dataset loaded"
    page = render_dashboard(
        {
            "app_name": settings.app_name,
            "dataset_name": dataset_name,
            "schedule": schedule_description(settings),
            "next_run": _next_run(request),
            "primary_metric": settings.primary_metric,
            "higher_is_better": settings.higher_is_better,
            "current": current,
            "versions": versions,
            "runs": runs,
            "observation_count": total,
            "class_counts": class_count_rows,
        }
    )
    return HTMLResponse(page)


def _next_run(request: Request) -> str:
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return "not scheduled"
    job = scheduler.get_job("retrain")
    if job is None or job.next_run_time is None:
        return "not scheduled"
    moment = job.next_run_time.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M UTC")
