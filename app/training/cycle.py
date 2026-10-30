"""Ingest new rows, train a candidate, and promote it only when it is better."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

from app.clock import utcnow
from app.config import Settings
from app.data.ingest import ingest_incoming, label_map_from_raw
from app.evaluation.comparison import decide_promotion
from app.evaluation.metrics import evaluate_classifier
from app.exceptions import DataError
from app.registry.artifacts import (
    load_bundle,
    resolve_artifact,
    restore_production,
    snapshot_production,
)
from app.registry.promotion import persist_candidate
from app.storage.database import session_scope
from app.storage.repository import (
    add_training_run,
    count_observations,
    count_observations_after,
    get_production,
    get_schema,
    latest_watermark,
    load_frame,
)
from app.training.split import stable_train_test_split
from app.training.trainer import build_estimator, estimator_params, fit_estimator

logger = logging.getLogger(__name__)

_TRAIN_LOCK = threading.Lock()


@dataclass
class CycleResult:
    """Outcome of one retraining check."""

    status: str
    message: str
    version: str | None = None
    primary_metric: str | None = None
    primary_metric_value: float | None = None
    baseline_primary_metric_value: float | None = None
    new_rows: int = 0
    ingested_rows: int = 0
    run_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_retraining_cycle(settings: Settings, trigger: str, force: bool = False) -> CycleResult:
    """Check for new data and train when there is something new, or when forced.

    Only one cycle runs at a time. A second caller receives status ``busy``.
    """
    if not _TRAIN_LOCK.acquire(blocking=False):
        logger.info("Retraining check skipped because another run is in progress.")
        return CycleResult(status="busy", message="A retraining run is already in progress.")
    try:
        return _run_locked(settings, trigger, force)
    finally:
        _TRAIN_LOCK.release()


def _run_locked(settings: Settings, trigger: str, force: bool) -> CycleResult:
    started = utcnow()
    try:
        ingested_rows = ingest_incoming(settings)
        return _train(settings, trigger, force, ingested_rows, started)
    except Exception as exc:
        logger.exception("Retraining run failed")
        message = _error_message(exc)
        _record_failure(settings, trigger, started, message)
        return CycleResult(status="failed", message=message)


def _train(settings, trigger: str, force: bool, ingested_rows: int, started) -> CycleResult:
    snapshot = None
    snapshot_needed = False
    try:
        with session_scope() as session:
            schema = get_schema(session)
            if schema is None:
                raise DataError("Dataset schema is not initialized. Seed the database before training.")
            total = count_observations(session)
            if total < 8:
                raise DataError(f"Need at least 8 observations to train; found {total}.")

            watermark = latest_watermark(session)
            new_rows = count_observations_after(session, watermark)
            production = get_production(session)
            logger.info(
                "Retraining check trigger=%s new_rows=%s ingested=%s force=%s",
                trigger,
                new_rows,
                ingested_rows,
                force,
            )
            if new_rows == 0 and production is not None and not force:
                message = "No new observations since the last training run."
                run = add_training_run(
                    session,
                    started_at=started,
                    finished_at=utcnow(),
                    trigger=trigger,
                    status="skipped",
                    message=message,
                    model_version_id=None,
                    new_rows=0,
                    ingested_rows=ingested_rows,
                )
                logger.info(message)
                return CycleResult(
                    status="skipped",
                    message=message,
                    new_rows=0,
                    ingested_rows=ingested_rows,
                    run_id=run.id,
                )

            feature_names = list(schema.feature_names)
            target_column = schema.target_column
            label_map = label_map_from_raw(schema.label_map)
            frame = load_frame(session, feature_names, target_column)
            data_watermark = int(frame["id"].max())
            train_df, test_df = stable_train_test_split(
                frame,
                id_column="id",
                target_column=target_column,
                test_modulus=settings.test_modulus,
                random_state=settings.random_state,
            )
            estimator = fit_estimator(
                build_estimator(settings),
                train_df,
                feature_names,
                target_column,
            )
            # Score the candidate first, then the current production model on
            # the same holdout. The stored historical metric is not reused,
            # because the holdout grows when new rows arrive.
            candidate_metrics = evaluate_classifier(
                estimator,
                test_df.loc[:, feature_names],
                test_df[target_column],
                settings.primary_metric,
            )
            baseline_metrics = None
            if production is not None:
                artifact = resolve_artifact(settings.models_dir, production.artifact_path)
                bundle = load_bundle(artifact)
                if list(bundle["feature_names"]) != feature_names:
                    raise DataError("Production model features do not match the dataset schema.")
                baseline_metrics = evaluate_classifier(
                    bundle["estimator"],
                    test_df.loc[:, feature_names],
                    test_df[target_column],
                    settings.primary_metric,
                )

            decision = decide_promotion(
                candidate_metrics,
                baseline_metrics,
                settings.primary_metric,
                higher_is_better=settings.higher_is_better,
            )
            if decision.promote:
                snapshot = snapshot_production(settings.models_dir)
                snapshot_needed = True
            row = persist_candidate(
                session,
                settings=settings,
                estimator=estimator,
                feature_names=feature_names,
                label_map=label_map,
                metrics=candidate_metrics,
                decision=decision,
                algorithm=settings.algorithm,
                hyperparams=estimator_params(estimator),
                training_rows=int(len(train_df)),
                test_rows=int(len(test_df)),
                data_watermark=data_watermark,
                primary_metric=settings.primary_metric,
            )
            status = "promoted" if decision.promote else "rejected"
            run = add_training_run(
                session,
                started_at=started,
                finished_at=utcnow(),
                trigger=trigger,
                status=status,
                message=decision.reason,
                model_version_id=row.id,
                new_rows=new_rows,
                ingested_rows=ingested_rows,
            )
            logger.info("Retraining %s: %s", status, decision.reason)
            result = CycleResult(
                status=status,
                message=decision.reason,
                version=row.version,
                primary_metric=row.primary_metric,
                primary_metric_value=row.primary_metric_value,
                baseline_primary_metric_value=row.baseline_primary_metric_value,
                new_rows=new_rows,
                ingested_rows=ingested_rows,
                run_id=run.id,
            )
        return result
    except Exception:
        if snapshot_needed:
            logger.warning("Rolling back the production artifact after a failed promotion.")
            restore_production(settings.models_dir, snapshot)
        raise


def _record_failure(settings: Settings, trigger: str, started, message: str) -> None:
    try:
        with session_scope() as session:
            add_training_run(
                session,
                started_at=started,
                finished_at=utcnow(),
                trigger=trigger,
                status="failed",
                message=message,
                model_version_id=None,
                new_rows=0,
                ingested_rows=0,
            )
    except Exception:
        logger.exception("Could not record the failed training run")


def _error_message(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__
