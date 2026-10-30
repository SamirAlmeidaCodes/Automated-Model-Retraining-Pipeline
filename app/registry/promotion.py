"""Store a candidate and publish it when the comparison says to."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.clock import utcnow
from app.config import Settings
from app.evaluation.comparison import PromotionDecision
from app.registry.artifacts import publish_production, save_bundle
from app.storage.models import ModelVersion

logger = logging.getLogger(__name__)


def persist_candidate(
    session: Session,
    *,
    settings: Settings,
    estimator,
    feature_names: list[str],
    label_map: dict[int, str],
    metrics: dict[str, float],
    decision: PromotionDecision,
    algorithm: str,
    hyperparams: dict,
    training_rows: int,
    test_rows: int,
    data_watermark: int,
    primary_metric: str,
) -> ModelVersion:
    """Write the candidate artifact and database row.

    The production file is replaced only when ``decision.promote`` is true.
    Callers should snapshot that file first and restore it if the surrounding
    transaction rolls back. The version id is assigned by the database, then
    written into the bundle so the file and the row share one name.
    """
    models_dir = Path(settings.models_dir)
    row = ModelVersion(
        version=f"pending-{uuid4().hex[:12]}",
        created_at=utcnow(),
        algorithm=algorithm,
        hyperparams=hyperparams,
        metrics=dict(metrics),
        primary_metric=primary_metric,
        primary_metric_value=decision.candidate_value,
        baseline_primary_metric_value=decision.baseline_value,
        artifact_path="pending",
        is_production=False,
        promoted=False,
        training_rows=training_rows,
        test_rows=test_rows,
        data_watermark=data_watermark,
        status="rejected",
        notes=decision.reason,
    )
    session.add(row)
    session.flush()
    row.version = f"v{row.id:04d}"
    relative_path = f"versions/{row.version}.joblib"
    row.artifact_path = relative_path
    save_bundle(
        models_dir / relative_path,
        estimator=estimator,
        feature_names=feature_names,
        label_map=label_map,
        version=row.version,
        algorithm=algorithm,
        metrics=metrics,
        primary_metric=primary_metric,
    )

    if decision.promote:
        session.execute(
            update(ModelVersion)
            .where(ModelVersion.is_production.is_(True))
            .values(is_production=False, status="archived")
        )
        row.is_production = True
        row.promoted = True
        row.status = "production"
        session.flush()
        publish_production(models_dir, models_dir / relative_path)
    else:
        row.promoted = False
        row.status = "rejected"
        session.flush()

    logger.info(
        "Stored %s status=%s %s=%.6f baseline=%s",
        row.version,
        row.status,
        primary_metric,
        decision.candidate_value,
        "none" if decision.baseline_value is None else f"{decision.baseline_value:.6f}",
    )
    return row
