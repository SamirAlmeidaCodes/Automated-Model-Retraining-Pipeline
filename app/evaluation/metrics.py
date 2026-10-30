"""Score a classifier on a holdout set."""

from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.exceptions import MetricError

logger = logging.getLogger(__name__)

# Higher is better for every metric in this set. The comparison step reads
# Settings.higher_is_better so a future lower-is-better metric can opt out.
SUPPORTED_PRIMARY_METRICS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc",
)


def evaluate_classifier(model, X_test, y_test, primary_metric: str) -> dict[str, float]:
    """Return classification metrics for ``model`` on one holdout set.

    ``roc_auc`` is included only for binary problems where the estimator
    exposes ``predict_proba``. The chosen primary metric must be present.
    """
    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        supported = ", ".join(SUPPORTED_PRIMARY_METRICS)
        raise MetricError(f"Primary metric {primary_metric!r} is not supported. Use one of: {supported}.")

    y_true = np.asarray(y_test)
    n_rows = int(y_true.shape[0])
    if n_rows == 0:
        raise MetricError("Evaluation set is empty.")
    if len(X_test) != n_rows:
        raise MetricError("Features and labels have different lengths.")

    try:
        y_pred = np.asarray(model.predict(X_test))
    except Exception as exc:
        raise MetricError(f"Model prediction failed during evaluation: {exc}") from exc

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if len(np.unique(y_true)) == 2:
        scores = _positive_class_scores(model, X_test)
        if scores is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            except ValueError as exc:
                logger.warning("ROC AUC could not be computed: %s", exc)

    if primary_metric not in metrics or not math.isfinite(metrics[primary_metric]):
        available = ", ".join(sorted(metrics))
        raise MetricError(
            f"Primary metric {primary_metric!r} is unavailable for this evaluation. Computed: {available}."
        )
    return metrics


def _positive_class_scores(model, X_test) -> np.ndarray | None:
    """Return probability scores for the positive class of a binary estimator.

    scikit-learn treats the greater label as the positive class. The column
    for that label is selected from ``classes_`` so the score matches
    ``roc_auc_score``.
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
    except Exception as exc:
        logger.warning("predict_proba failed while computing ROC AUC: %s", exc)
        return None
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        return None
    classes = list(getattr(model, "classes_", (0, 1)))
    if len(classes) != 2:
        return None
    positive = max(classes)
    return probabilities[:, list(classes).index(positive)]
