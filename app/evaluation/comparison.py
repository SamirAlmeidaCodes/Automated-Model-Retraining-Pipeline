"""Decide whether a candidate model should replace production."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from app.exceptions import MetricError


@dataclass(frozen=True)
class PromotionDecision:
    """Result of comparing one candidate with the current production model."""

    promote: bool
    reason: str
    candidate_value: float
    baseline_value: float | None


def decide_promotion(
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float] | None,
    primary_metric: str,
    *,
    higher_is_better: bool = True,
) -> PromotionDecision:
    """Promote only when the candidate is strictly better.

    A missing production model is promoted so the service has something to
    serve. Ties keep the current model so retraining does not churn the
    artifact when nothing improved. ``baseline_metrics`` must be scores from
    the same holdout as ``candidate_metrics``.
    """
    candidate_value = _require_metric(candidate_metrics, primary_metric, "Candidate")
    if baseline_metrics is None:
        return PromotionDecision(
            promote=True,
            reason="No production model exists. Promoting the first trained model.",
            candidate_value=candidate_value,
            baseline_value=None,
        )

    baseline_value = _require_metric(baseline_metrics, primary_metric, "Production model")
    if higher_is_better:
        better = candidate_value > baseline_value
        direction = "higher"
    else:
        better = candidate_value < baseline_value
        direction = "lower"

    if better:
        reason = (
            f"Promoting: {primary_metric} improved from {baseline_value:.6f} "
            f"to {candidate_value:.6f} ({direction} is better)."
        )
        return PromotionDecision(True, reason, candidate_value, baseline_value)

    if candidate_value == baseline_value:
        reason = (
            f"Keeping the production model: {primary_metric} is unchanged at {candidate_value:.6f}."
        )
    else:
        reason = (
            f"Keeping the production model: {primary_metric} moved from "
            f"{baseline_value:.6f} to {candidate_value:.6f}, which is not better."
        )
    return PromotionDecision(False, reason, candidate_value, baseline_value)


def _require_metric(metrics: Mapping[str, float], name: str, role: str) -> float:
    if name not in metrics:
        raise MetricError(f"{role} is missing the primary metric {name!r}.")
    try:
        value = float(metrics[name])
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{role} metric {name!r} is not a number.") from exc
    if not math.isfinite(value):
        raise MetricError(f"{role} metric {name!r} is not finite.")
    return value
