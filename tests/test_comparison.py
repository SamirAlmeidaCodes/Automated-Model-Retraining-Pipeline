"""Unit tests for the promotion rule."""

import math

import pytest

from app.evaluation.comparison import decide_promotion
from app.exceptions import MetricError


def test_first_model_is_promoted_when_there_is_no_baseline():
    decision = decide_promotion({"f1_weighted": 0.91}, None, "f1_weighted")
    assert decision.promote is True
    assert decision.baseline_value is None
    assert decision.candidate_value == pytest.approx(0.91)
    assert "first trained model" in decision.reason


def test_strict_improvement_promotes():
    decision = decide_promotion(
        {"f1_weighted": 0.94},
        {"f1_weighted": 0.90},
        "f1_weighted",
        higher_is_better=True,
    )
    assert decision.promote is True
    assert "improved" in decision.reason


def test_worse_metric_does_not_promote():
    decision = decide_promotion(
        {"f1_weighted": 0.80},
        {"f1_weighted": 0.91},
        "f1_weighted",
    )
    assert decision.promote is False
    assert "not better" in decision.reason


def test_tie_does_not_promote():
    decision = decide_promotion(
        {"f1_weighted": 0.9},
        {"f1_weighted": 0.9},
        "f1_weighted",
    )
    assert decision.promote is False
    assert "unchanged" in decision.reason


def test_lower_is_better_promotes_only_when_the_score_falls():
    improved = decide_promotion(
        {"f1_weighted": 0.2},
        {"f1_weighted": 0.4},
        "f1_weighted",
        higher_is_better=False,
    )
    worse = decide_promotion(
        {"f1_weighted": 0.5},
        {"f1_weighted": 0.4},
        "f1_weighted",
        higher_is_better=False,
    )
    assert improved.promote is True
    assert worse.promote is False


def test_missing_candidate_metric_raises():
    with pytest.raises(MetricError, match="Candidate"):
        decide_promotion({"accuracy": 1.0}, {"f1_weighted": 0.5}, "f1_weighted")


def test_missing_baseline_metric_raises():
    with pytest.raises(MetricError, match="Production"):
        decide_promotion({"f1_weighted": 1.0}, {"accuracy": 0.5}, "f1_weighted")


def test_non_numeric_metric_raises():
    with pytest.raises(MetricError, match="not a number"):
        decide_promotion({"f1_weighted": "high"}, None, "f1_weighted")


def test_non_finite_metric_raises():
    with pytest.raises(MetricError, match="not finite"):
        decide_promotion({"f1_weighted": math.nan}, None, "f1_weighted")
