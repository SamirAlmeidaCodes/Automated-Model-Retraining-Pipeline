"""Unit tests for holdout metrics."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from app.evaluation.metrics import evaluate_classifier
from app.exceptions import MetricError


class ScriptedModel:
    """Return a fixed prediction grid so the expected scores can be calculated by hand."""

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([0, 1, 1, 0])

    def predict_proba(self, features):
        return np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.4, 0.6],
                [0.7, 0.3],
            ]
        )


def test_weighted_metrics_match_hand_calculation():
    # Truth:      0, 1, 0, 0
    # Prediction: 0, 1, 1, 0
    # Class 0 precision 1, recall 2/3, f1 0.8, support 3.
    # Class 1 precision 0.5, recall 1, f1 2/3, support 1.
    model = ScriptedModel()
    labels = np.array([0, 1, 0, 0])
    features = np.zeros((4, 2))
    metrics = evaluate_classifier(model, features, labels, "f1_weighted")

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision_weighted"] == pytest.approx((1 * 3 + 0.5 * 1) / 4)
    assert metrics["recall_weighted"] == pytest.approx(0.75)
    assert metrics["f1_weighted"] == pytest.approx((0.8 * 3 + (2 / 3) * 1) / 4)
    # The only positive label has the highest score, so ROC AUC is 1.
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_perfect_predictions_score_one():
    class Perfect:
        classes_ = np.array([0, 1])

        def predict(self, features):
            return np.array([0, 1, 1, 0])

        def predict_proba(self, features):
            predicted = self.predict(features)
            probabilities = np.zeros((len(predicted), 2))
            probabilities[np.arange(len(predicted)), predicted] = 1
            return probabilities

    labels = np.array([0, 1, 1, 0])
    metrics = evaluate_classifier(Perfect(), np.zeros((4, 1)), labels, "accuracy")
    assert metrics["accuracy"] == pytest.approx(1)
    assert metrics["f1_weighted"] == pytest.approx(1)
    assert metrics["roc_auc"] == pytest.approx(1)


def test_missing_predicted_class_does_not_crash():
    class AlwaysZero:
        classes_ = np.array([0, 1])

        def predict(self, features):
            return np.zeros(len(features), dtype=int)

        def predict_proba(self, features):
            probabilities = np.zeros((len(features), 2))
            probabilities[:, 0] = 1
            return probabilities

    labels = np.array([0, 1, 0, 1])
    metrics = evaluate_classifier(AlwaysZero(), np.zeros((4, 1)), labels, "f1_weighted")
    assert 0 <= metrics["f1_weighted"] <= 1
    assert metrics["precision_weighted"] == pytest.approx(0.25)


def test_roc_auc_is_required_when_it_is_the_primary_metric_and_cannot_be_computed():
    class Multiclass:
        classes_ = np.array([0, 1, 2])

        def predict(self, features):
            return np.array([0, 1, 2, 0])

    with pytest.raises(MetricError, match="roc_auc"):
        evaluate_classifier(Multiclass(), np.zeros((4, 1)), np.array([0, 1, 2, 0]), "roc_auc")


def test_empty_holdout_raises():
    with pytest.raises(MetricError, match="empty"):
        evaluate_classifier(ScriptedModel(), np.empty((0, 2)), np.empty((0,)), "accuracy")


def test_length_mismatch_raises():
    with pytest.raises(MetricError, match="different lengths"):
        evaluate_classifier(ScriptedModel(), np.zeros((4, 2)), np.array([0, 1]), "accuracy")


def test_unknown_primary_metric_raises():
    with pytest.raises(MetricError, match="not supported"):
        evaluate_classifier(ScriptedModel(), np.zeros((4, 2)), np.array([0, 1, 0, 0]), "rmse")


def test_prediction_failure_is_wrapped():
    class Broken:
        def predict(self, features):
            raise RuntimeError("estimator exploded")

    with pytest.raises(MetricError, match="estimator exploded"):
        evaluate_classifier(Broken(), np.zeros((2, 1)), np.array([0, 1]), "accuracy")


def test_gradient_boosting_on_separable_data_scores_high():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(160, 2))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=0)
    model.fit(features[:120], labels[:120])
    metrics = evaluate_classifier(model, features[120:], labels[120:], "f1_weighted")
    assert metrics["f1_weighted"] > 0.8
    assert metrics["roc_auc"] > 0.8
