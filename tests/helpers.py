"""Picklable stand-in estimators used by promotion tests."""

from __future__ import annotations

import numpy as np


class ConstantClassifier:
    """Predict one class for every row. Defined at module scope so joblib can load it."""

    def __init__(self, label: int = 0):
        self.label = int(label)
        self.classes_ = np.array([0, 1])

    def fit(self, features, target=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, features):
        return np.full(len(features), self.label)

    def predict_proba(self, features):
        probabilities = np.zeros((len(features), 2))
        probabilities[:, self.label] = 1.0
        return probabilities
