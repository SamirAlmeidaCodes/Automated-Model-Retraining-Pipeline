"""Construct and fit the configured scikit-learn estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from app.config import Settings
from app.exceptions import DataError


def build_estimator(settings: Settings):
    """Return an unfitted gradient boosting or random forest classifier."""
    if settings.algorithm == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=settings.n_estimators,
            learning_rate=settings.learning_rate,
            max_depth=settings.max_depth,
            random_state=settings.random_state,
        )
    if settings.algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth,
            random_state=settings.random_state,
            n_jobs=1,
        )
    raise DataError(f"Unsupported algorithm {settings.algorithm!r}.")


def fit_estimator(estimator, train_df: pd.DataFrame, feature_names: list[str], target_column: str):
    """Fit ``estimator`` on the training rows and return it."""
    features = train_df.loc[:, list(feature_names)]
    target = train_df[target_column].astype(int)
    if target.nunique() < 2:
        raise DataError("Training data contains a single class. Add rows from another class.")
    estimator.fit(features, target)
    return estimator


def estimator_params(estimator) -> dict:
    """Return constructor arguments that can be stored as JSON."""
    cleaned: dict = {}
    for key, value in estimator.get_params(deep=False).items():
        if isinstance(value, np.generic):
            value = value.item()
        if value == "deprecated":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
    return cleaned
