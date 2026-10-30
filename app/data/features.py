"""Turn a prediction payload into a one-row frame."""

from __future__ import annotations

import math

import pandas as pd

from app.exceptions import DataError


def vectorize_features(features: dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    """Build a one-row frame whose columns follow the model's feature order."""
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise DataError("Missing features: " + ", ".join(missing))
    values: dict[str, list[float]] = {}
    for name in feature_names:
        try:
            number = float(features[name])
        except (TypeError, ValueError) as exc:
            raise DataError(f"Feature {name!r} is not numeric.") from exc
        if not math.isfinite(number):
            raise DataError(f"Feature {name!r} must be a finite number.")
        values[name] = [number]
    return pd.DataFrame(values, columns=list(feature_names))
