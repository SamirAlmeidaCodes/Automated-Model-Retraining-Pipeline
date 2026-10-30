"""Tests for the stable holdout and estimator construction."""

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from app.config import Settings
from app.exceptions import DataError
from app.training.split import stable_train_test_split
from app.training.trainer import build_estimator, estimator_params


def _frame(ids):
    return pd.DataFrame(
        {
            "id": ids,
            "x1": [0.1] * len(ids),
            "target": [index % 2 for index in range(len(ids))],
        }
    )


def test_stable_split_keeps_the_same_ids_in_the_holdout():
    frame = _frame(list(range(1, 21)))
    train, test = stable_train_test_split(
        frame,
        id_column="id",
        target_column="target",
        test_modulus=5,
        random_state=1,
    )
    assert sorted(test["id"].tolist()) == [5, 10, 15, 20]
    assert sorted(train["id"].tolist()) == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
    again_train, again_test = stable_train_test_split(
        frame.sample(frac=1, random_state=3),
        id_column="id",
        target_column="target",
        test_modulus=5,
        random_state=1,
    )
    assert sorted(again_test["id"].tolist()) == [5, 10, 15, 20]
    assert sorted(again_train["id"].tolist()) == sorted(train["id"].tolist())


def test_small_table_uses_a_seeded_fallback_split():
    frame = _frame(list(range(1, 9)))
    train, test = stable_train_test_split(
        frame,
        id_column="id",
        target_column="target",
        test_modulus=100,
        random_state=4,
    )
    assert len(train) + len(test) == 8
    assert len(test) >= 1
    assert set(train["id"]).isdisjoint(set(test["id"]))


def test_too_few_rows_raise():
    frame = _frame([1, 2, 3, 4])
    with pytest.raises(DataError, match="at least 8"):
        stable_train_test_split(
            frame,
            id_column="id",
            target_column="target",
            test_modulus=5,
            random_state=1,
        )


def test_build_estimator_honors_the_algorithm(settings: Settings):
    boosting = build_estimator(settings)
    assert isinstance(boosting, GradientBoostingClassifier)
    forest_settings = settings.model_copy(update={"algorithm": "random_forest", "n_estimators": 12})
    forest = build_estimator(forest_settings)
    assert isinstance(forest, RandomForestClassifier)
    params = estimator_params(forest)
    assert params["n_estimators"] == 12
    assert params["random_state"] == settings.random_state
