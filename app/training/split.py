"""Holdout split that stays stable as new rows arrive."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from app.exceptions import DataError

logger = logging.getLogger(__name__)


def stable_train_test_split(
    frame: pd.DataFrame,
    *,
    id_column: str,
    target_column: str,
    test_modulus: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by ``id % test_modulus == 0``.

    The same id always lands in the same side of the split, and those test
    rows are never used for training. The candidate and the current production
    model are later scored on this same holdout, so the promotion check is
    not comparing two numbers from different samples.

    A very small table falls back to a seeded random split so unit tests and
    tiny batches can still train.
    """
    if id_column not in frame.columns:
        raise DataError(f"Split frame is missing {id_column!r}.")
    if test_modulus < 2:
        raise DataError("TEST_MODULUS must be at least 2.")

    test_mask = frame[id_column].astype(int) % test_modulus == 0
    train = frame.loc[~test_mask].copy()
    test = frame.loc[test_mask].copy()
    if len(train) >= 5 and len(test) >= 3:
        return train.reset_index(drop=True), test.reset_index(drop=True)

    if len(frame) < 8:
        raise DataError(f"Need at least 8 observations to train and evaluate; found {len(frame)}.")
    logger.warning(
        "Stable holdout was too small (train=%s, test=%s). Using a seeded random split instead.",
        len(train),
        len(test),
    )
    train, test = _random_split(frame, target_column, random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _random_split(frame: pd.DataFrame, target_column: str, random_state: int):
    stratify = None
    counts = frame[target_column].value_counts()
    if frame[target_column].nunique() > 1 and int(counts.min()) >= 2:
        stratify = frame[target_column]
    try:
        return train_test_split(
            frame,
            test_size=0.25,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        logger.warning("Stratified split failed. Splitting without stratification.")
        return train_test_split(frame, test_size=0.25, random_state=random_state)
