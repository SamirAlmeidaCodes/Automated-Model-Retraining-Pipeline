"""Shared fixtures. Each test gets its own SQLite file and model directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.bootstrap import prepare_storage
from app.config import Settings


def write_synthetic_dataset(csv_path: Path, meta_path: Path, n: int = 120, seed: int = 7) -> None:
    """Write a separable two-feature classification problem."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = (x1 + 0.5 * x2 > 0).astype(int)
    pd.DataFrame({"x1": x1, "x2": x2, "target": target}).to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "dataset_name": "Synthetic separable",
                "target_column": "target",
                "label_map": {"0": "low", "1": "high"},
                "description": "Two-feature dataset used by the tests.",
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    incoming = data_dir / "incoming"
    incoming.mkdir()
    csv_path = data_dir / "observations.csv"
    meta_path = data_dir / "dataset_meta.json"
    write_synthetic_dataset(csv_path, meta_path)
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'pipeline.db'}",
        data_csv_path=str(csv_path),
        data_meta_path=str(meta_path),
        incoming_dir=str(incoming),
        models_dir=str(tmp_path / "models"),
        primary_metric="f1_weighted",
        higher_is_better=True,
        algorithm="gradient_boosting",
        n_estimators=25,
        learning_rate=0.1,
        max_depth=2,
        test_modulus=5,
        random_state=42,
        retrain_cron="0 2 * * *",
        retrain_interval_minutes=None,
        train_on_startup=True,
        scheduler_enabled=False,
        log_level="WARNING",
    )


@pytest.fixture
def database(settings: Settings) -> Settings:
    """Initialize SQLite and seed the synthetic CSV."""
    prepare_storage(settings)
    return settings
