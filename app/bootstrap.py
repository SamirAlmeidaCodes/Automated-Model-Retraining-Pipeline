"""Create directories, the database, and the initial production model."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.data.ingest import seed_if_empty
from app.registry.artifacts import reconcile_production_artifact
from app.storage.database import configure_database, init_db, session_scope
from app.storage.repository import get_production
from app.training.cycle import run_retraining_cycle

logger = logging.getLogger(__name__)


def prepare_storage(settings: Settings) -> int:
    """Create folders, initialize SQLite, and seed the sample data if needed.

    Returns the number of observations inserted by seeding.
    """
    Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
    incoming = Path(settings.incoming_dir)
    incoming.mkdir(parents=True, exist_ok=True)
    (incoming / "processed").mkdir(exist_ok=True)
    (incoming / "failed").mkdir(exist_ok=True)
    configure_database(settings.database_url)
    init_db()
    inserted = seed_if_empty(settings)
    reconcile_production_artifact(settings)
    return inserted


def bootstrap(settings: Settings) -> None:
    """Prepare storage and train the first model when production is empty."""
    inserted = prepare_storage(settings)
    logger.info("Dataset ready. Seeded %s new observations.", inserted)
    if not settings.train_on_startup:
        return
    with session_scope() as session:
        production = get_production(session)
    if production is not None:
        logger.info("Production model %s is already available.", production.version)
        return
    logger.info("No production model found. Training the initial model.")
    result = run_retraining_cycle(settings, trigger="startup", force=True)
    logger.info("Initial training finished with status=%s. %s", result.status, result.message)
    if result.status == "failed":
        logger.error("Initial training failed: %s", result.message)
