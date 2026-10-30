"""Scheduler configuration."""

from pathlib import Path

import pytest
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings
from app.scheduler.jobs import build_scheduler, schedule_description


def test_default_schedule_is_a_daily_cron(settings):
    enabled = settings.model_copy(update={"scheduler_enabled": True})
    scheduler = build_scheduler(enabled, lambda: None)
    job = scheduler.get_job("retrain")
    assert job is not None
    assert isinstance(job.trigger, CronTrigger)
    assert scheduler.running is False
    assert schedule_description(enabled) == "cron 0 2 * * * UTC"


def test_interval_overrides_cron(settings):
    custom = settings.model_copy(update={"retrain_interval_minutes": 15, "scheduler_enabled": True})
    scheduler = build_scheduler(custom, lambda: None)
    assert isinstance(scheduler.get_job("retrain").trigger, IntervalTrigger)
    assert schedule_description(custom) == "every 15 minutes"


def test_invalid_cron_is_rejected(settings):
    with pytest.raises(ValidationError):
        Settings(
            database_url=settings.database_url,
            data_csv_path=settings.data_csv_path,
            data_meta_path=settings.data_meta_path,
            incoming_dir=settings.incoming_dir,
            models_dir=settings.models_dir,
            retrain_cron="every day",
        )


def test_disabled_scheduler_label(settings):
    custom = settings.model_copy(update={"scheduler_enabled": False})
    assert schedule_description(custom) == "scheduler off"


def test_relative_paths_are_anchored_at_the_repository():
    settings = Settings(
        database_url="sqlite:///./db/pipeline.db",
        data_csv_path="./data/breast_cancer.csv",
        data_meta_path="./data/dataset_meta.json",
        incoming_dir="./data/incoming",
        models_dir="./models",
    )
    assert Path(settings.data_csv_path) == PROJECT_ROOT / "data" / "breast_cancer.csv"
    assert Path(settings.models_dir) == PROJECT_ROOT / "models"
    assert Path(settings.incoming_dir) == PROJECT_ROOT / "data" / "incoming"
    assert settings.database_url == "sqlite:///" + str(PROJECT_ROOT / "db" / "pipeline.db")
