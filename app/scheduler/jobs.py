"""Build the retraining schedule and the job it runs."""

from __future__ import annotations

import logging
from collections.abc import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import Settings
from app.training.cycle import run_retraining_cycle

logger = logging.getLogger(__name__)


def make_retrain_job(settings: Settings) -> Callable[[], None]:
    """Return the scheduled callable bound to ``settings``."""

    def _job() -> None:
        logger.info("Scheduled retraining check started.")
        try:
            result = run_retraining_cycle(settings, trigger="schedule")
        except Exception:
            logger.exception("Scheduled retraining check crashed.")
            return
        logger.info(
            "Scheduled retraining check finished status=%s message=%s",
            result.status,
            result.message,
        )

    return _job


def build_scheduler(settings: Settings, job: Callable[[], None]) -> BackgroundScheduler:
    """Create a scheduler with one retraining job. The caller starts it.

    ``RETRAIN_INTERVAL_MINUTES`` overrides the daily cron when it is set.
    The default cron is 02:00 UTC every day.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    if settings.retrain_interval_minutes:
        trigger = IntervalTrigger(minutes=settings.retrain_interval_minutes, timezone="UTC")
        logger.info("Retraining interval is every %s minutes.", settings.retrain_interval_minutes)
    else:
        minute, hour, day, month, day_of_week = settings.retrain_cron.split()
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone="UTC",
        )
        logger.info("Retraining cron is '%s' UTC.", settings.retrain_cron)
    scheduler.add_job(
        job,
        trigger=trigger,
        id="retrain",
        name="retrain",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60 * 60,
    )
    return scheduler


def schedule_description(settings: Settings) -> str:
    """Return a short label for the active schedule."""
    if not settings.scheduler_enabled:
        return "scheduler off"
    if settings.retrain_interval_minutes:
        minutes = settings.retrain_interval_minutes
        unit = "minute" if minutes == 1 else "minutes"
        return f"every {minutes} {unit}"
    return f"cron {settings.retrain_cron} UTC"
