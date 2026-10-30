"""FastAPI application that serves predictions and runs the retraining scheduler."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.bootstrap import bootstrap
from app.config import Settings, get_settings
from app.exceptions import DataError, MetricError, ModelNotReadyError
from app.logging_config import setup_logging
from app.scheduler.jobs import build_scheduler, make_retrain_job

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API. Pass ``settings`` in tests; otherwise the environment is used."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        app.state.settings = settings
        logger.info("Starting %s", settings.app_name)
        bootstrap(settings)
        scheduler = None
        if settings.scheduler_enabled:
            scheduler = build_scheduler(settings, make_retrain_job(settings))
            scheduler.start()
            app.state.scheduler = scheduler
            job = scheduler.get_job("retrain")
            logger.info("Scheduler started. Next run: %s", None if job is None else job.next_run_time)
        else:
            app.state.scheduler = None
            logger.info("Scheduler is disabled.")
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)
                logger.info("Scheduler stopped.")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Train a tabular classifier, retrain it on a schedule, and promote "
            "a new version only when it scores better than the current production model."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.scheduler = None
    app.include_router(router)

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready(_request, exc: ModelNotReadyError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(DataError)
    async def data_error(_request, exc: DataError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(MetricError)
    async def metric_error(_request, exc: MetricError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = create_app()
