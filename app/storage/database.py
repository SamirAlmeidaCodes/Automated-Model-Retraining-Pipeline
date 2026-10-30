"""Engine and session lifecycle for the SQLite database."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def configure_database(url: str) -> None:
    """Create the engine used by later sessions. Safe to call again."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _ensure_sqlite_directory(url)
    connect_args = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    _engine = engine
    _SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    logger.debug("Database configured")


def init_db() -> None:
    """Create tables that do not exist yet."""
    from app.storage.models import Base

    Base.metadata.create_all(get_engine())


def get_engine():
    """Return the configured engine."""
    if _engine is None:
        raise RuntimeError("Database is not configured. Call configure_database() first.")
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session and commit it. Roll back when the body raises."""
    if _SessionLocal is None:
        raise RuntimeError("Database is not configured. Call configure_database() first.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _ensure_sqlite_directory(url: str) -> None:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return
    raw = url[len(prefix) :]
    if not raw or raw == ":memory:" or raw.startswith("file:"):
        return
    Path(raw).parent.mkdir(parents=True, exist_ok=True)
