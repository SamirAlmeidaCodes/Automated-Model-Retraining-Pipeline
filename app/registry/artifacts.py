"""Save, load, and publish joblib model bundles."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

import joblib

from app.clock import utcnow
from app.config import Settings
from app.exceptions import ModelNotReadyError
from app.storage.database import session_scope
from app.storage.repository import get_production

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1
_REQUIRED_KEYS = {"bundle_version", "version", "estimator", "feature_names", "label_map", "metrics"}
_cache_lock = threading.Lock()
_cache: dict = {"path": None, "mtime": None, "bundle": None}


def save_bundle(path: Path, *, estimator, feature_names, label_map, version: str, algorithm: str, metrics: dict, primary_metric: str) -> None:
    """Atomically write a model bundle."""
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "version": version,
        "estimator": estimator,
        "feature_names": list(feature_names),
        "label_map": {int(key): str(value) for key, value in label_map.items()},
        "algorithm": algorithm,
        "metrics": dict(metrics),
        "primary_metric": primary_metric,
        "created_at": utcnow().isoformat(),
    }
    atomic_joblib_dump(payload, path)


def atomic_joblib_dump(payload: dict, path: Path) -> None:
    """Write a joblib file via a temporary sibling, then replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def load_bundle(path: Path) -> dict:
    """Load a bundle and check that the expected keys are present."""
    if not path.exists():
        raise ModelNotReadyError(f"Model artifact not found: {path.name}.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        logger.error("Could not read model artifact at %s", path)
        raise ModelNotReadyError(f"Could not read model artifact {path.name}: {exc}") from exc
    if not isinstance(bundle, dict) or not _REQUIRED_KEYS <= set(bundle):
        raise ModelNotReadyError(f"Model artifact {path.name} is not a pipeline bundle.")
    return bundle


def load_production_bundle(models_dir: Path) -> dict:
    """Return the published production bundle, using a small mtime cache."""
    path = Path(models_dir) / "production.joblib"
    if not path.exists():
        raise ModelNotReadyError("No production model is available.")
    with _cache_lock:
        mtime = path.stat().st_mtime
        if _cache["bundle"] is not None and _cache["path"] == str(path) and _cache["mtime"] == mtime:
            return _cache["bundle"]
        bundle = load_bundle(path)
        _cache["path"] = str(path)
        _cache["mtime"] = mtime
        _cache["bundle"] = bundle
        return bundle


def invalidate_cache() -> None:
    """Drop the cached production bundle."""
    with _cache_lock:
        _cache["path"] = None
        _cache["mtime"] = None
        _cache["bundle"] = None


def resolve_artifact(models_dir: Path | str, relative_path: str) -> Path:
    """Resolve a stored artifact path against the models directory."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return Path(models_dir) / path


def publish_production(models_dir: Path, version_artifact: Path) -> None:
    """Copy a versioned artifact onto ``production.joblib`` atomically."""
    if not version_artifact.exists():
        raise ModelNotReadyError(f"Cannot publish missing artifact {version_artifact.name}.")
    target = Path(models_dir) / "production.joblib"
    temporary = Path(models_dir) / "production.joblib.tmp"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(version_artifact, temporary)
    os.replace(temporary, target)
    invalidate_cache()


def snapshot_production(models_dir: Path) -> bytes | None:
    """Read the current production file so a failed transaction can restore it."""
    path = Path(models_dir) / "production.joblib"
    if not path.exists():
        return None
    return path.read_bytes()


def restore_production(models_dir: Path, payload: bytes | None) -> None:
    """Put the production file back to a previous snapshot."""
    path = Path(models_dir) / "production.joblib"
    if payload is None:
        if path.exists():
            path.unlink()
        invalidate_cache()
        return
    temporary = path.with_suffix(".joblib.restore")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    invalidate_cache()


def reconcile_production_artifact(settings: Settings) -> None:
    """Make ``production.joblib`` match the version marked production in the database.

    The file is replaced before the database commit. A crash in between is
    repaired here on the next startup by copying the registered artifact back.
    """
    models_dir = Path(settings.models_dir)
    production_path = models_dir / "production.joblib"
    with session_scope() as session:
        row = get_production(session)
        if row is None:
            return
        registered = row.version
        version_path = resolve_artifact(models_dir, row.artifact_path)
    if not version_path.exists():
        logger.error("Production model %s is registered but %s is missing.", registered, version_path)
        return
    needs_restore = not production_path.exists()
    if not needs_restore:
        try:
            bundle = load_bundle(production_path)
            needs_restore = bundle.get("version") != registered
        except ModelNotReadyError:
            needs_restore = True
    if needs_restore:
        logger.warning("Restoring production artifact to registered version %s.", registered)
        publish_production(models_dir, version_path)
