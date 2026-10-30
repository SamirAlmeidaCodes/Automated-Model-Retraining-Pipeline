"""Command line entry points for seeding, retraining, and serving."""

from __future__ import annotations

import argparse
import json

import uvicorn

from app.bootstrap import bootstrap, prepare_storage
from app.config import get_settings
from app.logging_config import setup_logging
from app.storage.database import session_scope
from app.storage.repository import get_production
from app.training.cycle import run_retraining_cycle


def main(argv: list[str] | None = None) -> int:
    """Run ``serve``, ``init``, or ``retrain``."""
    parser = argparse.ArgumentParser(description="Automated model retraining pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="Run the API and the retraining scheduler")
    sub.add_parser("init", help="Create the database, load the sample dataset, and train if needed")
    retrain = sub.add_parser("retrain", help="Ingest new CSV files and run one retraining check")
    retrain.add_argument(
        "--force",
        action="store_true",
        help="Train even when no new rows have arrived since the last run",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)

    if args.command == "serve":
        uvicorn.run(
            "app.api.main:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )
        return 0

    if args.command == "init":
        inserted = prepare_storage(settings)
        print(f"Seeded {inserted} observations.")
        with session_scope() as session:
            production = get_production(session)
        if production is None and settings.train_on_startup:
            result = run_retraining_cycle(settings, trigger="startup", force=True)
            print(json.dumps(result.to_dict(), indent=2))
            return 1 if result.status == "failed" else 0
        if production is not None:
            print(f"Production model {production.version} is already available.")
        return 0

    prepare_storage(settings)
    result = run_retraining_cycle(settings, trigger="manual", force=args.force)
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.status in {"failed", "busy"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
