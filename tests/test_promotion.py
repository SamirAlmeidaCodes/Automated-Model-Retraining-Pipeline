"""Promotion side effects: artifacts, history, and the retraining cycle."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.evaluation.comparison import PromotionDecision
from app.evaluation.metrics import evaluate_classifier
from app.registry.artifacts import load_bundle, reconcile_production_artifact
from app.registry.promotion import persist_candidate
from app.storage.database import configure_database, init_db, session_scope
from app.storage.repository import get_production, list_runs, list_versions
from app.training.cycle import run_retraining_cycle
from tests.helpers import ConstantClassifier


def _decision(promote: bool, candidate: float, baseline: float | None, reason: str) -> PromotionDecision:
    return PromotionDecision(promote, reason, candidate, baseline)


def _persist(settings, session, *, promote: bool, candidate: float, baseline: float | None, label: int = 0):
    metrics = {
        "accuracy": candidate,
        "precision_weighted": candidate,
        "recall_weighted": candidate,
        "f1_weighted": candidate,
    }
    baseline_metrics = None if baseline is None else {"f1_weighted": baseline}
    decision = _decision(
        promote,
        candidate,
        baseline,
        "promote" if promote else "keep",
    )
    if baseline_metrics is None and not promote:
        raise AssertionError("A first model has to be promoted in this helper.")
    return persist_candidate(
        session,
        settings=settings,
        estimator=ConstantClassifier(label),
        feature_names=["x1", "x2"],
        label_map={0: "low", 1: "high"},
        metrics=metrics,
        decision=decision,
        algorithm="gradient_boosting",
        hyperparams={"n_estimators": 1},
        training_rows=10,
        test_rows=4,
        data_watermark=14,
        primary_metric="f1_weighted",
    )


def test_first_candidate_becomes_production_and_stores_metrics(database, settings):
    with session_scope() as session:
        row = _persist(settings, session, promote=True, candidate=0.91, baseline=None)
        version = row.version
        created_at = row.created_at
    artifact = Path(settings.models_dir) / "versions" / f"{version}.joblib"
    bundle = load_bundle(artifact)
    production = load_bundle(Path(settings.models_dir) / "production.joblib")

    assert version == "v0001"
    assert created_at is not None
    assert bundle["metrics"]["f1_weighted"] == pytest.approx(0.91)
    assert bundle["version"] == version
    assert production["version"] == version
    with session_scope() as session:
        current = get_production(session)
        assert current.version == version
        assert current.is_production is True
        assert current.status == "production"
        assert current.primary_metric_value == pytest.approx(0.91)


def test_worse_and_tied_candidates_do_not_replace_production(database, settings):
    with session_scope() as session:
        first = _persist(settings, session, promote=True, candidate=0.9, baseline=None, label=1)
        first_version = first.version
    with session_scope() as session:
        worse = _persist(settings, session, promote=False, candidate=0.4, baseline=0.9, label=0)
        worse_version = worse.version
    with session_scope() as session:
        tied = _persist(settings, session, promote=False, candidate=0.9, baseline=0.9, label=0)
        tied_version = tied.version

    production = load_bundle(Path(settings.models_dir) / "production.joblib")
    assert production["version"] == first_version
    assert (Path(settings.models_dir) / "versions" / f"{worse_version}.joblib").exists()
    assert (Path(settings.models_dir) / "versions" / f"{tied_version}.joblib").exists()
    with session_scope() as session:
        current = get_production(session)
        versions = list_versions(session)
        assert current.version == first_version
        assert {row.version: row.status for row in versions} == {
            first_version: "production",
            worse_version: "rejected",
            tied_version: "rejected",
        }


def test_better_candidate_archives_the_previous_production_model(database, settings):
    with session_scope() as session:
        first = _persist(settings, session, promote=True, candidate=0.8, baseline=None)
        first_version = first.version
    with session_scope() as session:
        rejected = _persist(settings, session, promote=False, candidate=0.7, baseline=0.8)
        rejected_version = rejected.version
    with session_scope() as session:
        promoted = _persist(settings, session, promote=True, candidate=0.95, baseline=0.8)
        promoted_version = promoted.version

    assert load_bundle(Path(settings.models_dir) / "production.joblib")["version"] == promoted_version
    with session_scope() as session:
        statuses = {row.version: (row.status, row.is_production, row.promoted) for row in list_versions(session)}
    assert statuses[first_version] == ("archived", False, True)
    assert statuses[rejected_version] == ("rejected", False, False)
    assert statuses[promoted_version] == ("production", True, True)


def test_reconcile_restores_the_registered_production_artifact(database, settings):
    with session_scope() as session:
        first = _persist(settings, session, promote=True, candidate=0.8, baseline=None)
        first_version = first.version
    with session_scope() as session:
        second = _persist(settings, session, promote=True, candidate=0.99, baseline=0.8)
        second_version = second.version

    production_path = Path(settings.models_dir) / "production.joblib"
    stale = Path(settings.models_dir) / "versions" / f"{first_version}.joblib"
    production_path.write_bytes(stale.read_bytes())
    assert load_bundle(production_path)["version"] == first_version

    reconcile_production_artifact(settings)
    assert load_bundle(production_path)["version"] == second_version


def test_cycle_promotes_the_first_model_and_skips_when_nothing_new_arrived(database, settings):
    first = run_retraining_cycle(settings, trigger="manual", force=True)
    assert first.status == "promoted"
    assert first.version == "v0001"
    assert first.primary_metric == "f1_weighted"
    assert first.primary_metric_value is not None

    second = run_retraining_cycle(settings, trigger="manual", force=False)
    assert second.status == "skipped"
    assert "No new observations" in second.message

    bundle = joblib.load(Path(settings.models_dir) / "production.joblib")
    assert bundle["version"] == first.version
    assert bundle["metrics"]["f1_weighted"] == pytest.approx(first.primary_metric_value)
    with session_scope() as session:
        versions = list_versions(session)
        runs = list_runs(session)
    assert len(versions) == 1
    assert versions[0].created_at is not None
    assert {run.status for run in runs} == {"promoted", "skipped"}


def test_cycle_rejects_a_worse_candidate_and_promotes_a_better_one(database, settings, monkeypatch):
    first = run_retraining_cycle(settings, trigger="manual", force=True)
    assert first.status == "promoted"
    calls = {"n": 0}
    real_evaluate = evaluate_classifier

    def worse_candidate(model, features, labels, primary):
        metrics = dict(real_evaluate(model, features, labels, primary))
        metrics[primary] = 0.10 if calls["n"] == 0 else 0.90
        calls["n"] += 1
        return metrics

    monkeypatch.setattr("app.training.cycle.evaluate_classifier", worse_candidate)
    rejected = run_retraining_cycle(settings, trigger="manual", force=True)
    assert rejected.status == "rejected"
    assert rejected.primary_metric_value == pytest.approx(0.10)
    assert rejected.baseline_primary_metric_value == pytest.approx(0.90)
    with session_scope() as session:
        assert get_production(session).version == first.version

    calls["n"] = 0

    def better_candidate(model, features, labels, primary):
        metrics = dict(real_evaluate(model, features, labels, primary))
        metrics[primary] = 0.99 if calls["n"] == 0 else 0.40
        calls["n"] += 1
        return metrics

    monkeypatch.setattr("app.training.cycle.evaluate_classifier", better_candidate)
    promoted = run_retraining_cycle(settings, trigger="manual", force=True)
    assert promoted.status == "promoted"
    assert promoted.primary_metric_value == pytest.approx(0.99)
    with session_scope() as session:
        current = get_production(session)
        history = list_versions(session)
    assert current.version == promoted.version
    assert current.version != first.version
    assert len(history) == 3
    assert load_bundle(Path(settings.models_dir) / "production.joblib")["version"] == promoted.version


def test_publish_failure_keeps_the_current_production_model(database, settings, monkeypatch):
    first = run_retraining_cycle(settings, trigger="manual", force=True)
    assert first.status == "promoted"
    calls = {"n": 0}

    def better_candidate(model, features, labels, primary):
        metrics = dict(evaluate_classifier(model, features, labels, primary))
        metrics[primary] = 0.99 if calls["n"] == 0 else 0.20
        calls["n"] += 1
        return metrics

    def broken_publish(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("app.training.cycle.evaluate_classifier", better_candidate)
    monkeypatch.setattr("app.registry.promotion.publish_production", broken_publish)
    failed = run_retraining_cycle(settings, trigger="manual", force=True)

    assert failed.status == "failed"
    assert "disk full" in failed.message
    assert load_bundle(Path(settings.models_dir) / "production.joblib")["version"] == first.version
    with session_scope() as session:
        assert get_production(session).version == first.version
        assert len(list_versions(session)) == 1
        assert list_runs(session)[0].status == "failed"


def test_incoming_batch_is_ingested_and_a_bad_file_is_isolated(database, settings):
    incoming = Path(settings.incoming_dir)
    pd.DataFrame({"x1": [2.0, -2.0], "x2": [0.2, -0.1], "target": [1, 0]}).to_csv(incoming / "batch.csv", index=False)
    (incoming / "bad.csv").write_text("not,the,schema\n1,2,3\n", encoding="utf-8")

    result = run_retraining_cycle(settings, trigger="manual")

    assert result.status == "promoted"
    assert result.ingested_rows == 2
    assert result.new_rows >= 2
    assert not (incoming / "batch.csv").exists()
    assert not (incoming / "bad.csv").exists()
    assert list((incoming / "processed").glob("*batch.csv"))
    assert list((incoming / "failed").glob("*bad.csv"))


def test_missing_schema_is_recorded_as_a_failed_run(settings):
    configure_database(settings.database_url)
    init_db()
    result = run_retraining_cycle(settings, trigger="manual", force=True)
    assert result.status == "failed"
    assert "schema" in result.message.lower()
    with session_scope() as session:
        runs = list_runs(session)
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].finished_at is not None
