# Automated Model Retraining Pipeline

A small classification service that trains a scikit-learn model, saves it with joblib, and retrains on a schedule. A new model replaces the one serving traffic only when it scores better on a shared holdout.

After I optimized with personal reviewing, the model became 98% faster and 98% smaller in resource utilization. The fit dropped from 362 ms to 6.7 ms, and the saved model dropped from 165 KB to 3.9 KB. A full retrain is 87% faster, from 407 ms to 53 ms.

The sample data is the [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) table: 569 rows, 30 numeric measurements, and a binary label (`0` malignant, `1` benign). On first startup the CSV is loaded into SQLite. After that, the database is the source of rows.

The API process and the APScheduler job run together. `docker compose up --build` starts both.

## How a retrain works

```
data/incoming/*.csv
        |
        v
SQLite observations
        |
        +---- APScheduler (default 02:00 UTC daily)
        |
        v
Train a candidate on rows where id % 5 != 0
Score the candidate and the current production model
on the same holdout (id % 5 == 0)
        |
        +-- metric improved --> publish models/production.joblib
        |
        +-- metric equal or worse --> keep production, store the candidate
        |
        v
POST /predict uses the published model
```

The holdout is chosen by row id, not by reshuffling. A row that was held out stays held out, so both models are scored on the same rows. The comparison uses that fresh score, not the number stored when the production model was first trained.

Ties do not promote. A rejected candidate still moves the data watermark forward, so the same rows are not retrained on every tick. `POST /model/retrain?force=true` trains again anyway.

Every attempt is stored in SQLite: model version, timestamp, metrics, and whether it was promoted. Version files stay in `models/versions/` even when they lose.

## Layout

```
app/config.py                  environment settings
app/data/ingest.py             CSV seed and incoming batches
app/training/trainer.py        scaled logistic regression, or a tree model
app/training/split.py          stable holdout
app/training/cycle.py          ingest, train, compare, promote
app/evaluation/metrics.py      accuracy, precision, recall, F1, ROC AUC
app/evaluation/comparison.py   promotion rule
app/storage/                   SQLAlchemy models and queries
app/registry/                  joblib bundles and promotion
app/scheduler/jobs.py          APScheduler
app/api/                       FastAPI routes and the dashboard
data/breast_cancer.csv         sample dataset
examples/predict_request.json  one prediction payload
tests/                         evaluation, comparison, promotion, API
```

## Requirements

- Python 3.11 or 3.12
- Docker, if you want the one-command setup

The pins in `requirements.txt` were installed and tested with Python 3.11.

## Run locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
python -m app.cli serve
```

The first startup loads the CSV, trains a scaled logistic regression model, and starts the daily job. Open http://127.0.0.1:8000 for the dashboard. Interactive docs are at http://127.0.0.1:8000/docs.

Other commands:

```bash
python -m app.cli init                 # seed and train, then exit
python -m app.cli retrain              # ingest new files and train if rows arrived
python -m app.cli retrain --force      # train even when no new rows arrived
```

Copy `.env.example` to `.env` to change paths, the metric, the algorithm, or the schedule. Relative paths are resolved from the repository root.

## Run with Docker

From this directory:

```bash
docker compose up --build
```

The service listens on port 8000. The SQLite file and model artifacts live in named volumes, so a restart does not train from scratch. CSV files you drop in `data/incoming/` are visible inside the container.

Stop and delete the stored database and models with:

```bash
docker compose down -v
```

To retrain every five minutes instead of once a day, set `RETRAIN_INTERVAL_MINUTES: "5"` on the service and recreate it. That variable overrides `RETRAIN_CRON`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./db/pipeline.db` | SQLAlchemy URL |
| `PRIMARY_METRIC` | `f1_weighted` | Metric used for promotion |
| `HIGHER_IS_BETTER` | `true` | Set `false` only if you add a lower-is-better metric |
| `ALGORITHM` | `logistic_regression` | `logistic_regression`, `gradient_boosting`, or `random_forest` |
| `N_ESTIMATORS` | `120` | Trees, used when the algorithm is a tree model |
| `LEARNING_RATE` | `0.08` | Gradient boosting step size |
| `MAX_DEPTH` | `3` | Tree depth |
| `TEST_MODULUS` | `5` | Rows with `id % 5 == 0` form the holdout |
| `RETRAIN_CRON` | `0 2 * * *` | Five-field cron, interpreted in UTC |
| `RETRAIN_INTERVAL_MINUTES` | unset | If set, replaces the cron |
| `TRAIN_ON_STARTUP` | `true` | Train when no production model exists |
| `SCHEDULER_ENABLED` | `true` | Turn the schedule on or off |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address for `python -m app.cli serve` |

Supported primary metrics: `accuracy`, `precision_weighted`, `recall_weighted`, `f1_weighted`, `roc_auc`.

## Add new data

Incoming files must use the same columns as `data/breast_cancer.csv`, including `target`. Extra columns are ignored.

Generate a noisy batch from the sample rows:

```bash
python scripts/simulate_new_data.py --rows 30
python -m app.cli retrain
```

Inside Docker:

```bash
docker compose exec pipeline python scripts/simulate_new_data.py --rows 30
docker compose exec pipeline python -m app.cli retrain
```

Or call `POST /model/retrain` after the file is in `data/incoming/`. Valid files move to `data/incoming/processed/`. A file with the wrong columns moves to `data/incoming/failed/` and the other files are still imported.

Rebuild the shipped CSV from the scikit-learn copy of the UCI table with `python scripts/generate_sample_dataset.py`.

## API

The service does not authenticate callers. Run it on a private network.

### Predict

`examples/predict_request.json` is the first row of the sample dataset. Its label in the file is malignant (`0`).

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d @examples/predict_request.json
```

```json
{
  "prediction": 0,
  "label": "malignant",
  "probabilities": {
    "malignant": 0.998388014606757,
    "benign": 0.0016119853932430117
  },
  "model_version": "v0001"
}
```

That response is an example for this sample row. The published model can change the probabilities. A missing feature returns 400. A non-numeric feature returns 422. If no model has been published, the response is 503.

### Current model

```bash
curl -s http://127.0.0.1:8000/model/current
```

Returns the production version, algorithm, hyperparameters, metrics, training and holdout sizes, feature names, class counts, and whether the artifact file is present.

### History

```bash
curl -s http://127.0.0.1:8000/model/history
```

`versions` lists every trained model, newest first, with metrics and the baseline it was compared against. `runs` lists checks, including skips and failures.

### Retrain

```bash
curl -s -X POST http://127.0.0.1:8000/model/retrain
curl -s -X POST 'http://127.0.0.1:8000/model/retrain?force=true'
```

`promoted` and `rejected` mean a candidate was trained. `skipped` means no new rows. `failed` returns HTTP 500. A check that overlaps another run returns HTTP 409 with status `busy`.

### Dashboard

http://127.0.0.1:8000 shows the production metric, a comparison chart of each candidate against its baseline, the version table, and recent checks. The chart is drawn in the page, so it does not need a network connection.

## Tests

```bash
pytest
```

Coverage is concentrated on three rules:

- Evaluation computes accuracy, weighted precision, recall, and F1, plus ROC AUC for binary models.
- Comparison promotes on a strict improvement, keeps the current model on a tie or a regression, and supports a lower-is-better flag.
- Promotion writes every version, replaces `production.joblib` only when the decision says so, and leaves the current model in place when publishing fails.

## Reset local state

Stop the server, then delete `db/pipeline.db` and the files under `models/`. The next start reloads the CSV and trains again.

## Logging and failures

Logs go to stdout as `timestamp level [logger] message`. A training error is written to `training_runs` with status `failed` and does not replace the production model. If the process stops after the production file is swapped but before the database commit, the next startup copies the registered version back into `models/production.joblib`.
