FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY examples ./examples

RUN mkdir -p /app/db /app/models /app/data/incoming

EXPOSE 8000

# One process serves the API and runs the APScheduler retraining job.
CMD ["python", "-m", "app.cli", "serve"]
