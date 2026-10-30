"""Response models for the HTTP API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Feature values for one row. Keys are the dataset's column names."""

    features: dict[str, float] = Field(min_length=1)


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probabilities: dict[str, float] | None = None
    model_version: str


class VersionResponse(BaseModel):
    id: int
    version: str
    created_at: str
    algorithm: str
    hyperparams: dict
    metrics: dict[str, float]
    primary_metric: str
    primary_metric_value: float
    baseline_primary_metric_value: float | None
    delta: float | None
    is_production: bool
    promoted: bool
    status: str
    training_rows: int
    test_rows: int
    data_watermark: int
    notes: str | None
    artifact_path: str


class RunResponse(BaseModel):
    id: int
    started_at: str
    finished_at: str | None
    trigger: str
    status: str
    message: str | None
    model_version_id: int | None
    new_rows: int
    ingested_rows: int


class ClassCount(BaseModel):
    label: str
    count: int


class CurrentModelResponse(VersionResponse):
    dataset_name: str
    description: str | None = None
    feature_names: list[str]
    label_map: dict[str, str]
    observation_count: int
    class_counts: list[ClassCount]
    artifact_available: bool


class HistoryResponse(BaseModel):
    versions: list[VersionResponse]
    runs: list[RunResponse]


class RetrainResponse(BaseModel):
    status: str
    message: str
    version: str | None = None
    primary_metric: str | None = None
    primary_metric_value: float | None = None
    baseline_primary_metric_value: float | None = None
    new_rows: int = 0
    ingested_rows: int = 0
    run_id: int | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str | None = None
    scheduler_enabled: bool
