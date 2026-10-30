"""Errors raised by the retraining pipeline."""


class PipelineError(Exception):
    """Base error for expected pipeline failures."""


class DataError(PipelineError):
    """Raised when a dataset file or observation cannot be used."""


class MetricError(PipelineError):
    """Raised when a metric cannot be computed or compared."""


class ModelNotReadyError(PipelineError):
    """Raised when a prediction is requested and no usable model is available."""
