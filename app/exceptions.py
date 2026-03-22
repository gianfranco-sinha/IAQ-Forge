"""Domain exception hierarchy for iaq4j.

Every subclass carries a ``DomainErrorCode`` so that error handlers can
map failures to structured responses without string-matching messages.
"""

from app.schemas import DomainErrorCode


class IAQError(Exception):
    """Base class for all iaq4j domain errors."""

    code: DomainErrorCode  # overridden by each subclass

    def __init__(self, message: str, suggestion: str = None):
        super().__init__(message)
        self.suggestion = suggestion


class NoDataError(IAQError):
    """Data source returned zero usable rows."""

    code = DomainErrorCode.NO_DATA


class InsufficientDataError(IAQError):
    """Not enough data for the requested operation (e.g. window too large)."""

    code = DomainErrorCode.INSUFFICIENT_DATA


class SchemaMismatchError(IAQError):
    """Model artifact schema does not match current config."""

    code = DomainErrorCode.SCHEMA_MISMATCH


class TrainingDivergedError(IAQError):
    """Training loss became NaN/Inf or otherwise diverged."""

    code = DomainErrorCode.TRAINING_DIVERGED


class InfluxUnreachableError(IAQError):
    """Cannot connect to InfluxDB."""

    code = DomainErrorCode.INFLUX_UNREACHABLE


class CheckpointNotFoundError(IAQError):
    """Requested training checkpoint does not exist."""

    code = DomainErrorCode.CHECKPOINT_NOT_FOUND


class StaleConfigError(IAQError):
    """Configuration is outdated relative to loaded artifacts."""

    code = DomainErrorCode.STALE_CONFIG


class NegativeR2Error(IAQError):
    """Model evaluation produced negative R² (worse than mean predictor)."""

    code = DomainErrorCode.NEGATIVE_R2


class ConfigurationError(IAQError):
    """Required configuration or setup step was not completed."""

    code = DomainErrorCode.CONFIGURATION
