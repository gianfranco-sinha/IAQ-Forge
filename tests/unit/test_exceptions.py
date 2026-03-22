"""Tests for the IAQError domain exception hierarchy."""

import pytest

from app.exceptions import (
    IAQError,
    NoDataError,
    InsufficientDataError,
    SchemaMismatchError,
    TrainingDivergedError,
    InfluxUnreachableError,
    CheckpointNotFoundError,
    StaleConfigError,
    NegativeR2Error,
    ConfigurationError,
    ServiceUnreachableError,
)
from app.schemas import DomainErrorCode


# ── Hierarchy tests ──────────────────────────────────────────────────────


class TestIAQErrorHierarchy:
    """Every subtype inherits IAQError and Exception."""

    SUBTYPES = [
        NoDataError,
        InsufficientDataError,
        SchemaMismatchError,
        TrainingDivergedError,
        InfluxUnreachableError,
        CheckpointNotFoundError,
        StaleConfigError,
        NegativeR2Error,
        ConfigurationError,
        ServiceUnreachableError,
    ]

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_inherits_iaq_error(self, cls):
        err = cls("test")
        assert isinstance(err, IAQError)

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_inherits_exception(self, cls):
        err = cls("test")
        assert isinstance(err, Exception)

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_catchable_as_iaq_error(self, cls):
        with pytest.raises(IAQError):
            raise cls("test")


# ── DomainErrorCode mapping ─────────────────────────────────────────────


class TestDomainErrorCodeMapping:
    """Each subtype carries the correct DomainErrorCode."""

    EXPECTED = {
        NoDataError: DomainErrorCode.NO_DATA,
        InsufficientDataError: DomainErrorCode.INSUFFICIENT_DATA,
        SchemaMismatchError: DomainErrorCode.SCHEMA_MISMATCH,
        TrainingDivergedError: DomainErrorCode.TRAINING_DIVERGED,
        InfluxUnreachableError: DomainErrorCode.INFLUX_UNREACHABLE,
        CheckpointNotFoundError: DomainErrorCode.CHECKPOINT_NOT_FOUND,
        StaleConfigError: DomainErrorCode.STALE_CONFIG,
        NegativeR2Error: DomainErrorCode.NEGATIVE_R2,
        ConfigurationError: DomainErrorCode.CONFIGURATION,
        ServiceUnreachableError: DomainErrorCode.SERVICE_UNREACHABLE,
    }

    @pytest.mark.parametrize("cls,expected_code", EXPECTED.items())
    def test_code_matches(self, cls, expected_code):
        err = cls("msg")
        assert err.code == expected_code
        assert err.code.value == expected_code.value

    def test_all_domain_codes_covered(self):
        """Every DomainErrorCode has a corresponding exception subtype."""
        covered_codes = {cls.code for cls in self.EXPECTED}
        all_codes = set(DomainErrorCode)
        assert covered_codes == all_codes


# ── Suggestion field ─────────────────────────────────────────────────────


class TestSuggestionField:
    def test_defaults_to_none(self):
        err = NoDataError("no data")
        assert err.suggestion is None

    def test_custom_suggestion(self):
        err = NoDataError("no data", suggestion="Widen time range")
        assert err.suggestion == "Widen time range"

    def test_message_preserved(self):
        err = InsufficientDataError("only 5 rows", suggestion="Get more data")
        assert str(err) == "only 5 rows"
        assert err.suggestion == "Get more data"


# ── Integration: FSM _fire_error populates FailureInfo from IAQError ──────


class TestFailPopulatesFailureInfo:
    def test_iaq_error_populates_code_and_suggestion(self):
        from training.pipeline import TrainingPipeline, PipelineState
        from training.data_sources import SyntheticSource

        pipeline = TrainingPipeline(
            model_type="mlp",
            source=SyntheticSource(num_samples=10),
        )
        err = NoDataError("empty", suggestion="check source")
        info = pipeline._fsm._fire_error(PipelineState.INGESTION, err)
        assert info.error_code == "NO_DATA"
        assert info.suggestion == "check source"

    def test_non_iaq_error_leaves_code_none(self):
        from training.pipeline import TrainingPipeline, PipelineState
        from training.data_sources import SyntheticSource

        pipeline = TrainingPipeline(
            model_type="mlp",
            source=SyntheticSource(num_samples=10),
        )
        err = RuntimeError("boom")
        info = pipeline._fsm._fire_error(PipelineState.TRAINING, err)
        assert info.error_code is None
        assert info.suggestion is None
