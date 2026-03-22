"""Tests for pipeline ingestion validation checks.

Covers: timezone normalization, chronological ordering, duplicate timestamps,
NaT handling, sampling interval, unit conversion (declared + heuristic),
NaN/outlier handling, and source-level unit overrides.
"""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import app.builtin_profiles  # noqa: F401
from app.exceptions import IAQError
from app.builtin_profiles import BME680Profile
from app.profiles import get_iaq_standard, get_sensor_profile
from training.data_sources import CSVDataSource, DataSource, InfluxDBSource, SyntheticSource
from training.pipeline import (
    IssueSeverity,
    PipelineState,
    PreprocessingReport,
    TrainingPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bme680_df(
    n=200,
    tz="UTC",
    freq="3s",
    start="2026-01-01",
    seed=42,
    index=None,
):
    """Build a realistic BME680 DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    if index is None:
        index = pd.date_range(start, periods=n, freq=freq, tz=tz)
    data = {
        "temperature": rng.uniform(20, 26, n),
        "rel_humidity": rng.uniform(40, 60, n),
        "pressure": rng.uniform(1000, 1020, n),
        "voc_resistance": rng.uniform(10000, 200000, n),
        "iaq": rng.uniform(0, 300, n),
        "iaq_accuracy": np.full(n, 3),
    }
    return pd.DataFrame(data, index=index)


class StubSource(DataSource):
    """DataSource that returns a pre-built DataFrame."""

    def __init__(self, df, unit_overrides=None):
        self._df = df
        self._unit_overrides = unit_overrides or {}

    @property
    def name(self):
        return "StubSource"

    @property
    def provided_units(self):
        return self._unit_overrides

    def validate(self):
        pass

    def fetch(self):
        return self._df.copy()


def _run_ingestion(df, unit_overrides=None):
    """Run only the INGESTION stage, return the pipeline and its report."""
    source = StubSource(df, unit_overrides=unit_overrides)
    pipeline = TrainingPipeline(
        source=source,
        model_type="mlp",
        epochs=2,
        window_size=5,
        min_samples=10,
    )
    # Drive SOURCE_ACCESS then INGESTION
    sa_result = pipeline._do_source_access()
    result = pipeline._do_ingestion()
    return pipeline, result


def _report_messages(pipeline):
    """Extract all report issue messages."""
    return [issue.message for issue in pipeline._report.issues]


def _report_issues_by_severity(pipeline, severity):
    return [i for i in pipeline._report.issues if i.severity == severity]


# ===================================================================
# Timezone normalization
# ===================================================================

class TestTimezoneNormalization:
    def test_naive_timestamps_localized_to_utc(self):
        df = _make_bme680_df(tz=None)
        assert df.index.tz is None
        pipeline, _ = _run_ingestion(df)
        assert str(pipeline._df.index.tz) == "UTC"
        msgs = _report_messages(pipeline)
        assert any("localized to UTC" in m for m in msgs)

    def test_non_utc_converted_to_utc(self):
        df = _make_bme680_df(tz="US/Eastern")
        pipeline, _ = _run_ingestion(df)
        assert str(pipeline._df.index.tz) == "UTC"
        msgs = _report_messages(pipeline)
        assert any("Converted timestamps from" in m and "to UTC" in m for m in msgs)

    def test_already_utc_no_report(self):
        df = _make_bme680_df(tz="UTC")
        pipeline, _ = _run_ingestion(df)
        assert str(pipeline._df.index.tz) == "UTC"
        msgs = _report_messages(pipeline)
        # No tz-related issue
        assert not any("localized" in m.lower() or "converted timestamps" in m.lower() for m in msgs)


# ===================================================================
# Sampling interval
# ===================================================================

class TestSamplingInterval:
    def test_normal_interval_no_warning(self):
        """BME680 expects 3s; data at 3s should be fine."""
        df = _make_bme680_df(freq="3s")
        pipeline, _ = _run_ingestion(df)
        msgs = _report_messages(pipeline)
        assert not any("sampling interval" in m.lower() for m in msgs)

    def test_high_interval_warns(self):
        """Interval 10x expected → warning."""
        df = _make_bme680_df(freq="60s")  # 60s vs expected 3s → ratio 20
        pipeline, _ = _run_ingestion(df)
        msgs = _report_messages(pipeline)
        assert any("sampling interval" in m.lower() for m in msgs)

    def test_very_low_interval_warns(self):
        """Interval 0.1x expected → warning."""
        df = _make_bme680_df(freq="200ms")  # 0.2s vs 3s → ratio ~0.067
        pipeline, _ = _run_ingestion(df)
        msgs = _report_messages(pipeline)
        assert any("sampling interval" in m.lower() for m in msgs)


# ===================================================================
# Chronological ordering
# ===================================================================

class TestChronologicalOrdering:
    def test_out_of_order_sorted(self):
        df = _make_bme680_df()
        # Reverse the order
        df = df.iloc[::-1]
        assert not df.index.is_monotonic_increasing
        pipeline, _ = _run_ingestion(df)
        assert pipeline._df.index.is_monotonic_increasing
        msgs = _report_messages(pipeline)
        assert any("chronological order" in m.lower() for m in msgs)

    def test_in_order_no_warning(self):
        df = _make_bme680_df()
        pipeline, _ = _run_ingestion(df)
        msgs = _report_messages(pipeline)
        assert not any("chronological order" in m.lower() for m in msgs)


# ===================================================================
# Duplicate timestamps
# ===================================================================

class TestDuplicateTimestamps:
    def test_duplicates_removed(self):
        df = _make_bme680_df(n=200)
        # Create duplicates by repeating first 5 rows
        df = pd.concat([df, df.iloc[:5]])
        pipeline, _ = _run_ingestion(df)
        assert not pipeline._df.index.duplicated().any()
        msgs = _report_messages(pipeline)
        assert any("duplicate timestamp" in m.lower() for m in msgs)

    def test_no_duplicates_no_warning(self):
        df = _make_bme680_df()
        pipeline, _ = _run_ingestion(df)
        msgs = _report_messages(pipeline)
        assert not any("duplicate" in m.lower() for m in msgs)


# ===================================================================
# NaT timestamps
# ===================================================================

class TestNaTTimestamps:
    def test_nat_removed_with_error(self):
        df = _make_bme680_df(n=200)
        # Inject NaT into 3 positions
        idx = df.index.to_list()
        idx[10] = pd.NaT
        idx[20] = pd.NaT
        idx[30] = pd.NaT
        df.index = pd.DatetimeIndex(idx, tz="UTC")
        pipeline, _ = _run_ingestion(df)
        assert not pipeline._df.index.isna().any()
        errors = _report_issues_by_severity(pipeline, IssueSeverity.ERROR)
        assert any("NaT" in e.message or "missing timestamp" in e.message for e in errors)
        # Should record 3 rows affected
        nat_issue = next(e for e in errors if "NaT" in e.message or "missing timestamp" in e.message)
        assert nat_issue.rows_affected == 3


# ===================================================================
# NaN value handling
# ===================================================================

class TestNaNHandling:
    def test_nan_rows_dropped(self):
        df = _make_bme680_df(n=200)
        df.iloc[5, 0] = np.nan
        df.iloc[10, 2] = np.nan
        pipeline, _ = _run_ingestion(df)
        assert not pipeline._df.isna().any().any()
        msgs = _report_messages(pipeline)
        assert any("nan" in m.lower() for m in msgs)


# ===================================================================
# Outlier handling
# ===================================================================

class TestOutlierHandling:
    def test_out_of_range_values_dropped(self):
        df = _make_bme680_df(n=200)
        # Temperature valid range: [-40, 85] — put some way out
        df.iloc[0, df.columns.get_loc("temperature")] = 100.0
        df.iloc[1, df.columns.get_loc("temperature")] = -50.0
        pipeline, _ = _run_ingestion(df)
        msgs = _report_messages(pipeline)
        assert any("outside valid range" in m and "temperature" in m for m in msgs)
        # Those rows should be gone
        assert len(pipeline._df) < 200

    def test_in_range_values_kept(self):
        df = _make_bme680_df(n=200)
        pipeline, _ = _run_ingestion(df)
        # All values within range — no outlier warning
        msgs = _report_messages(pipeline)
        assert not any("outside valid range" in m for m in msgs)
        assert len(pipeline._df) == 200


# ===================================================================
# Unit conversion — declared (source provides non-canonical)
# ===================================================================

class TestUnitConversionDeclared:
    def test_source_declares_kohm_converts_to_ohm(self):
        """Source says voc_resistance is in kΩ — pipeline auto-converts."""
        df = _make_bme680_df(n=200)
        # Set voc_resistance to kΩ values (10-500 kΩ)
        df["voc_resistance"] = np.random.default_rng(42).uniform(10, 500, 200)
        pipeline, _ = _run_ingestion(df, unit_overrides={"voc_resistance": "kΩ"})
        # After conversion, values should be in Ω (10k-500k range)
        assert pipeline._df["voc_resistance"].median() > 5000
        msgs = _report_messages(pipeline)
        assert any("converted from kΩ" in m for m in msgs)

    def test_source_declares_fahrenheit_converts_to_celsius(self):
        """Source says temperature is in °F — pipeline auto-converts."""
        df = _make_bme680_df(n=200)
        # Set temperature to °F values
        df["temperature"] = np.random.default_rng(42).uniform(68, 80, 200)
        pipeline, _ = _run_ingestion(df, unit_overrides={"temperature": "°F"})
        # After conversion, values should be in °C (20-26.7 range)
        assert pipeline._df["temperature"].median() < 30
        msgs = _report_messages(pipeline)
        assert any("converted from °F" in m for m in msgs)

    def test_canonical_unit_no_conversion(self):
        """Source declares canonical unit — no conversion."""
        df = _make_bme680_df(n=200)
        pipeline, _ = _run_ingestion(df, unit_overrides={"voc_resistance": "Ω"})
        # No conversion message
        msgs = _report_messages(pipeline)
        assert not any("converted from" in m and "voc_resistance" in m for m in msgs)


# ===================================================================
# Unit heuristic detection (no declared units)
# ===================================================================

class TestUnitHeuristicDetection:
    def test_median_outside_range_suggests_alternate_unit(self):
        """Data outside canonical range + no alternate fits → error in report."""
        df = _make_bme680_df(n=200)
        # Set voc_resistance to 5M-10M Ω — above 2MΩ max, no alternate unit
        # conversion can rescue (kΩ→5G, MΩ→5T), all rows dropped as outliers
        df["voc_resistance"] = np.random.default_rng(42).uniform(5e6, 1e7, 200)
        source = StubSource(df)
        pipeline = TrainingPipeline(
            source=source, model_type="mlp", epochs=2, window_size=5, min_samples=10,
        )
        pipeline._do_source_access()
        with pytest.raises(IAQError):
            pipeline._do_ingestion()
        msgs = _report_messages(pipeline)
        assert any("possible unit mismatch" in m.lower() and "voc_resistance" in m for m in msgs)

    def test_median_inside_range_no_warning(self):
        """Data in canonical range — no heuristic warning."""
        df = _make_bme680_df(n=200)
        pipeline, _ = _run_ingestion(df)
        msgs = _report_messages(pipeline)
        assert not any("possible unit mismatch" in m.lower() for m in msgs)


# ===================================================================
# Unit auto-detection and conversion (no declared units)
# ===================================================================

class TestUnitAutoDetection:
    # ── voc_resistance (Ω, alternates: kΩ, MΩ) ──────────────────────

    def test_resistance_in_kohm_auto_converted(self):
        """voc_resistance in kΩ range [10, 200] → auto-converted to Ω."""
        df = _make_bme680_df(n=1200)
        df["voc_resistance"] = np.random.default_rng(42).uniform(10, 200, 1200)
        pipeline, _ = _run_ingestion(df)
        sample = pipeline._df["voc_resistance"].sample(n=1000, random_state=42)
        avg = sample.mean()
        assert 1000 <= avg <= 200000, f"Average {avg:.1f} Ω outside valid range"
        msgs = _report_messages(pipeline)
        assert any("auto-detected" in m and "kΩ" in m for m in msgs)
        warnings = _report_issues_by_severity(pipeline, IssueSeverity.WARNING)
        assert any("voc_resistance" in w.message and "auto-detected" in w.message for w in warnings)

    def test_resistance_in_ohm_no_conversion(self):
        """voc_resistance already in Ω range [10000, 200000] → no conversion."""
        df = _make_bme680_df(n=1200)
        pipeline, _ = _run_ingestion(df)
        sample = pipeline._df["voc_resistance"].sample(n=1000, random_state=42)
        avg = sample.mean()
        assert 1000 <= avg <= 200000, f"Average {avg:.1f} Ω outside valid range"
        msgs = _report_messages(pipeline)
        assert not any("auto-detected" in m and "voc_resistance" in m for m in msgs)

    def test_resistance_no_unit_fits_reports_error(self):
        """voc_resistance way above max — no unit fits → ERROR in report."""
        df = _make_bme680_df(n=200)
        # Raw: 5M-10M > 2M max; kΩ: 5G-10G; MΩ: 5T-10T — no match
        df["voc_resistance"] = np.random.default_rng(42).uniform(5e6, 1e7, 200)
        source = StubSource(df)
        pipeline = TrainingPipeline(
            source=source, model_type="mlp", epochs=2, window_size=5, min_samples=10,
        )
        pipeline._do_source_access()
        with pytest.raises(IAQError):
            pipeline._do_ingestion()
        errors = _report_issues_by_severity(pipeline, IssueSeverity.ERROR)
        assert any(
            "voc_resistance" in e.message and "no alternate unit" in e.message
            for e in errors
        )

    # ── temperature (°C, alternates: °F, K) ──────────────────────────

    def test_temperature_in_fahrenheit_auto_converted(self):
        """temperature in °F range [68, 176] → auto-converted to °C."""
        df = _make_bme680_df(n=1200)
        df["temperature"] = np.random.default_rng(42).uniform(68, 176, 1200)
        pipeline, _ = _run_ingestion(df)
        sample = pipeline._df["temperature"].sample(n=1000, random_state=42)
        avg = sample.mean()
        assert -40 <= avg <= 85, f"Average {avg:.1f} °C outside valid range"
        msgs = _report_messages(pipeline)
        assert any("auto-detected" in m and "°F" in m for m in msgs)

    def test_temperature_in_celsius_no_conversion(self):
        """temperature already in °C range [20, 26] → no conversion."""
        df = _make_bme680_df(n=1200)
        pipeline, _ = _run_ingestion(df)
        sample = pipeline._df["temperature"].sample(n=1000, random_state=42)
        avg = sample.mean()
        assert -40 <= avg <= 85, f"Average {avg:.1f} °C outside valid range"
        msgs = _report_messages(pipeline)
        assert not any("auto-detected" in m and "temperature" in m for m in msgs)

    def test_temperature_no_unit_fits_reports_error(self):
        """temperature in [500, 600] — no unit fits → ERROR in report."""
        df = _make_bme680_df(n=200)
        # °F→°C: 260-316 (above 85), K→°C: 227-327 (above 85) — no match
        df["temperature"] = np.random.default_rng(42).uniform(500, 600, 200)
        source = StubSource(df)
        pipeline = TrainingPipeline(
            source=source, model_type="mlp", epochs=2, window_size=5, min_samples=10,
        )
        pipeline._do_source_access()
        with pytest.raises(IAQError):
            pipeline._do_ingestion()
        errors = _report_issues_by_severity(pipeline, IssueSeverity.ERROR)
        assert any(
            "temperature" in e.message and "no alternate unit" in e.message
            for e in errors
        )

    # ── pressure (hPa, alternates: Pa, kPa, mbar, inHg) ─────────────

    def test_pressure_in_pascals_auto_converted(self):
        """pressure in Pa range [30000, 110000] → auto-converted to hPa."""
        df = _make_bme680_df(n=1200)
        df["pressure"] = np.random.default_rng(42).uniform(30000, 110000, 1200)
        pipeline, _ = _run_ingestion(df)
        sample = pipeline._df["pressure"].sample(n=1000, random_state=42)
        avg = sample.mean()
        assert 300 <= avg <= 1100, f"Average {avg:.1f} hPa outside valid range"
        msgs = _report_messages(pipeline)
        assert any("auto-detected" in m and "Pa" in m for m in msgs)

    def test_pressure_in_kpa_auto_converted(self):
        """pressure in kPa range [30, 110] → auto-converted to hPa."""
        df = _make_bme680_df(n=1200)
        df["pressure"] = np.random.default_rng(42).uniform(30, 110, 1200)
        pipeline, _ = _run_ingestion(df)
        sample = pipeline._df["pressure"].sample(n=1000, random_state=42)
        avg = sample.mean()
        assert 300 <= avg <= 1100, f"Average {avg:.1f} hPa outside valid range"
        msgs = _report_messages(pipeline)
        assert any("auto-detected" in m and "kPa" in m for m in msgs)

    def test_pressure_in_hpa_no_conversion(self):
        """pressure already in hPa range [1000, 1020] → no conversion."""
        df = _make_bme680_df(n=1200)
        pipeline, _ = _run_ingestion(df)
        sample = pipeline._df["pressure"].sample(n=1000, random_state=42)
        avg = sample.mean()
        assert 300 <= avg <= 1100, f"Average {avg:.1f} hPa outside valid range"
        msgs = _report_messages(pipeline)
        assert not any("auto-detected" in m and "pressure" in m for m in msgs)

    def test_pressure_no_unit_fits_reports_error(self):
        """pressure in [0.1, 0.5] — no unit fits → ERROR in report."""
        df = _make_bme680_df(n=200)
        # Pa: 0.001-0.005, kPa: 1-5, mbar: 0.1-0.5, inHg: 3.4-16.9 — all below 300
        df["pressure"] = np.random.default_rng(42).uniform(0.1, 0.5, 200)
        source = StubSource(df)
        pipeline = TrainingPipeline(
            source=source, model_type="mlp", epochs=2, window_size=5, min_samples=10,
        )
        pipeline._do_source_access()
        with pytest.raises(IAQError):
            pipeline._do_ingestion()
        errors = _report_issues_by_severity(pipeline, IssueSeverity.ERROR)
        assert any(
            "pressure" in e.message and "no alternate unit" in e.message
            for e in errors
        )

    # ── cross-feature isolation ──────────────────────────────────────

    def test_other_features_not_affected(self):
        """All features in canonical range — no conversion attempted."""
        df = _make_bme680_df(n=200)
        original_temps = df["temperature"].values.copy()
        pipeline, _ = _run_ingestion(df)
        np.testing.assert_array_almost_equal(
            pipeline._df["temperature"].values, original_temps
        )
        msgs = _report_messages(pipeline)
        assert not any("auto-detected" in m for m in msgs)


# ===================================================================
# Source-level unit overrides (integration with DataSource subclasses)
# ===================================================================

class TestSourceUnitOverrides:
    def test_csv_data_source_provided_units(self):
        source = CSVDataSource("dummy.csv", unit_overrides={"voc_resistance": "kΩ"})
        assert source.provided_units == {"voc_resistance": "kΩ"}

    def test_csv_data_source_no_overrides(self):
        source = CSVDataSource("dummy.csv")
        assert source.provided_units == {}

    def test_influxdb_source_provided_units(self):
        source = InfluxDBSource(unit_overrides={"temperature": "°F"})
        assert source.provided_units == {"temperature": "°F"}

    def test_influxdb_source_no_overrides(self):
        source = InfluxDBSource()
        assert source.provided_units == {}

    def test_synthetic_source_no_overrides(self):
        source = SyntheticSource()
        assert source.provided_units == {}

    def test_base_class_default_empty(self):
        """DataSource ABC default returns empty dict."""
        source = StubSource(_make_bme680_df())
        assert source.provided_units == {}


# ===================================================================
# No DatetimeIndex — timestamps checks skipped gracefully
# ===================================================================

class TestNoDatetimeIndex:
    def test_rangeindex_skips_timestamp_checks(self):
        """When DataFrame has no DatetimeIndex, tz/ordering checks are skipped."""
        df = _make_bme680_df(n=200)
        df = df.reset_index(drop=True)  # RangeIndex instead of DatetimeIndex
        pipeline, _ = _run_ingestion(df)
        msgs = _report_messages(pipeline)
        assert any("no datetimeindex" in m.lower() for m in msgs)
        # No tz or ordering warnings
        assert not any("localized" in m.lower() for m in msgs)
        assert not any("chronological" in m.lower() for m in msgs)


# ===================================================================
# PreprocessingReport
# ===================================================================

class TestPreprocessingReport:
    def test_add_and_retrieve(self):
        report = PreprocessingReport()
        report.add(IssueSeverity.ERROR, "ingestion", "bad data", rows_affected=5)
        report.add(IssueSeverity.WARNING, "ingestion", "minor issue")
        report.add(IssueSeverity.INFO, "ingestion", "fyi")
        assert len(report.issues) == 3
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert report.has_errors

    def test_to_dict(self):
        report = PreprocessingReport()
        report.add(IssueSeverity.WARNING, "ingestion", "test warning")
        d = report.to_dict()
        assert d["summary"]["warnings"] == 1
        assert d["summary"]["total"] == 1

    def test_empty_report(self):
        report = PreprocessingReport()
        assert not report.has_errors
        assert len(report.issues) == 0
