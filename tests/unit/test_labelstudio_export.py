"""Tests for Label Studio exporter — before/after cleanse views."""

from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from integrations.label_studio.exporter import (
    LabelStudioExporter,
    cleanse_dataframe,
)


@pytest.fixture
def bme680_profile():
    import app.builtin_profiles  # noqa: F401
    from app.builtin_profiles import BME680Profile
    return BME680Profile()


@pytest.fixture
def bsec_standard():
    import app.builtin_profiles  # noqa: F401
    from app.profiles import get_iaq_standard
    return get_iaq_standard()


@pytest.fixture
def raw_df():
    """Raw DataFrame with a mix of valid, out-of-range, and NaN rows."""
    data = {
        "temperature": [22.5, 23.0, 24.0, np.nan, 25.0],
        "rel_humidity": [45.0, 50.0, 55.0, 60.0, 65.0],
        "pressure": [1013.2, 1012.0, 1011.0, 1010.0, 1009.0],
        "voc_resistance": [150000, 5000000, 80000, 100000, 120000],
        "iaq": [75.0, 80.0, 60.0, 90.0, 70.0],
        "iaq_accuracy": [3, 3, 3, 3, 1],
    }
    timestamps = pd.date_range("2026-01-22T15:00:00Z", periods=5, freq="3s")
    return pd.DataFrame(data, index=timestamps)


@pytest.fixture
def exporter():
    """LabelStudioExporter with explicit params (no config lookup needed)."""
    with patch("integrations.config.get_integration_config") as mock_int, \
         patch("app.config.settings") as mock_settings:
        mock_int.return_value = {}
        mock_settings.get_label_studio_config.return_value = {
            "url": "http://localhost:8080",
            "api_key": "test-key",
        }
        exp = LabelStudioExporter(
            project_id=1,
            url="http://localhost:8080",
            api_key="test-key",
            batch_size=2,
        )
    return exp


class TestCleanseDataframe:
    def test_removes_out_of_range_rows(self, raw_df, bme680_profile):
        """Row with voc_resistance=5000000 should be dropped."""
        clean_df, reasons, _ = cleanse_dataframe(raw_df, bme680_profile, "iaq")

        assert len(clean_df) < len(raw_df)
        # Row index 1 (voc_resistance=5M) should be dropped
        assert raw_df.index[1] not in clean_df.index
        assert "voc_resistance" in reasons.loc[raw_df.index[1]]

    def test_removes_nan_rows(self, raw_df, bme680_profile):
        """Row with NaN temperature should be dropped."""
        clean_df, reasons, _ = cleanse_dataframe(raw_df, bme680_profile, "iaq")

        assert raw_df.index[3] not in clean_df.index
        assert "NaN" in reasons.loc[raw_df.index[3]]

    def test_removes_low_quality_rows(self, raw_df, bme680_profile):
        """Row with iaq_accuracy=1 should be dropped (quality_min=2)."""
        clean_df, reasons, _ = cleanse_dataframe(raw_df, bme680_profile, "iaq")

        assert raw_df.index[4] not in clean_df.index
        assert "iaq_accuracy" in reasons.loc[raw_df.index[4]]

    def test_keeps_valid_rows(self, raw_df, bme680_profile):
        """Rows 0 and 2 should survive cleansing."""
        clean_df, _, _ = cleanse_dataframe(raw_df, bme680_profile, "iaq")

        assert raw_df.index[0] in clean_df.index
        assert raw_df.index[2] in clean_df.index

    def test_ingest_report_counts(self, raw_df, bme680_profile):
        """Ingest report should contain correct row counts."""
        _, _, report = cleanse_dataframe(raw_df, bme680_profile, "iaq")

        assert report["raw_rows"] == 5
        assert report["dropped_rows"] == 3  # out-of-range, NaN, low quality
        assert report["clean_rows"] == 2
        assert "discontinuities" in report
        assert "gap_threshold_seconds" in report

    def test_discontinuity_detection(self, bme680_profile):
        """Gaps > threshold should be counted as discontinuities."""
        # 3 rows with a 60s gap between row 1 and 2
        data = {
            "temperature": [22.0, 23.0, 24.0],
            "rel_humidity": [50.0, 51.0, 52.0],
            "pressure": [1013.0, 1013.0, 1013.0],
            "voc_resistance": [100000, 110000, 120000],
            "iaq": [75.0, 80.0, 85.0],
            "iaq_accuracy": [3, 3, 3],
        }
        timestamps = pd.to_datetime([
            "2026-01-02T10:00:00",
            "2026-01-02T10:00:03",
            "2026-01-02T10:01:03",  # 60s gap
        ])
        df = pd.DataFrame(data, index=timestamps)

        _, _, report = cleanse_dataframe(df, bme680_profile, "iaq",
                                         gap_threshold_seconds=30.0)

        assert report["discontinuities"] == 1
        assert len(report["gaps"]) == 1
        assert report["gaps"][0]["gap_seconds"] == 60.0

    def test_no_discontinuities_below_threshold(self, bme680_profile):
        """Gaps below threshold should not be counted."""
        data = {
            "temperature": [22.0, 23.0],
            "rel_humidity": [50.0, 51.0],
            "pressure": [1013.0, 1013.0],
            "voc_resistance": [100000, 110000],
            "iaq": [75.0, 80.0],
            "iaq_accuracy": [3, 3],
        }
        timestamps = pd.to_datetime([
            "2026-01-02T10:00:00",
            "2026-01-02T10:00:03",  # 3s gap, below 30s threshold
        ])
        df = pd.DataFrame(data, index=timestamps)

        _, _, report = cleanse_dataframe(df, bme680_profile, "iaq",
                                         gap_threshold_seconds=30.0)

        assert report["discontinuities"] == 0


class TestBuildBeforeAfterTasks:
    def test_kept_rows_have_raw_and_clean(self, exporter, raw_df, bme680_profile, bsec_standard):
        # Use only the first valid row
        clean_df = raw_df.iloc[[0]].copy()

        tasks = exporter.build_before_after_tasks(
            raw_df, clean_df, bme680_profile, bsec_standard,
            include_dropped=False,
        )

        assert len(tasks) == 1
        data = tasks[0]["data"]
        assert data["cleanse_status"] == "kept"
        assert "raw_temperature" in data
        assert "clean_temperature" in data
        assert data["raw_temperature"] == 22.5
        assert data["clean_temperature"] == 22.5
        assert "timestamp" in data

    def test_dropped_rows_have_raw_only(self, exporter, raw_df, bme680_profile, bsec_standard):
        # Clean DF is empty — all rows dropped
        clean_df = raw_df.iloc[:0].copy()
        drop_reasons = pd.Series(
            {raw_df.index[0]: "test reason"},
            name="cleanse_reason",
        )

        tasks = exporter.build_before_after_tasks(
            raw_df, clean_df, bme680_profile, bsec_standard,
            drop_reasons=drop_reasons,
            include_dropped=True,
        )

        assert len(tasks) == len(raw_df)
        dropped_task = tasks[0]
        data = dropped_task["data"]
        assert data["cleanse_status"] == "dropped"
        assert "raw_temperature" in data
        assert "clean_temperature" not in data
        assert data["cleanse_reason"] == "test reason"

    def test_exclude_dropped(self, exporter, raw_df, bme680_profile, bsec_standard):
        clean_df = raw_df.iloc[[0]].copy()

        tasks = exporter.build_before_after_tasks(
            raw_df, clean_df, bme680_profile, bsec_standard,
            include_dropped=False,
        )

        # Only kept rows
        assert len(tasks) == 1
        assert all(t["data"]["cleanse_status"] == "kept" for t in tasks)


class TestExportBatching:
    def test_chunks_by_batch_size(self, exporter):
        """5 tasks with batch_size=2 should produce 3 POST calls."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            tasks = [{"data": {"x": i}} for i in range(5)]
            stats = exporter.export_tasks(tasks)

        assert stats["batches"] == 3
        assert stats["created"] == 5
        assert mock_post.call_count == 3

    def test_dry_run_no_post(self, exporter):
        """Dry run should not make any HTTP calls."""
        tasks = [{"data": {"x": i}} for i in range(5)]

        with patch("requests.post") as mock_post:
            stats = exporter.export_tasks(tasks, dry_run=True)

        assert stats["dry_run"] is True
        assert stats["total"] == 5
        assert stats["batches"] == 3  # ceil(5/2)
        mock_post.assert_not_called()


class TestValidateConnection:
    def test_connection_failure(self, exporter):
        """ConnectionError on health check should raise."""
        import requests as real_requests

        with patch("requests.get", side_effect=real_requests.exceptions.ConnectionError("refused")):
            with pytest.raises(ConnectionError, match="Cannot reach"):
                exporter.validate()

    def test_missing_project_id_raises(self):
        """No project_id should raise ConfigurationError."""
        from app.exceptions import ConfigurationError

        with patch("integrations.config.get_integration_config") as mock_int, \
             patch("app.config.settings") as mock_settings:
            mock_int.return_value = {}
            mock_settings.get_label_studio_config.return_value = {
                "url": "http://localhost:8080",
                "api_key": "test-key",
            }
            exp = LabelStudioExporter(
                url="http://localhost:8080",
                api_key="test-key",
            )

        with pytest.raises(ConfigurationError, match="project_id"):
            exp.validate()
