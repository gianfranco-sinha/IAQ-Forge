"""Tier 2: PreprocessingReport, pipeline validation, full orchestrate e2e."""
import json
import re

import pytest

import app.builtin_profiles  # noqa: F401
from app.config import settings
from app.exceptions import IAQError
from training.pipeline import (
    IssueSeverity,
    PipelineError,
    PipelineResult,
    PipelineState,
    PreprocessingReport,
    TrainingPipeline,
)
from training.data_sources import SyntheticSource


# ── TestPreprocessingReport ──────────────────────────────────────────────

class TestPreprocessingReport:
    def test_empty_report(self):
        r = PreprocessingReport()
        assert not r.has_errors
        assert r.errors == []
        assert r.warnings == []

    def test_add_error(self):
        r = PreprocessingReport()
        r.add(IssueSeverity.ERROR, "ingestion", "bad data")
        assert r.has_errors
        assert len(r.errors) == 1
        assert r.errors[0].message == "bad data"

    def test_add_warning(self):
        r = PreprocessingReport()
        r.add(IssueSeverity.WARNING, "scaling", "outliers found")
        assert len(r.warnings) == 1
        assert not r.has_errors

    def test_add_info(self):
        r = PreprocessingReport()
        r.add(IssueSeverity.INFO, "windowing", "200 windows created")
        assert len(r.issues) == 1
        assert r.errors == []
        assert r.warnings == []

    def test_mixed_issues(self):
        r = PreprocessingReport()
        r.add(IssueSeverity.ERROR, "a", "err")
        r.add(IssueSeverity.WARNING, "b", "warn1")
        r.add(IssueSeverity.WARNING, "c", "warn2")
        r.add(IssueSeverity.INFO, "d", "info")
        assert len(r.errors) == 1
        assert len(r.warnings) == 2
        assert len(r.issues) == 4

    def test_log_summary_no_issues(self):
        r = PreprocessingReport()
        r.log_summary()  # should not raise

    def test_log_summary_with_issues(self):
        r = PreprocessingReport()
        r.add(IssueSeverity.ERROR, "stage", "msg", rows_affected=10)
        r.add(IssueSeverity.WARNING, "stage", "msg2")
        r.log_summary()  # should not raise


# ── TestPipelineValidation ───────────────────────────────────────────────

class TestPipelineValidation:
    def test_invalid_model_type_raises(self):
        with pytest.raises(IAQError, match="Unsupported model type"):
            TrainingPipeline(source=SyntheticSource(100), model_type="xgboost")

    def test_output_dir_outside_base_raises(self, patched_models_base):
        with pytest.raises(IAQError, match="output_dir must be under"):
            TrainingPipeline(
                source=SyntheticSource(100),
                model_type="mlp",
                output_dir="/tmp/evil",
            )

    def test_output_dir_under_base_ok(self, patched_models_base):
        out = str(patched_models_base / "mlp")
        pipeline = TrainingPipeline(
            source=SyntheticSource(100),
            model_type="mlp",
            output_dir=out,
        )
        assert pipeline._output_dir.name == "mlp"

    def test_state_starts_idle(self):
        pipeline = TrainingPipeline(
            source=SyntheticSource(100), model_type="mlp"
        )
        assert pipeline.state == PipelineState.IDLE

    def test_resume_default_false(self):
        pipeline = TrainingPipeline(
            source=SyntheticSource(100), model_type="mlp"
        )
        assert pipeline._resume is False


# ── TestPipelineE2E ──────────────────────────────────────────────────────

class TestPipelineE2E:
    @pytest.fixture(autouse=True)
    def _setup(self, patched_models_base, monkeypatch):
        """Isolate all pipeline e2e tests to tmp_path with controlled config."""
        self.base = patched_models_base
        # Use default config so build_model uses consistent window_size/num_features
        test_cfg = settings._get_default_model_config()
        # Ensure mlp window_size matches what we pass (5)
        test_cfg["mlp"]["window_size"] = 5
        test_cfg["mlp"]["num_features"] = 15
        test_cfg["global"]["num_features"] = 15
        monkeypatch.setattr(settings, "_model_config_cache", test_cfg)

    def _run_pipeline(self, model_type="mlp", **overrides):
        kwargs = dict(epochs=2, window_size=5, min_samples=50)
        kwargs.update(overrides)
        pipeline = TrainingPipeline(
            source=SyntheticSource(200),
            model_type=model_type,
            **kwargs,
        )
        return pipeline, pipeline.orchestrate()

    def test_orchestrate_success(self):
        _, result = self._run_pipeline()
        assert isinstance(result, PipelineResult)

    def test_result_has_metrics(self):
        _, result = self._run_pipeline()
        for key in ("mae", "rmse", "r2"):
            assert key in result.metrics

    def test_result_has_version(self):
        _, result = self._run_pipeline()
        assert result.version
        assert re.match(r"mlp-\d+\.\d+\.\d+", result.version)

    def test_result_has_merkle_hash(self):
        _, result = self._run_pipeline()
        assert result.merkle_root_hash
        assert len(result.merkle_root_hash) >= 16  # hex string

    def test_result_model_dir_exists(self):
        _, result = self._run_pipeline()
        assert result.model_dir.is_dir()

    def test_artifacts_saved(self):
        _, result = self._run_pipeline()
        d = result.model_dir
        for name in ("model.pt", "config.json", "feature_scaler.pkl", "target_scaler.pkl"):
            assert (d / name).exists(), f"Missing artifact: {name}"

    def test_config_json_has_version(self):
        _, result = self._run_pipeline()
        cfg = json.loads((result.model_dir / "config.json").read_text())
        assert "version" in cfg
        assert "schema_fingerprint" in cfg

    def test_stage_results_count(self):
        _, result = self._run_pipeline()
        assert len(result.stage_results) == 9  # SOURCE_ACCESS through SAVING

    def test_stage_callback_fires(self):
        calls = []
        pipeline = TrainingPipeline(
            source=SyntheticSource(200),
            model_type="mlp",
            epochs=2, window_size=5, min_samples=50,
        )
        pipeline.on_stage_complete(lambda state, res: calls.append(state))
        pipeline.orchestrate()
        assert len(calls) == 9

    def test_stage_enter_callback_fires(self):
        calls = []
        pipeline = TrainingPipeline(
            source=SyntheticSource(200),
            model_type="mlp",
            epochs=2, window_size=5, min_samples=50,
        )
        pipeline.on_stage_enter(lambda state, res: calls.append(state))
        pipeline.orchestrate()
        assert len(calls) == 9

    def test_error_callback_fires(self):
        errors = []
        pipeline = TrainingPipeline(
            source=SyntheticSource(200),
            model_type="mlp",
            epochs=2, window_size=5, min_samples=999999,  # too high
        )
        pipeline.on_error(lambda info: errors.append(info))
        with pytest.raises(PipelineError):
            pipeline.orchestrate()
        assert len(errors) == 1

    def test_pipeline_result_not_interrupted(self):
        _, result = self._run_pipeline()
        assert result.interrupted is False

    def test_collect_run_params(self):
        pipeline = TrainingPipeline(
            source=SyntheticSource(200),
            model_type="mlp",
            epochs=2, window_size=5, min_samples=50,
        )
        params = pipeline.collect_run_params()
        assert params["model_type"] == "mlp"
        assert params["epochs"] == 2
        assert params["sensor_type"] == "bme680"
