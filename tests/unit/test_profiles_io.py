"""Tier 2: Profile registry resolution (get_sensor_profile / get_iaq_standard)."""
import pytest

import app.builtin_profiles  # noqa: F401
from app.builtin_profiles import BME680Profile, BSECStandard
from app.config import settings
from app.profiles import get_iaq_standard, get_sensor_profile


class TestGetSensorProfile:
    def test_default_bme680(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {
            "sensor": {"type": "bme680"}
        })
        profile = get_sensor_profile()
        assert isinstance(profile, BME680Profile)
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_unknown_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {
            "sensor": {"type": "nonexistent_sensor"}
        })
        with pytest.raises(ValueError, match="Unknown sensor profile"):
            get_sensor_profile()
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_missing_sensor_config_defaults(self, monkeypatch):
        """Returns BME680Profile when 'sensor' key absent from config."""
        monkeypatch.setattr(settings, "_model_config_cache", {})
        profile = get_sensor_profile()
        assert isinstance(profile, BME680Profile)
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_missing_type_key_defaults(self, monkeypatch):
        """Returns BME680Profile when sensor dict has no 'type' key."""
        monkeypatch.setattr(settings, "_model_config_cache", {
            "sensor": {"features": ["temperature"]}
        })
        profile = get_sensor_profile()
        assert isinstance(profile, BME680Profile)
        monkeypatch.setattr(settings, "_model_config_cache", None)


class TestGetIAQStandard:
    def test_default_bsec(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {
            "iaq_standard": {"type": "bsec"}
        })
        standard = get_iaq_standard()
        assert isinstance(standard, BSECStandard)
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_unknown_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {
            "iaq_standard": {"type": "nonexistent_standard"}
        })
        with pytest.raises(ValueError, match="Unknown IAQ standard"):
            get_iaq_standard()
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_missing_standard_config_defaults(self, monkeypatch):
        """Returns BSECStandard when 'iaq_standard' key absent."""
        monkeypatch.setattr(settings, "_model_config_cache", {})
        standard = get_iaq_standard()
        assert isinstance(standard, BSECStandard)
        monkeypatch.setattr(settings, "_model_config_cache", None)

    def test_missing_type_key_defaults(self, monkeypatch):
        """Returns BSECStandard when iaq_standard dict has no 'type' key."""
        monkeypatch.setattr(settings, "_model_config_cache", {
            "iaq_standard": {"scale": "custom"}
        })
        standard = get_iaq_standard()
        assert isinstance(standard, BSECStandard)
        monkeypatch.setattr(settings, "_model_config_cache", None)
