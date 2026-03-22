"""Tests for app/standards.py — YAML-driven IAQ standards."""
import math

import pytest

from app.standards import (
    StandardDef,
    YAMLStandard,
    get_standard_def,
    list_standards,
    register_yaml_standards,
)


@pytest.fixture(autouse=True)
def _ensure_registered():
    """Ensure YAML standards are registered for every test."""
    register_yaml_standards()


# ── Loading all 4 standards ───────────────────────────────────────────────


class TestYAMLLoading:
    def test_bsec_loads(self):
        d = get_standard_def("bsec")
        assert d.name == "bsec"
        assert d.target_column == "iaq"
        assert d.scale_range == (0.0, 500.0)
        assert d.higher_is_worse is True
        assert len(d.categories) == 5

    def test_epa_aqi_loads(self):
        d = get_standard_def("epa_aqi")
        assert d.name == "epa_aqi"
        assert d.target_column == "aqi"
        assert d.scale_range == (0.0, 500.0)
        assert d.higher_is_worse is True
        assert len(d.categories) == 6

    def test_iaqi_loads(self):
        d = get_standard_def("iaqi")
        assert d.name == "iaqi"
        assert d.target_column == "iaqi"
        assert d.scale_range == (0.0, 100.0)
        assert d.higher_is_worse is False
        assert len(d.categories) == 5

    def test_reset_air_loads(self):
        d = get_standard_def("reset_air")
        assert d.name == "reset_air"
        assert d.target_column == "reset_air"
        assert d.scale_range == (0.0, 100.0)
        assert d.higher_is_worse is False
        assert len(d.categories) == 4

    def test_unknown_raises_configuration_error(self):
        from app.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown IAQ standard"):
            get_standard_def("nonexistent")


# ── YAMLStandard wrapping ────────────────────────────────────────────────


class TestYAMLStandard:
    def test_properties(self):
        s = YAMLStandard(get_standard_def("bsec"))
        assert s.name == "bsec"
        assert s.target_column == "iaq"
        assert s.scale_range == (0.0, 500.0)
        assert s.higher_is_worse is True
        assert s.description == "Bosch BSEC indoor air quality index"

    def test_clamp(self):
        s = YAMLStandard(get_standard_def("bsec"))
        assert s.clamp(-10) == 0.0
        assert s.clamp(600) == 500.0
        assert s.clamp(250) == 250.0

    def test_categorize_bsec(self):
        s = YAMLStandard(get_standard_def("bsec"))
        assert s.categorize(25) == "Excellent"
        assert s.categorize(75) == "Good"
        assert s.categorize(150) == "Moderate"
        assert s.categorize(250) == "Poor"
        assert s.categorize(400) == "Very Poor"

    def test_categorize_iaqi(self):
        s = YAMLStandard(get_standard_def("iaqi"))
        assert s.categorize(10) == "Very Poor"
        assert s.categorize(90) == "Excellent"

    def test_category_distribution(self):
        s = YAMLStandard(get_standard_def("bsec"))
        values = [10, 20, 60, 80, 120, 250, 350]
        dist = s.category_distribution(values)
        assert dist["Excellent"] == 2
        assert dist["Good"] == 2
        assert dist["Moderate"] == 1
        assert dist["Poor"] == 1
        assert dist["Very Poor"] == 1

    def test_inf_upper_bound(self):
        d = get_standard_def("bsec")
        last_bound, last_name = d.categories[-1]
        assert math.isinf(last_bound)
        assert last_name == "Very Poor"


# ── Registration ─────────────────────────────────────────────────────────


class TestRegistration:
    def test_register_populates_registry(self):
        from app.profiles import _STANDARD_REGISTRY
        for name in ("bsec", "epa_aqi", "iaqi", "reset_air"):
            assert name in _STANDARD_REGISTRY

    def test_registry_classes_instantiate(self):
        from app.profiles import _STANDARD_REGISTRY
        for name in ("bsec", "epa_aqi", "iaqi", "reset_air"):
            instance = _STANDARD_REGISTRY[name]()
            assert instance.name == name


# ── list_standards ───────────────────────────────────────────────────────


class TestListStandards:
    def test_returns_all_four(self):
        standards = list_standards()
        names = [s.name for s in standards]
        assert set(names) == {"bsec", "epa_aqi", "iaqi", "reset_air"}
