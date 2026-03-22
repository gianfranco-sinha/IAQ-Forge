"""Tests for app/quantities.py — safe eval, registry lookup, unit conversion."""
import pytest
from app.quantities import (
    _safe_eval,
    get_quantity,
    convert_to_canonical,
    list_quantities,
    reload_registry,
)


# ── _safe_eval ─────────────────────────────────────────────────────────────


class TestSafeEval:
    def test_addition(self):
        assert _safe_eval("{value} + 1", 5.0) == 6.0

    def test_multiplication(self):
        assert _safe_eval("{value} * 2", 3.0) == 6.0

    def test_division(self):
        assert _safe_eval("{value} / 4", 12.0) == 3.0

    def test_power(self):
        assert _safe_eval("{value} ** 2", 3.0) == 9.0

    def test_floor_division(self):
        assert _safe_eval("{value} // 2", 7.0) == 3.0

    def test_unary_neg(self):
        assert _safe_eval("-{value}", 5.0) == -5.0

    def test_unary_pos(self):
        assert _safe_eval("+{value}", 5.0) == 5.0

    def test_combined_fahrenheit_to_celsius(self):
        result = _safe_eval("({value} - 32) * 5 / 9", 32.0)
        assert abs(result - 0.0) < 1e-9

    def test_combined_fahrenheit_212(self):
        result = _safe_eval("({value} - 32) * 5 / 9", 212.0)
        assert abs(result - 100.0) < 1e-9

    def test_value_zero(self):
        assert _safe_eval("{value} + 10", 0.0) == 10.0

    def test_negative_value(self):
        assert _safe_eval("{value} * 2", -3.0) == -6.0

    def test_large_value(self):
        assert _safe_eval("{value} + 1", 1e15) == 1e15 + 1

    # Security boundary
    def test_rejects_import(self):
        with pytest.raises(ValueError):
            _safe_eval("__import__('os')", 1.0)

    def test_rejects_function_call(self):
        with pytest.raises(ValueError):
            _safe_eval("print(1)", 1.0)

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError):
            _safe_eval("{value}.__class__", 1.0)

    def test_rejects_list_node(self):
        with pytest.raises(ValueError):
            _safe_eval("[1, 2, 3]", 1.0)

    def test_rejects_dict_node(self):
        with pytest.raises(ValueError):
            _safe_eval("{'a': 1}", 1.0)

    def test_rejects_unknown_variable(self):
        with pytest.raises(ValueError):
            _safe_eval("x + 1", 1.0)


# ── get_quantity ───────────────────────────────────────────────────────────


class TestGetQuantity:
    def test_temperature(self):
        q = get_quantity("temperature")
        assert q.name == "temperature"
        assert q.canonical_unit == "°C"

    def test_relative_humidity(self):
        q = get_quantity("relative_humidity")
        assert q.canonical_unit == "%RH"

    def test_barometric_pressure(self):
        q = get_quantity("barometric_pressure")
        assert q.canonical_unit == "hPa"

    def test_voc_resistance(self):
        q = get_quantity("voc_resistance")
        assert q.canonical_unit == "Ω"
        assert q.valid_range is not None

    def test_unknown_raises_configuration_error(self):
        from app.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown quantity"):
            get_quantity("nonexistent_quantity")

    def test_has_description(self):
        q = get_quantity("temperature")
        assert q.description  # non-empty

    def test_has_valid_range(self):
        q = get_quantity("temperature")
        assert q.valid_range == (-40, 85)


# ── convert_to_canonical ──────────────────────────────────────────────────


class TestConvertToCanonical:
    def test_fahrenheit_32_to_celsius(self):
        result = convert_to_canonical(32.0, "°F", "temperature")
        assert abs(result - 0.0) < 1e-9

    def test_fahrenheit_212_to_celsius(self):
        result = convert_to_canonical(212.0, "°F", "temperature")
        assert abs(result - 100.0) < 1e-9

    def test_kelvin_to_celsius(self):
        result = convert_to_canonical(273.15, "K", "temperature")
        assert abs(result - 0.0) < 1e-9

    def test_pa_to_hpa(self):
        result = convert_to_canonical(101325, "Pa", "barometric_pressure")
        assert abs(result - 1013.25) < 1e-9

    def test_kohm_to_ohm(self):
        result = convert_to_canonical(10.0, "kΩ", "voc_resistance")
        assert abs(result - 10000.0) < 1e-9

    def test_same_unit_passthrough(self):
        result = convert_to_canonical(25.0, "°C", "temperature")
        assert result == 25.0

    def test_unknown_unit_raises(self):
        from app.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="No conversion"):
            convert_to_canonical(100.0, "unknown_unit", "temperature")


# ── list_quantities ───────────────────────────────────────────────────────


class TestListQuantities:
    def test_non_empty(self):
        qs = list_quantities()
        assert len(qs) > 0

    def test_contains_temperature(self):
        names = [q.name for q in list_quantities()]
        assert "temperature" in names

    def test_contains_voc_resistance(self):
        names = [q.name for q in list_quantities()]
        assert "voc_resistance" in names


# ── reload_registry ───────────────────────────────────────────────────────


class TestReloadRegistry:
    def test_reload_then_get(self):
        reload_registry()
        q = get_quantity("temperature")
        assert q.name == "temperature"
