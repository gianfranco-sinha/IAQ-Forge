# ============================================================================
# File: app/standards.py
# YAML-driven IAQ standard registry — loads iaq_standards.yaml, provides
# generic YAMLStandard implementations compatible with IAQStandard ABC.
# ============================================================================
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from app.profiles import IAQStandard, register_standard


@dataclass
class StandardDef:
    """Parsed YAML standard definition."""
    name: str
    description: str
    target_column: str
    scale_range: Tuple[float, float]
    higher_is_worse: bool
    categories: List[Tuple[float, str]]


class YAMLStandard(IAQStandard):
    """Generic IAQStandard implementation wrapping a StandardDef."""

    def __init__(self, defn: StandardDef) -> None:
        self._defn = defn

    @property
    def name(self) -> str:
        return self._defn.name

    @property
    def description(self) -> str:
        return self._defn.description

    @property
    def target_column(self) -> str:
        return self._defn.target_column

    @property
    def scale_range(self) -> Tuple[float, float]:
        return self._defn.scale_range

    @property
    def higher_is_worse(self) -> bool:
        return self._defn.higher_is_worse

    @property
    def categories(self) -> List[Tuple[float, str]]:
        return self._defn.categories


# ---------------------------------------------------------------------------
# Registry loader (lazy singleton)
# ---------------------------------------------------------------------------
_definitions: Optional[Dict[str, StandardDef]] = None
_YAML_PATH = Path(__file__).resolve().parent.parent / "iaq_standards.yaml"


def _load_definitions() -> Dict[str, StandardDef]:
    global _definitions
    if _definitions is not None:
        return _definitions

    with open(_YAML_PATH) as f:
        raw = yaml.safe_load(f)

    _definitions = {}
    for name, entry in raw.items():
        sr = entry["scale_range"]
        cats = [
            (float(c["upper_bound"]), c["name"])
            for c in entry["categories"]
        ]
        _definitions[name] = StandardDef(
            name=name,
            description=entry["description"],
            target_column=entry["target_column"],
            scale_range=(float(sr[0]), float(sr[1])),
            higher_is_worse=entry["higher_is_worse"],
            categories=cats,
        )

    return _definitions


def reload_definitions() -> None:
    """Force-reload the standard definitions from disk."""
    global _definitions
    _definitions = None
    _load_definitions()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _make_standard_class(defn: StandardDef) -> type:
    """Create a zero-arg class compatible with registry's cls() pattern."""

    class _Standard(YAMLStandard):
        def __init__(self) -> None:
            super().__init__(defn)

    _Standard.__name__ = f"{defn.name.upper()}Standard"
    _Standard.__qualname__ = _Standard.__name__
    return _Standard


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_yaml_standards() -> None:
    """Register all YAML-defined standards into the profile registry."""
    for name, defn in _load_definitions().items():
        cls = _make_standard_class(defn)
        register_standard(name, cls)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_standard_def(name: str) -> StandardDef:
    """Look up a standard definition by name. Raises ConfigurationError if not found."""
    from app.exceptions import ConfigurationError

    defs = _load_definitions()
    if name not in defs:
        raise ConfigurationError(
            f"Unknown IAQ standard: '{name}'. "
            f"Available: {sorted(defs.keys())}",
            suggestion="Check iaq_standard.type in model_config.yaml",
        )
    return defs[name]


def list_standards() -> List[YAMLStandard]:
    """Return YAMLStandard instances for all defined standards."""
    return [YAMLStandard(defn) for defn in _load_definitions().values()]
