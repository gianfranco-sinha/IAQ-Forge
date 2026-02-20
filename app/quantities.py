# ============================================================================
# File: app/quantities.py
# Central physical quantity registry — loads quantities.yaml, provides
# typed lookup and safe unit conversion.
# ============================================================================
import ast
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class AlternateUnit:
    """A non-canonical unit with its conversion expression."""
    unit: str
    convert_expr: str


@dataclass
class Quantity:
    """A physical quantity from the registry."""
    name: str
    canonical_unit: str
    description: str
    valid_range: Optional[Tuple[float, float]] = None
    kind: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    alternate_units: Dict[str, AlternateUnit] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Safe AST-based expression evaluator (no eval())
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str, value: float) -> float:
    """Evaluate a simple arithmetic expression with a single variable ``value``.

    Only permits numeric literals, the variable name ``value``, and basic
    arithmetic operators (+, -, *, /, //, **).  No function calls, attribute
    access, or other constructs are allowed.
    """
    substituted = expr.replace("{value}", str(value))
    tree = ast.parse(substituted, mode="eval")

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "value":
            return value
        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_fn(_walk(node.operand))
        raise ValueError(
            f"Unsupported expression node: {type(node).__name__}"
        )

    return _walk(tree)


# ---------------------------------------------------------------------------
# Registry loader (lazy singleton)
# ---------------------------------------------------------------------------
_registry: Optional[Dict[str, Quantity]] = None
_YAML_PATH = Path(__file__).resolve().parent.parent / "quantities.yaml"


def _load_registry() -> Dict[str, Quantity]:
    global _registry
    if _registry is not None:
        return _registry

    with open(_YAML_PATH) as f:
        raw = yaml.safe_load(f)

    _registry = {}
    for name, entry in raw.items():
        alt_units = {}
        for unit_name, unit_cfg in (entry.get("alternate_units") or {}).items():
            alt_units[unit_name] = AlternateUnit(
                unit=unit_name, convert_expr=unit_cfg["convert"]
            )

        vr = entry.get("valid_range")
        _registry[name] = Quantity(
            name=name,
            canonical_unit=entry["canonical_unit"],
            description=entry["description"],
            valid_range=tuple(vr) if vr else None,
            kind=entry.get("kind"),
            aliases=entry.get("aliases", []),
            alternate_units=alt_units,
        )

    return _registry


def reload_registry() -> None:
    """Force-reload the quantity registry from disk."""
    global _registry
    _registry = None
    _load_registry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_quantity(name: str) -> Quantity:
    """Look up a quantity by name. Raises KeyError if not found."""
    reg = _load_registry()
    if name not in reg:
        raise KeyError(
            f"Unknown quantity: '{name}'. "
            f"Available: {sorted(reg.keys())}"
        )
    return reg[name]


def list_quantities() -> List[Quantity]:
    """Return all registered quantities."""
    return list(_load_registry().values())


def convert_to_canonical(value: float, from_unit: str, quantity_name: str) -> float:
    """Convert a value from ``from_unit`` to the quantity's canonical unit.

    If ``from_unit`` already matches the canonical unit, returns ``value``
    unchanged.
    """
    q = get_quantity(quantity_name)
    if from_unit == q.canonical_unit:
        return value
    alt = q.alternate_units.get(from_unit)
    if alt is None:
        raise ValueError(
            f"No conversion from '{from_unit}' to '{q.canonical_unit}' "
            f"for quantity '{quantity_name}'. "
            f"Known alternate units: {list(q.alternate_units.keys())}"
        )
    return _safe_eval(alt.convert_expr, value)
