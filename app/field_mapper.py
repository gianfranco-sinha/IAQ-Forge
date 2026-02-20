# ============================================================================
# File: app/field_mapper.py
# Tiered field mapping engine: exact match → fuzzy match.
# Maps CSV column headers (or API field names) to iaq4j internal features.
# ============================================================================
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.profiles import SensorProfile
from app.quantities import get_quantity, list_quantities

logger = logging.getLogger("app.field_mapper")

_TIMESTAMP_NAMES = {"timestamp", "time", "datetime", "date", "ts", "epoch", "created_at"}


@dataclass
class FieldMatch:
    """A single source→target field mapping."""
    source_field: str
    target_quantity: str
    target_feature: str
    confidence: float
    method: str  # "exact" or "fuzzy"
    inferred_unit: Optional[str] = None


@dataclass
class MappingResult:
    """Result of mapping source fields to iaq4j features."""
    matches: List[FieldMatch] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    timestamp_column: Optional[str] = None


class FieldMapper:
    """Maps external field names to iaq4j internal feature names.

    Uses a tiered strategy:
    1. Exact match (case-insensitive, ignoring hyphens/underscores)
    2. Fuzzy match via rapidfuzz (optional dependency)
    """

    def __init__(self, profile: SensorProfile, fuzzy_threshold: int = 70):
        self.profile = profile
        self.fuzzy_threshold = fuzzy_threshold
        self._build_candidates()

    def _build_candidates(self) -> None:
        """Build lookup tables from profile feature_quantities + registry."""
        # candidate_name → (feature_name, quantity_name)
        self._candidates: Dict[str, Tuple[str, str]] = {}

        for feat, qty_name in self.profile.feature_quantities.items():
            q = get_quantity(qty_name)
            # Feature name itself
            self._candidates[self._normalize(feat)] = (feat, qty_name)
            # Quantity name
            self._candidates[self._normalize(qty_name)] = (feat, qty_name)
            # Aliases from registry
            for alias in q.aliases:
                self._candidates[self._normalize(alias)] = (feat, qty_name)

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a field name for exact matching."""
        return re.sub(r"[-_ ]+", "", name.lower().strip())

    def map_fields(
        self,
        source_fields: List[str],
        sample_values: Optional[Dict[str, List[float]]] = None,
    ) -> MappingResult:
        """Map source field names to iaq4j features.

        Args:
            source_fields: column headers from CSV or API payload.
            sample_values: optional dict of field→sample values for range
                validation during fuzzy matching.

        Returns:
            MappingResult with matches, unresolved fields, and timestamp column.
        """
        result = MappingResult()

        # Detect timestamp column first
        remaining = list(source_fields)
        ts_col = self._detect_timestamp(remaining)
        if ts_col:
            result.timestamp_column = ts_col
            remaining.remove(ts_col)

        # Track which quantities have been matched to prevent duplicates
        matched_quantities: set = set()
        available_candidates = dict(self._candidates)

        for src in remaining:
            norm = self._normalize(src)

            # Tier 1: Exact match
            if norm in available_candidates:
                feat, qty_name = available_candidates[norm]
                if qty_name not in matched_quantities:
                    result.matches.append(FieldMatch(
                        source_field=src,
                        target_quantity=qty_name,
                        target_feature=feat,
                        confidence=1.0,
                        method="exact",
                    ))
                    matched_quantities.add(qty_name)
                    continue

            # Tier 2: Fuzzy match
            match = self._fuzzy_match(src, available_candidates, matched_quantities, sample_values)
            if match:
                result.matches.append(match)
                matched_quantities.add(match.target_quantity)
            else:
                result.unresolved.append(src)

        return result

    def _detect_timestamp(self, fields: List[str]) -> Optional[str]:
        """Identify timestamp columns by name."""
        for f in fields:
            if self._normalize(f) in {self._normalize(n) for n in _TIMESTAMP_NAMES}:
                return f
        return None

    def _fuzzy_match(
        self,
        source_field: str,
        candidates: Dict[str, Tuple[str, str]],
        matched_quantities: set,
        sample_values: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[FieldMatch]:
        """Try fuzzy matching against candidate names."""
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            return None

        candidate_names = [
            cn for cn, (_, qty) in candidates.items()
            if qty not in matched_quantities
        ]
        if not candidate_names:
            return None

        best = process.extractOne(
            self._normalize(source_field),
            candidate_names,
            scorer=fuzz.ratio,
        )
        if best is None:
            return None

        matched_name, score, _idx = best
        if score < self.fuzzy_threshold:
            return None

        feat, qty_name = candidates[matched_name]

        # Range validation bonus/penalty
        if sample_values and source_field in sample_values:
            q = get_quantity(qty_name)
            if q.valid_range:
                vals = sample_values[source_field]
                in_range = sum(1 for v in vals if q.valid_range[0] <= v <= q.valid_range[1])
                ratio = in_range / len(vals) if vals else 0
                if ratio > 0.8:
                    score = min(100, score + 10)
                elif ratio < 0.2:
                    score = max(0, score - 20)

        confidence = score / 100.0

        return FieldMatch(
            source_field=source_field,
            target_quantity=qty_name,
            target_feature=feat,
            confidence=confidence,
            method="fuzzy",
        )

    @staticmethod
    def sample_csv(path: str, n_rows: int = 10) -> Tuple[List[str], Dict[str, List[float]]]:
        """Read CSV headers and N sample rows.

        Returns:
            (headers, sample_values) where sample_values maps each numeric
            column to a list of float values.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with open(p, newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            headers = [h.strip() for h in headers]

            sample_values: Dict[str, List[float]] = {h: [] for h in headers}
            for i, row in enumerate(reader):
                if i >= n_rows:
                    break
                for h, val in zip(headers, row):
                    try:
                        sample_values[h].append(float(val))
                    except (ValueError, TypeError):
                        pass

        return headers, sample_values

    @staticmethod
    def format_report(result: MappingResult) -> str:
        """Format a MappingResult as a human-readable table."""
        lines = []
        lines.append(f"{'Source Field':<25} {'→':>2} {'Target Feature':<20} {'Quantity':<22} {'Conf':>5}  {'Method'}")
        lines.append("-" * 90)

        for m in sorted(result.matches, key=lambda x: x.confidence, reverse=True):
            lines.append(
                f"{m.source_field:<25} {'→':>2} {m.target_feature:<20} "
                f"{m.target_quantity:<22} {m.confidence:5.0%}  {m.method}"
            )

        if result.timestamp_column:
            lines.append(f"\nTimestamp column: {result.timestamp_column}")

        if result.unresolved:
            lines.append(f"\nUnresolved fields ({len(result.unresolved)}):")
            for u in result.unresolved:
                lines.append(f"  - {u}")

        return "\n".join(lines)
