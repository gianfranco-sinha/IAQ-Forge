"""Label Studio exporter — pushes InfluxDB data with before/after cleanse views."""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.profiles import SensorProfile
from app.standards import IAQStandard

logger = logging.getLogger("integrations.label_studio.exporter")


def detect_discontinuities(
    df: pd.DataFrame,
    gap_threshold_seconds: float = 30.0,
) -> List[Dict]:
    """Detect time gaps exceeding the threshold in a DatetimeIndex DataFrame.

    Returns:
        List of dicts with keys: index (position), before, after, gap_seconds.
    """
    gaps = []
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return gaps

    deltas = df.index.to_series().diff()
    threshold = pd.Timedelta(seconds=gap_threshold_seconds)

    for i, (idx, delta) in enumerate(deltas.items()):
        if pd.notna(delta) and delta > threshold:
            gaps.append({
                "index": i,
                "before": df.index[i - 1].isoformat(),
                "after": idx.isoformat(),
                "gap_seconds": delta.total_seconds(),
            })

    return gaps


def cleanse_dataframe(
    df: pd.DataFrame,
    profile: SensorProfile,
    target_column: str,
    gap_threshold_seconds: float = 30.0,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Apply the same cleansing logic the training pipeline uses.

    Returns:
        (clean_df, drop_reasons, report) — clean_df has dropped rows removed,
        drop_reasons is a Series indexed by the dropped row indices with
        human-readable reason strings, report contains ingest statistics
        including discontinuity count.
    """
    reasons: Dict[int, str] = {}
    valid_ranges = profile.valid_ranges

    # Check each row against valid ranges
    for idx in df.index:
        row = df.loc[idx]
        for feature, (lo, hi) in valid_ranges.items():
            if feature not in df.columns:
                continue
            val = row.get(feature)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                reasons[idx] = f"{feature} is NaN"
                break
            if val < lo or val > hi:
                reasons[idx] = f"{feature} outside valid range [{lo}, {hi}]"
                break

    # Also drop rows with NaN in required columns
    required = list(profile.raw_features) + [target_column]
    available_required = [c for c in required if c in df.columns]
    for idx in df.index:
        if idx in reasons:
            continue
        row = df.loc[idx]
        for col in available_required:
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                reasons[idx] = f"{col} is NaN"
                break

    # Quality filter
    if (
        profile.quality_column
        and profile.quality_column in df.columns
        and profile.quality_min is not None
    ):
        for idx in df.index:
            if idx in reasons:
                continue
            val = df.loc[idx].get(profile.quality_column)
            if val is not None and val < profile.quality_min:
                reasons[idx] = (
                    f"{profile.quality_column} ({val}) < {profile.quality_min}"
                )

    drop_indices = list(reasons.keys())
    drop_reasons = pd.Series(reasons, name="cleanse_reason")
    clean_df = df.drop(index=drop_indices)

    # Detect discontinuities in cleaned data
    gaps = detect_discontinuities(clean_df, gap_threshold_seconds)

    report = {
        "raw_rows": len(df),
        "clean_rows": len(clean_df),
        "dropped_rows": len(drop_indices),
        "discontinuities": len(gaps),
        "gap_threshold_seconds": gap_threshold_seconds,
        "gaps": gaps,
    }

    return clean_df, drop_reasons, report


class LabelStudioExporter:
    """Exports InfluxDB data to Label Studio with before/after cleanse views."""

    def __init__(
        self,
        project_id: int = None,
        url: str = None,
        api_key: str = None,
        batch_size: int = 50,
    ):
        from integrations.config import get_integration_config
        from app.config import settings

        int_cfg = get_integration_config("label_studio")
        ls_cfg = settings.get_label_studio_config()

        self._url = (url or int_cfg.get("url") or ls_cfg["url"]).rstrip("/")
        self._api_key = api_key or int_cfg.get("api_key") or ls_cfg["api_key"]
        self._project_id = (
            project_id or int_cfg.get("project_id") or ls_cfg.get("project_id")
        )
        self._batch_size = batch_size or int_cfg.get("batch_size", 50)

    def validate(self) -> None:
        """Verify Label Studio is reachable and the project exists."""
        import requests

        from app.exceptions import ConfigurationError

        if not self._project_id:
            raise ConfigurationError(
                "Label Studio project_id is required. "
                "Set label_studio.project_id in integrations.yaml "
                "or pass --project-id on the CLI."
            )
        if not self._api_key:
            raise ConfigurationError(
                "Label Studio API key is required. "
                "Set LABEL_STUDIO_API_KEY env var."
            )

        headers = {"Authorization": f"Token {self._api_key}"}

        try:
            resp = requests.get(
                f"{self._url}/api/health", headers=headers, timeout=10
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            from app.exceptions import ServiceUnreachableError

            raise ServiceUnreachableError(
                f"Cannot reach Label Studio at {self._url}. Is it running? ({e})",
                suggestion="Check that Label Studio is running and the URL is correct",
            ) from e
        except requests.exceptions.HTTPError as e:
            from app.exceptions import ServiceUnreachableError

            raise ServiceUnreachableError(
                f"Label Studio health check failed ({resp.status_code}): {e}",
                suggestion="Check Label Studio logs and API key",
            ) from e

        try:
            resp = requests.get(
                f"{self._url}/api/projects/{self._project_id}",
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            if resp.status_code == 404:
                raise ConfigurationError(
                    f"Label Studio project {self._project_id} not found at {self._url}."
                )
            raise

        project_title = resp.json().get("title", f"project {self._project_id}")
        logger.info(
            "Connected to Label Studio project %d: '%s'",
            self._project_id,
            project_title,
        )

    def build_before_after_tasks(
        self,
        raw_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        profile: SensorProfile,
        standard: IAQStandard,
        drop_reasons: Optional[pd.Series] = None,
        include_dropped: bool = True,
    ) -> List[dict]:
        """Build task payloads with raw/clean paired values.

        Args:
            raw_df: Full DataFrame before cleansing.
            clean_df: DataFrame after cleansing (subset of raw_df).
            profile: Active sensor profile.
            standard: Active IAQ standard.
            drop_reasons: Series mapping dropped indices to reason strings.
            include_dropped: Whether to include rows that were dropped during cleansing.

        Returns:
            List of task dicts ready for Label Studio import.
        """
        target = standard.target_column
        raw_features = list(profile.raw_features)
        data_columns = raw_features + ([target] if target in raw_df.columns else [])

        tasks = []
        kept_indices = clean_df.index

        # Kept rows — have both raw and clean values
        for idx in kept_indices:
            raw_row = raw_df.loc[idx]
            clean_row = clean_df.loc[idx]

            task_data = {"cleanse_status": "kept"}

            # Timestamp
            if isinstance(idx, pd.Timestamp):
                task_data["timestamp"] = idx.isoformat()

            for col in data_columns:
                raw_val = raw_row.get(col)
                clean_val = clean_row.get(col)
                task_data[f"raw_{col}"] = _safe_value(raw_val)
                task_data[f"clean_{col}"] = _safe_value(clean_val)

            tasks.append({"data": task_data})

        # Dropped rows — raw values only
        if include_dropped:
            dropped_indices = raw_df.index.difference(kept_indices)
            for idx in dropped_indices:
                raw_row = raw_df.loc[idx]

                task_data = {"cleanse_status": "dropped"}

                if isinstance(idx, pd.Timestamp):
                    task_data["timestamp"] = idx.isoformat()

                for col in data_columns:
                    raw_val = raw_row.get(col)
                    task_data[f"raw_{col}"] = _safe_value(raw_val)

                if drop_reasons is not None and idx in drop_reasons.index:
                    task_data["cleanse_reason"] = str(drop_reasons.loc[idx])

                tasks.append({"data": task_data})

        return tasks

    def export_tasks(
        self, tasks: List[dict], dry_run: bool = False
    ) -> dict:
        """POST tasks to Label Studio in batches.

        Args:
            tasks: List of task dicts (each with a "data" key).
            dry_run: If True, skip HTTP calls and return stats only.

        Returns:
            Dict with export stats: total, batches, created.
        """
        import requests

        stats = {
            "total": len(tasks),
            "batches": 0,
            "created": 0,
            "dry_run": dry_run,
        }

        if dry_run:
            stats["batches"] = math.ceil(len(tasks) / self._batch_size) if tasks else 0
            return stats

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
        }

        for i in range(0, len(tasks), self._batch_size):
            batch = tasks[i : i + self._batch_size]
            resp = requests.post(
                f"{self._url}/api/projects/{self._project_id}/import",
                json=batch,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            stats["batches"] += 1
            stats["created"] += len(batch)

            logger.info(
                "Batch %d: imported %d tasks (%d/%d)",
                stats["batches"],
                len(batch),
                stats["created"],
                len(tasks),
            )

        return stats


def _safe_value(val):
    """Convert numpy/pandas values to JSON-safe Python types."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        return None if math.isnan(v) else v
    if isinstance(val, float) and math.isnan(val):
        return None
    return val
