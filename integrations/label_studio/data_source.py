"""Label Studio data source — fetches labeled training data from a Label Studio project."""

import logging
from typing import Optional

import pandas as pd

from app.exceptions import ConfigurationError, SchemaMismatchError
from training.data_sources import DataSource

logger = logging.getLogger("integrations.label_studio")


class LabelStudioDataSource(DataSource):
    """Fetches labeled training data from a Label Studio project.

    Tasks in the project are expected to have sensor readings in their ``data``
    payload (keyed by the active SensorProfile's raw_features plus the
    IAQStandard's target_column).  Annotations may contain a corrected IAQ
    number or a "reject" flag; unannotated tasks use the original BSEC value
    from the task data.

    Annotation schema (defined in Label Studio labeling config):
        - Number tag named ``iaq_corrected``: override the IAQ value for a task.
        - Choices tag with choice ``"reject"``: exclude the task from training.

    See fetch() for the full annotation resolution logic (Stage 2).
    """

    def __init__(self, project_id: int = None, url: str = None, api_key: str = None):
        """
        Args:
            project_id: Label Studio project ID. Falls back to integrations.yaml,
                then label_studio.project_id in model_config.yaml.
            url: Label Studio server URL. Falls back to integrations.yaml,
                then label_studio.url in model_config.yaml.
            api_key: API token. Falls back to LABEL_STUDIO_API_KEY env var,
                then integrations.yaml, then model_config.yaml.
        """
        from integrations.config import get_integration_config
        from app.config import settings

        int_cfg = get_integration_config("label_studio")
        ls_cfg = settings.get_label_studio_config()

        self._url = (url or int_cfg.get("url") or ls_cfg["url"]).rstrip("/")
        self._api_key = api_key or int_cfg.get("api_key") or ls_cfg["api_key"]
        self._project_id = project_id or int_cfg.get("project_id") or ls_cfg.get("project_id")
        self._fetch_stats: Optional[dict] = None

    @property
    def name(self) -> str:
        return f"LabelStudio({self._url}/projects/{self._project_id})"

    def validate(self) -> None:
        """Verify Label Studio is reachable and the project exists."""
        import requests

        if not self._project_id:
            raise ConfigurationError(
                "Label Studio project_id is required. "
                "Set label_studio.project_id in model_config.yaml "
                "or pass --ls-project-id on the CLI."
            )
        if not self._api_key:
            raise ConfigurationError(
                "Label Studio API key is required. "
                "Set LABEL_STUDIO_API_KEY env var or label_studio.api_key in model_config.yaml."
            )

        headers = {"Authorization": f"Token {self._api_key}"}

        # Health check
        try:
            resp = requests.get(f"{self._url}/api/health", headers=headers, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot reach Label Studio at {self._url}. Is it running? ({e})"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(
                f"Label Studio health check failed ({resp.status_code}): {e}"
            ) from e

        # Project existence check
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
                    f"Label Studio project {self._project_id} not found at {self._url}. "
                    f"Check the project ID."
                )
            raise

        project_title = resp.json().get("title", f"project {self._project_id}")
        logger.info(
            "Connected to Label Studio project %d: '%s' (%s)",
            self._project_id,
            project_title,
            self._url,
        )

    def fetch(self) -> pd.DataFrame:
        """Export labeled tasks and assemble a training DataFrame.

        Annotation resolution logic:
        - Tasks with a ``Choices`` result containing ``"reject"`` (case-insensitive)
          are excluded entirely.
        - Tasks with a ``Number`` result named ``iaq_corrected`` have their target
          value overridden by the annotator-supplied number.
        - Unannotated tasks (or tasks whose annotations are all cancelled/skipped)
          use the original target value from the task ``data`` payload.

        The active ``SensorProfile`` and ``IAQStandard`` determine which columns
        are required.  ``sensor.field_mapping`` from ``model_config.yaml`` is
        applied to translate external field names to internal ones.

        Returns a DataFrame whose columns include all of ``SensorProfile.raw_features``
        plus ``IAQStandard.target_column``.  The index is a ``DatetimeIndex`` when a
        timestamp column is detectable, otherwise a plain RangeIndex (the pipeline
        handles both).
        """
        import requests

        from app.config import settings
        from app.profiles import get_iaq_standard, get_sensor_profile

        profile = get_sensor_profile()
        standard = get_iaq_standard()
        cfg = settings.load_model_config()
        field_mapping = cfg.get("sensor", {}).get("field_mapping", {})

        headers = {"Authorization": f"Token {self._api_key}"}

        logger.info(
            "Exporting tasks from Label Studio project %d (%s)...",
            self._project_id,
            self._url,
        )
        resp = requests.get(
            f"{self._url}/api/projects/{self._project_id}/export",
            headers=headers,
            params={"exportType": "JSON"},
            timeout=120,
        )
        resp.raise_for_status()
        tasks = resp.json()
        logger.info("Received %d tasks from Label Studio", len(tasks))

        stats = {
            "total": len(tasks),
            "accepted": 0,
            "rejected": 0,
            "iaq_corrected": 0,
            "unannotated": 0,
        }
        rows = []

        for task in tasks:
            task_data = dict(task.get("data", {}))
            annotations = task.get("annotations", [])

            rejected = False
            iaq_override = None

            for annotation in annotations:
                if annotation.get("was_cancelled") or annotation.get("skipped"):
                    continue
                for result in annotation.get("result", []):
                    r_type = result.get("type", "")
                    from_name = result.get("from_name", "")
                    value = result.get("value", {})

                    if r_type == "choices":
                        choices = [c.lower() for c in value.get("choices", [])]
                        if "reject" in choices:
                            rejected = True
                            break
                    elif r_type == "number" and from_name == "iaq_corrected":
                        iaq_override = value.get("number")
                if rejected:
                    break

            if rejected:
                stats["rejected"] += 1
                continue

            if field_mapping:
                task_data = {field_mapping.get(k, k): v for k, v in task_data.items()}

            active_annotations = [
                a
                for a in annotations
                if not a.get("was_cancelled") and not a.get("skipped")
            ]
            if iaq_override is not None:
                task_data[standard.target_column] = float(iaq_override)
                stats["iaq_corrected"] += 1
            elif not active_annotations:
                stats["unannotated"] += 1

            if "created_at" in task and "_created_at" not in task_data:
                task_data["_created_at"] = task["created_at"]

            stats["accepted"] += 1
            rows.append(task_data)

        logger.info(
            "LabelStudio annotation resolution: %d total -> %d accepted "
            "(%d iaq_corrected, %d unannotated, %d rejected)",
            stats["total"],
            stats["accepted"],
            stats["iaq_corrected"],
            stats["unannotated"],
            stats["rejected"],
        )

        if not rows:
            from app.exceptions import NoDataError

            raise NoDataError(
                f"No usable tasks in Label Studio project {self._project_id} "
                f"(total={stats['total']}, rejected={stats['rejected']}). "
                f"Ensure tasks have data payloads and are not all rejected.",
                suggestion="Check Label Studio project for accepted tasks",
            )

        df = pd.DataFrame(rows)

        ts_candidates = {"timestamp", "time", "datetime", "date", "ts", "_created_at"}
        for col in list(df.columns):
            if col.lower().strip() in ts_candidates:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.set_index(col)
                logger.info("Set timestamp index from column: %s", col)
                break

        required = list(profile.raw_features) + [standard.target_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SchemaMismatchError(
                f"Label Studio tasks are missing required fields: {missing}. "
                f"Available: {list(df.columns)}. "
                f"Check that task data payloads include these fields, "
                f"or add a sensor.field_mapping in model_config.yaml."
            )

        if (
            profile.quality_column
            and profile.quality_column in df.columns
            and profile.quality_min is not None
        ):
            before = len(df)
            df = df[df[profile.quality_column] >= profile.quality_min]
            logger.info(
                "Quality filter (%s >= %s): %d -> %d rows",
                profile.quality_column,
                profile.quality_min,
                before,
                len(df),
            )

        df = df.dropna(subset=required)

        self._fetch_stats = stats

        logger.info(
            "LabelStudio fetch complete: %d training rows (columns: %s)",
            len(df),
            list(df.columns),
        )
        return df

    @property
    def metadata(self) -> dict:
        meta = {
            "source_type": "label_studio",
            "url": self._url,
            "project_id": self._project_id,
        }
        if self._fetch_stats is not None:
            meta["annotation_stats"] = self._fetch_stats
        return meta
