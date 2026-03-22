"""Value objects, enums, exception, and type aliases for the training pipeline."""

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("training.pipeline")


class PipelineState(enum.Enum):
    IDLE = "idle"
    SOURCE_ACCESS = "source_access"
    INGESTION = "ingestion"
    FEATURE_ENGINEERING = "feature_engineering"
    WINDOWING = "windowing"
    SCALING = "scaling"
    SPLITTING = "splitting"
    TRAINING = "training"
    EVALUATION = "evaluation"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


class IssueSeverity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Issue:
    severity: IssueSeverity
    stage: str
    message: str
    rows_affected: Optional[int] = None

    def to_dict(self) -> dict:
        d = {"severity": self.severity.value, "stage": self.stage, "message": self.message}
        if self.rows_affected is not None:
            d["rows_affected"] = self.rows_affected
        return d


@dataclass
class PreprocessingReport:
    issues: List[Issue] = field(default_factory=list)

    def add(
        self,
        severity: IssueSeverity,
        stage: str,
        message: str,
        rows_affected: int = None,
    ):
        self.issues.append(Issue(severity, stage, message, rows_affected))

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def has_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    def to_dict(self) -> dict:
        infos = [i for i in self.issues if i.severity == IssueSeverity.INFO]
        return {
            "issues": [i.to_dict() for i in self.issues],
            "summary": {
                "total": len(self.issues),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "infos": len(infos),
            },
        }

    def log_summary(self):
        if not self.issues:
            logger.info("[preprocessing_report] No issues found")
            return
        errors = self.errors
        warnings = self.warnings
        infos = [i for i in self.issues if i.severity == IssueSeverity.INFO]
        logger.info(
            "[preprocessing_report] %d issue(s): %d error, %d warning, %d info",
            len(self.issues),
            len(errors),
            len(warnings),
            len(infos),
        )
        for issue in self.issues:
            log_fn = {
                IssueSeverity.ERROR: logger.error,
                IssueSeverity.WARNING: logger.warning,
                IssueSeverity.INFO: logger.info,
            }[issue.severity]
            msg = "[preprocessing_report] [%s] %s"
            args = [issue.stage, issue.message]
            if issue.rows_affected is not None:
                msg += " (%d rows)"
                args.append(issue.rows_affected)
            log_fn(msg, *args)


@dataclass
class StageResult:
    state: PipelineState
    duration_seconds: float
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    columns: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureInfo:
    failed_state: PipelineState
    error: Exception
    stage_results: List[StageResult]
    error_code: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class PipelineResult:
    metrics: Dict[str, float]
    training_history: Dict
    model_dir: Path
    stage_results: List[StageResult]
    preprocessing_report: PreprocessingReport = field(
        default_factory=PreprocessingReport
    )
    version: str = ""
    merkle_root_hash: str = ""
    artifact_paths: List[Path] = field(default_factory=list)
    interrupted: bool = False

    @property
    def warnings(self) -> List[str]:
        """Surface preprocessing warnings as a flat string list."""
        return [
            f"[{issue.stage}] {issue.message}"
            + (f" ({issue.rows_affected} rows)" if issue.rows_affected else "")
            for issue in self.preprocessing_report.warnings
        ]


class PipelineError(Exception):
    def __init__(self, message: str, failure_info: FailureInfo):
        super().__init__(message)
        self.failure_info = failure_info


StageCallback = Callable[[PipelineState, StageResult], None]
ErrorCallback = Callable[[FailureInfo], None]
