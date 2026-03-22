"""Generic finite state machine for sequential stage execution.

Domain-free: knows nothing about training, models, data, or ML.
Reusable for any pipeline that runs named stages in sequence.
"""

import logging
import time
from enum import Enum
from typing import Callable, List, Optional

from training.pipeline.types import FailureInfo, PipelineError, StageResult

logger = logging.getLogger("training.pipeline")


class StageMachine:
    """Runs named stages in sequence with timing, callbacks, and error handling.

    Usage::

        fsm = StageMachine()
        fsm.add_stage(MyState.STEP_A, do_step_a)
        fsm.add_stage(MyState.STEP_B, do_step_b)
        fsm.on_stage_enter(my_enter_cb)
        fsm.on_stage_complete(my_complete_cb)
        fsm.on_error(my_error_cb)
        results = fsm.run()
    """

    def __init__(self):
        self._stages: List[tuple] = []  # [(state, fn), ...]
        self._results: List[StageResult] = []
        self._state: Optional[Enum] = None
        self._failure: Optional[FailureInfo] = None

        self._enter_callbacks: list = []
        self._complete_callbacks: list = []
        self._error_callbacks: list = []

    def add_stage(self, state: Enum, fn: Callable[[], StageResult]) -> None:
        """Register a stage to execute in order."""
        self._stages.append((state, fn))

    def on_stage_enter(self, *callbacks) -> "StageMachine":
        """Register callback(s) fired when a stage begins. Returns self for chaining."""
        self._enter_callbacks.extend(callbacks)
        return self

    def on_stage_complete(self, *callbacks) -> "StageMachine":
        """Register callback(s) fired when a stage completes. Returns self for chaining."""
        self._complete_callbacks.extend(callbacks)
        return self

    def on_error(self, *callbacks) -> "StageMachine":
        """Register callback(s) fired on stage failure. Returns self for chaining."""
        self._error_callbacks.extend(callbacks)
        return self

    @property
    def state(self) -> Optional[Enum]:
        """Current FSM state (last entered stage)."""
        return self._state

    @property
    def failure(self) -> Optional[FailureInfo]:
        """FailureInfo if a stage raised, else None."""
        return self._failure

    @property
    def stage_results(self) -> List[StageResult]:
        """Results collected so far."""
        return list(self._results)

    def _fire_enter(self, state: Enum) -> None:
        self._state = state
        logger.info("[%s] Starting...", state.value)
        for cb in self._enter_callbacks:
            cb(state, None)

    def _fire_complete(self, state: Enum, result: StageResult) -> None:
        msg = "[%s] Completed in %.2fs"
        args: list = [state.value, result.duration_seconds]
        if result.rows_out is not None:
            msg += " (%d rows)"
            args.append(result.rows_out)
        logger.info(msg, *args)
        logger.debug("[%s] extra=%s", state.value, result.extra)
        self._results.append(result)
        for cb in self._complete_callbacks:
            cb(state, result)

    def _fire_error(self, state: Enum, error: Exception) -> FailureInfo:
        from app.exceptions import IAQError

        logger.error(
            "[%s] FAILED: %s: %s",
            state.value,
            type(error).__name__,
            error,
            exc_info=True,
        )
        error_code = error.code.value if isinstance(error, IAQError) else None
        suggestion = error.suggestion if isinstance(error, IAQError) else None
        info = FailureInfo(
            failed_state=state,
            error=error,
            stage_results=list(self._results),
            error_code=error_code,
            suggestion=suggestion,
        )
        self._failure = info
        for cb in self._error_callbacks:
            cb(info)
        return info

    def run(
        self,
        *,
        interrupted: Callable[[], bool] = None,
        after_stage: Callable[[Enum, StageResult], None] = None,
    ) -> List[StageResult]:
        """Execute all stages in order.

        Args:
            interrupted: If provided, checked after each stage. Stops early
                if it returns True.
            after_stage: Optional hook called after each stage completes
                (after callbacks). Receives (state, result). Useful for
                domain-specific logic like logging preprocessing reports
                or skipping remaining stages on interrupt.

        Returns:
            List of StageResult for completed stages.
        """
        for state, fn in self._stages:
            self._fire_enter(state)
            try:
                result = fn()
            except Exception as e:
                info = self._fire_error(state, e)
                raise PipelineError(
                    f"Pipeline failed at {state.value}: {type(e).__name__}: {e}",
                    failure_info=info,
                ) from e
            self._fire_complete(state, result)

            if after_stage is not None:
                after_stage(state, result)

            if interrupted and interrupted():
                break

        return list(self._results)
