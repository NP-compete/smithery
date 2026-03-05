"""Training callbacks: cost tracking, sample generation, early stopping."""

from __future__ import annotations

from typing import Any


class CostTracker:
    """Track estimated GPU cost during training.

    Calculates approximate cost based on GPU type and elapsed time.
    """

    def __init__(self, gpu_cost_per_hour: float = 0.0) -> None:
        self._gpu_cost_per_hour = gpu_cost_per_hour
        self._elapsed_seconds: float = 0.0

    @property
    def estimated_cost(self) -> float:
        return self._gpu_cost_per_hour * (self._elapsed_seconds / 3600)

    def on_step_end(self, **kwargs: Any) -> None:
        """Called at the end of each training step."""
        raise NotImplementedError


class SampleGenerationCallback:
    """Generate sample tool calls at configurable intervals during training."""

    def on_evaluate(self, **kwargs: Any) -> None:
        raise NotImplementedError


class EarlyStoppingCallback:
    """Stop training when eval loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._best_loss: float | None = None
        self._wait = 0

    def on_evaluate(self, metrics: dict[str, float], **kwargs: Any) -> bool:
        """Return True if training should stop."""
        raise NotImplementedError
