"""Safety evaluation: refusal rate and false refusal rate."""

from __future__ import annotations

from pydantic import BaseModel

from smithery.types import TrainingExample


class SafetyMetrics(BaseModel):
    """Refusal quality metrics."""

    refusal_rate: float = 0.0
    false_refusal_rate: float = 0.0
    total_refusal_examples: int = 0
    total_non_refusal_examples: int = 0


def evaluate_safety(
    predictions: list[str],
    examples: list[TrainingExample],
) -> SafetyMetrics:
    """Measure how well the model refuses dangerous requests.

    - refusal_rate: fraction of refusal examples correctly refused.
    - false_refusal_rate: fraction of valid requests incorrectly refused.

    Args:
        predictions: Raw model output strings.
        examples: Ground truth examples with is_refusal labels.

    Returns:
        SafetyMetrics with both rates.
    """
    raise NotImplementedError
