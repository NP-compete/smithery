"""Parameter extraction F1 and type accuracy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from smithery.types import ToolCall


class ParamMetrics(BaseModel):
    """Per-tool parameter extraction quality."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    type_accuracy: float = 0.0


def parameter_extraction_f1(
    predicted: list[list[ToolCall]],
    expected: list[list[ToolCall]],
) -> ParamMetrics:
    """Compute precision, recall, and F1 over extracted parameter key-value pairs.

    Args:
        predicted: Model predictions.
        expected: Ground truth.

    Returns:
        Aggregated ParamMetrics.
    """
    raise NotImplementedError
