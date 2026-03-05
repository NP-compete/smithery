"""Tool selection accuracy metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smithery.types import ToolCall


def tool_selection_accuracy(
    predicted: list[list[ToolCall]],
    expected: list[list[ToolCall]],
) -> float:
    """Compute the fraction of examples where the correct tool was selected.

    For multi-tool examples, all tools must match (order-insensitive).

    Args:
        predicted: Model predictions (one list of calls per example).
        expected: Ground truth (one list of calls per example).

    Returns:
        Accuracy as a float in [0, 1].
    """
    raise NotImplementedError
