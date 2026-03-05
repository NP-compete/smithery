"""Synthetic data generation: single, multi-tool, multi-step, and refusal examples."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smithery.types import DifficultyLevel, ToolDefinition, TrainingExample

if TYPE_CHECKING:
    from collections.abc import Sequence


class ToolCallGenerator:
    """Generates synthetic tool-calling training data from tool definitions.

    Args:
        tools: Tool definitions to generate training data for.
    """

    def __init__(self, tools: Sequence[ToolDefinition]) -> None:
        self._tools = list(tools)

    def generate(
        self,
        num_examples: int = 5000,
        difficulty_levels: Sequence[str | DifficultyLevel] | None = None,
    ) -> list[TrainingExample]:
        """Generate *num_examples* synthetic training examples.

        Args:
            num_examples: Total number of examples to produce.
            difficulty_levels: Which difficulty tiers to include.
                Defaults to all levels.

        Returns:
            List of validated training examples.
        """
        raise NotImplementedError
