"""Orchestrator: run all metrics and produce a unified report."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from smithery.types import EvalMetrics

if TYPE_CHECKING:
    from smithery.types import ToolDefinition, TrainingExample


class AgentBenchmark:
    """Run the full evaluation suite against a fine-tuned model.

    Args:
        model_path: Path to the fine-tuned model directory.
        tools: Tool definitions used during training.
        test_set: Held-out test examples.
    """

    def __init__(
        self,
        model_path: str | Path,
        tools: list[ToolDefinition],
        test_set: list[TrainingExample],
    ) -> None:
        self._model_path = Path(model_path)
        self._tools = tools
        self._test_set = test_set

    def run(self) -> EvalMetrics:
        """Execute all evaluation metrics and return aggregated results.

        Runs:
            1. Tool selection accuracy
            2. Parameter extraction F1
            3. Multi-step plan success
            4. Safety refusal rate
            5. No-tool accuracy
            6. Latency measurement

        Returns:
            EvalMetrics with all scores populated.
        """
        raise NotImplementedError
