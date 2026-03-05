"""QLoRA training loop wrapping PEFT + TRL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smithery.types import TrainResult

if TYPE_CHECKING:
    from smithery.config import SmitheryConfig
    from smithery.types import TrainingExample


class AgentTrainer:
    """Fine-tune a causal LM for tool calling with QLoRA.

    Args:
        config: Validated training configuration.
    """

    def __init__(self, config: SmitheryConfig) -> None:
        self._config = config

    def train(self, dataset: list[TrainingExample]) -> TrainResult:
        """Run the full training loop.

        Steps:
            1. Load base model with quantisation.
            2. Attach LoRA adapters.
            3. Format data into the configured chat template.
            4. Train with SFTTrainer from TRL.
            5. Return training summary.

        Args:
            dataset: Pre-validated training examples.

        Returns:
            TrainResult with loss, cost, and model path.
        """
        raise NotImplementedError
