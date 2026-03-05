"""Interactive REPL for testing tool calls against a fine-tuned model."""

from __future__ import annotations

from pathlib import Path


def start_repl(
    model_path: str | Path,
    tools_path: str | Path,
) -> None:
    """Launch an interactive session for testing tool calls.

    Loads the model and tool definitions, then enters a loop where
    the user types queries and sees the model's tool-call responses
    with formatted output.

    Args:
        model_path: Path to the fine-tuned (or merged) model.
        tools_path: Path to the tool definitions JSON.
    """
    raise NotImplementedError
