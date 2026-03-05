"""Convert between ChatML, Hermes, ReAct, and OpenAI tool-call formats."""

from __future__ import annotations

from typing import Any

from smithery.types import TrainingExample


def to_chatml(example: TrainingExample) -> list[dict[str, Any]]:
    """Convert a training example to ChatML format with tool annotations."""
    raise NotImplementedError


def to_hermes(example: TrainingExample) -> list[dict[str, Any]]:
    """Convert a training example to NousResearch Hermes format."""
    raise NotImplementedError


def to_react(example: TrainingExample) -> list[dict[str, Any]]:
    """Convert a training example to ReAct (Thought/Action/Observation) format."""
    raise NotImplementedError


def from_openai(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling format to smithery canonical form."""
    raise NotImplementedError


def from_json(path: str) -> list[dict[str, Any]]:
    """Load tool definitions from a raw JSON file."""
    raise NotImplementedError


def from_hub(dataset_id: str) -> list[dict[str, Any]]:
    """Load an existing dataset from Hugging Face Hub (xLAM, Toucan, etc.)."""
    raise NotImplementedError
