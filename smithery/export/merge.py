"""Merge LoRA adapter weights into the base model."""

from __future__ import annotations

from pathlib import Path


def merge_adapter(
    adapter_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load a LoRA adapter and merge it into the base model.

    Reads the adapter config to find the base model, loads both in
    float16, merges weights, and saves the combined model.

    Args:
        adapter_path: Path to the LoRA adapter directory.
        output_path: Where to save the merged model.

    Returns:
        Path to the merged model directory.
    """
    raise NotImplementedError
