"""Export models to GGUF format for llama.cpp and Ollama."""

from __future__ import annotations

from pathlib import Path

QUANTIZATION_LEVELS = [
    "q2_k",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "q4_0",
    "q4_1",
    "q4_k_s",
    "q4_k_m",
    "q5_0",
    "q5_1",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
    "f16",
    "f32",
]


def export_gguf(
    model_path: str | Path,
    output_path: str | Path | None = None,
    *,
    quantize: str = "q4_k_m",
) -> Path:
    """Convert a HuggingFace model to GGUF format.

    Args:
        model_path: Path to the merged model directory.
        output_path: Output file path. Defaults to ``<model_path>.gguf``.
        quantize: Quantization level. Must be one of QUANTIZATION_LEVELS.

    Returns:
        Path to the exported GGUF file.

    Raises:
        ValueError: If the quantization level is not supported.
    """
    if quantize not in QUANTIZATION_LEVELS:
        msg = f"Unsupported quantization '{quantize}'. Choose from: {QUANTIZATION_LEVELS}"
        raise ValueError(msg)
    raise NotImplementedError
