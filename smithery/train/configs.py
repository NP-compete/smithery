"""Pre-built model presets for common base models."""

from __future__ import annotations

from smithery.config import LoRAConfig, SmitheryConfig, TrainingConfig

PRESETS: dict[str, SmitheryConfig] = {
    "phi3-tool-calling": SmitheryConfig(
        base_model="microsoft/Phi-3.5-mini-instruct",
        lora=LoRAConfig(r=16, alpha=32),
        training=TrainingConfig(epochs=3, learning_rate=2e-4, max_seq_length=4096),
    ),
    "llama3.2-react": SmitheryConfig(
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        lora=LoRAConfig(r=16, alpha=32),
        training=TrainingConfig(epochs=3, learning_rate=2e-4, max_seq_length=4096),
    ),
    "qwen2.5-mcp": SmitheryConfig(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        lora=LoRAConfig(r=16, alpha=32),
        training=TrainingConfig(epochs=3, learning_rate=2e-4, max_seq_length=4096),
    ),
    "mistral-7b-tools": SmitheryConfig(
        base_model="mistralai/Mistral-7B-Instruct-v0.3",
        lora=LoRAConfig(r=16, alpha=32),
        training=TrainingConfig(epochs=3, learning_rate=1e-4, max_seq_length=4096),
    ),
}


def get_preset(name: str) -> SmitheryConfig:
    """Retrieve a pre-built config by name.

    Raises:
        KeyError: If the preset name is not found.
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        msg = f"Unknown preset '{name}'. Available: {available}"
        raise KeyError(msg)
    return PRESETS[name].model_copy(deep=True)
