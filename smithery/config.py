"""YAML config loader and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation hyperparameters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


class TrainingConfig(BaseModel):
    """Training loop parameters."""

    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03


class EvalConfig(BaseModel):
    """Post-training evaluation settings."""

    run_after_training: bool = True
    metrics: list[str] = Field(
        default_factory=lambda: ["tool_accuracy", "param_f1", "safety"]
    )


class DataConfig(BaseModel):
    """Data formatting and splitting."""

    format: str = "chatml_tools"
    train_split: float = 0.9
    shuffle: bool = True


class SmitheryConfig(BaseModel):
    """Top-level training recipe configuration."""

    base_model: str
    method: str = "qlora"
    quantization: str = "4bit"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    data: DataConfig = Field(default_factory=DataConfig)


def load_config(path: str | Path) -> SmitheryConfig:
    """Load and validate a YAML training config.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Validated SmitheryConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the YAML content is invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    return SmitheryConfig(**raw)
