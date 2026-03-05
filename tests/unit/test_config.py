"""Tests for smithery.config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from smithery.config import (
    LoRAConfig,
    SmitheryConfig,
    TrainingConfig,
    load_config,
)


class TestLoRAConfig:
    def test_defaults(self) -> None:
        config = LoRAConfig()
        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules


class TestTrainingConfig:
    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.max_seq_length == 4096


class TestSmitheryConfig:
    def test_minimal(self) -> None:
        config = SmitheryConfig(base_model="microsoft/Phi-3.5-mini-instruct")
        assert config.method == "qlora"
        assert config.quantization == "4bit"
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_custom_lora(self) -> None:
        config = SmitheryConfig(
            base_model="test/model",
            lora=LoRAConfig(r=32, alpha=64),
        )
        assert config.lora.r == 32
        assert config.lora.alpha == 64


class TestLoadConfig:
    def test_load_phi3_config(self, configs_dir: str) -> None:
        config = load_config(Path(configs_dir) / "phi3-tool-calling.yaml")
        assert config.base_model == "microsoft/Phi-3.5-mini-instruct"
        assert config.method == "qlora"
        assert config.lora.r == 16

    def test_load_mistral_config(self, configs_dir: str) -> None:
        config = load_config(Path(configs_dir) / "mistral-7b-tools.yaml")
        assert config.base_model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.training.learning_rate == 1e-4

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
