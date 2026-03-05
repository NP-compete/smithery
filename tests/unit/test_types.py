"""Tests for smithery.types Pydantic models."""

from __future__ import annotations

from smithery.types import (
    DifficultyLevel,
    EvalMetrics,
    ExportFormat,
    ToolCall,
    ToolDefinition,
    TrainingExample,
    TrainResult,
)


class TestToolDefinition:
    def test_json_schema_basic(self, sample_tool: ToolDefinition) -> None:
        schema = sample_tool.json_schema
        assert schema["name"] == "get_weather"
        assert "city" in schema["parameters"]["properties"]
        assert "city" in schema["parameters"]["required"]

    def test_json_schema_optional_not_required(self, sample_tool: ToolDefinition) -> None:
        schema = sample_tool.json_schema
        assert "units" not in schema["parameters"]["required"]

    def test_json_schema_enum(self, sample_tool: ToolDefinition) -> None:
        schema = sample_tool.json_schema
        assert schema["parameters"]["properties"]["units"]["enum"] == [
            "celsius",
            "fahrenheit",
        ]

    def test_empty_parameters(self) -> None:
        tool = ToolDefinition(name="noop", description="Does nothing.")
        schema = tool.json_schema
        assert schema["parameters"]["properties"] == {}
        assert schema["parameters"]["required"] == []


class TestTrainingExample:
    def test_single_tool_example(self, sample_training_example: TrainingExample) -> None:
        assert len(sample_training_example.tool_calls) == 1
        assert sample_training_example.tool_calls[0].name == "get_weather"
        assert sample_training_example.difficulty == DifficultyLevel.SINGLE_TOOL
        assert sample_training_example.is_refusal is False

    def test_refusal_example(self, sample_refusal_example: TrainingExample) -> None:
        assert sample_refusal_example.is_refusal is True
        assert sample_refusal_example.tool_calls == []
        assert sample_refusal_example.difficulty == DifficultyLevel.REFUSAL


class TestEvalMetrics:
    def test_defaults(self) -> None:
        metrics = EvalMetrics()
        assert metrics.tool_selection_accuracy == 0.0
        assert metrics.parameter_extraction_f1 == 0.0

    def test_to_markdown(self) -> None:
        metrics = EvalMetrics(tool_selection_accuracy=0.96, parameter_extraction_f1=0.93)
        md = metrics.to_markdown()
        assert "96.0%" in md
        assert "0.93" in md
        assert "| Metric" in md


class TestToolCall:
    def test_basic(self) -> None:
        call = ToolCall(name="get_weather", arguments={"city": "Tokyo"})
        assert call.name == "get_weather"
        assert call.arguments["city"] == "Tokyo"

    def test_empty_arguments(self) -> None:
        call = ToolCall(name="noop")
        assert call.arguments == {}


class TestTrainResult:
    def test_basic(self) -> None:
        result = TrainResult(
            model_path="./output/test",
            final_loss=0.5,
            gpu_hours=1.5,
            estimated_cost=0.75,
        )
        assert result.model_path == "./output/test"
        assert result.adapter_size_mb == 0.0


class TestEnums:
    def test_difficulty_levels(self) -> None:
        assert DifficultyLevel.SINGLE_TOOL.value == "single_tool"
        assert DifficultyLevel.REFUSAL.value == "refusal"

    def test_export_formats(self) -> None:
        assert ExportFormat.GGUF.value == "gguf"
        assert ExportFormat.HF_HUB.value == "hf_hub"
