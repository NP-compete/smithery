"""Core domain types shared across all smithery modules."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """A single parameter in a tool definition."""

    name: str
    type: str
    description: str = ""
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


class ToolDefinition(BaseModel):
    """Canonical representation of a callable tool."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    return_type: str = "string"

    @property
    def json_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.enum:
                properties[p.name]["enum"] = p.enum
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class DifficultyLevel(str, Enum):
    """Complexity level for synthetic training examples."""

    SINGLE_TOOL = "single_tool"
    MULTI_TOOL = "multi_tool"
    MULTI_STEP = "multi_step"
    REFUSAL = "refusal"


class ToolCall(BaseModel):
    """A single tool invocation with arguments."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TrainingExample(BaseModel):
    """One training row: user message → tool calls (or refusal)."""

    messages: list[dict[str, Any]]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.SINGLE_TOOL
    is_refusal: bool = False


class EvalMetrics(BaseModel):
    """Aggregated evaluation metrics for a fine-tuned model."""

    tool_selection_accuracy: float = 0.0
    parameter_extraction_f1: float = 0.0
    multi_step_plan_success: float = 0.0
    safety_refusal_rate: float = 0.0
    no_tool_accuracy: float = 0.0
    avg_latency_tok_per_sec: float = 0.0

    def to_markdown(self) -> str:
        """Render metrics as a markdown table."""
        rows = [
            ("Tool Selection Acc.", f"{self.tool_selection_accuracy:.1%}"),
            ("Parameter Extraction F1", f"{self.parameter_extraction_f1:.2f}"),
            ("Multi-step Plan Success", f"{self.multi_step_plan_success:.1%}"),
            ("Safety Refusal Rate", f"{self.safety_refusal_rate:.1%}"),
            ("No-Tool Accuracy", f"{self.no_tool_accuracy:.1%}"),
            ("Avg. Latency (tok/s)", f"{self.avg_latency_tok_per_sec:.0f}"),
        ]
        header = "| Metric | Score |\n|---|---|\n"
        body = "\n".join(f"| {name} | {val} |" for name, val in rows)
        return header + body


class TrainResult(BaseModel):
    """Summary of a completed training run."""

    model_path: str
    final_loss: float
    gpu_hours: float
    estimated_cost: float
    adapter_size_mb: float = 0.0


class ExportFormat(str, Enum):
    """Supported model export formats."""

    GGUF = "gguf"
    HF_HUB = "hf_hub"
    MERGED = "merged"
