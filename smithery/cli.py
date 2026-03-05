"""Typer CLI — all smithery commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="smithery",
    help="Forge tool-calling models from your API definitions.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

# ---------------------------------------------------------------------------
# Sub-command groups
# ---------------------------------------------------------------------------

data_app = typer.Typer(help="Generate and manage training data.", no_args_is_help=True)
app.add_typer(data_app, name="data")

train_app = typer.Typer(help="Fine-tune models.", no_args_is_help=True)
app.add_typer(train_app, name="train")

eval_app = typer.Typer(help="Evaluate tool-calling accuracy.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")

export_app = typer.Typer(help="Export models for deployment.", no_args_is_help=True)
app.add_typer(export_app, name="export")

serve_app = typer.Typer(help="Serve and test models.", no_args_is_help=True)
app.add_typer(serve_app, name="serve")


# ---------------------------------------------------------------------------
# smithery data
# ---------------------------------------------------------------------------


@data_app.command("generate")
def data_generate(
    tools: Annotated[Path, typer.Option(help="Path to tool definitions JSON.")],
    num_examples: Annotated[int, typer.Option(help="Number of examples to generate.")] = 5000,
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("./data"),
) -> None:
    """Generate synthetic training data from tool definitions."""
    raise NotImplementedError


@data_app.command("import-mcp")
def data_import_mcp(
    server: Annotated[Path, typer.Option(help="Path to MCP server config.")],
    num_examples: Annotated[int, typer.Option(help="Number of examples.")] = 5000,
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("./data"),
) -> None:
    """Import tools from an MCP server and generate training data."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# smithery train
# ---------------------------------------------------------------------------


@train_app.callback(invoke_without_command=True)
def train(
    config: Annotated[Path, typer.Option(help="Path to training YAML config.")],
    data: Annotated[Path, typer.Option(help="Path to training JSONL file.")],
    output: Annotated[
        Optional[Path], typer.Option(help="Output directory for trained model.")
    ] = None,
) -> None:
    """Fine-tune a model with QLoRA."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# smithery eval
# ---------------------------------------------------------------------------


@eval_app.callback(invoke_without_command=True)
def evaluate(
    model: Annotated[Path, typer.Option(help="Path to fine-tuned model.")],
    test_set: Annotated[Path, typer.Option(help="Path to test JSONL file.")],
    tools: Annotated[Optional[Path], typer.Option(help="Path to tool definitions.")] = None,
) -> None:
    """Evaluate tool-calling accuracy."""
    raise NotImplementedError


@eval_app.command("compare")
def eval_compare(
    model_a: Annotated[Path, typer.Option(help="First model path.")],
    model_b: Annotated[str, typer.Option(help="Second model path or API name.")],
    test_set: Annotated[Path, typer.Option(help="Path to test JSONL.")],
    tools: Annotated[Path, typer.Option(help="Path to tool definitions.")],
) -> None:
    """Compare two models side-by-side."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# smithery export
# ---------------------------------------------------------------------------


@export_app.command("merge")
def export_merge(
    model: Annotated[Path, typer.Option(help="Path to LoRA adapter.")],
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("./merged"),
) -> None:
    """Merge LoRA adapter into base model."""
    raise NotImplementedError


@export_app.command("gguf")
def export_gguf(
    model: Annotated[Path, typer.Option(help="Path to merged model.")],
    quantize: Annotated[str, typer.Option(help="Quantization level.")] = "q4_k_m",
    output: Annotated[Optional[Path], typer.Option(help="Output GGUF file path.")] = None,
) -> None:
    """Export model to GGUF format for Ollama / llama.cpp."""
    raise NotImplementedError


@export_app.command("hf-hub")
def export_hf_hub(
    model: Annotated[Path, typer.Option(help="Path to model.")],
    repo: Annotated[str, typer.Option(help="Hugging Face Hub repo id.")],
) -> None:
    """Push model to Hugging Face Hub."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# smithery serve
# ---------------------------------------------------------------------------


@serve_app.command("test")
def serve_test(
    model: Annotated[Path, typer.Option(help="Path to model.")],
    tools: Annotated[Path, typer.Option(help="Path to tool definitions.")],
) -> None:
    """Interactive REPL for testing tool calls."""
    raise NotImplementedError
