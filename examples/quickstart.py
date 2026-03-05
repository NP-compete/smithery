#!/usr/bin/env python3
"""Quickstart: generate data, train, evaluate, and export — end to end.

Usage:
    python examples/quickstart.py --tools ./my_tools.json
"""

from __future__ import annotations

from pathlib import Path

from smithery import __version__
from smithery.config import load_config
from smithery.data import MCPImporter, ToolCallGenerator, validate_dataset
from smithery.eval import AgentBenchmark
from smithery.train import AgentTrainer


def main(tools_path: str = "./my_tools.json") -> None:
    print(f"smithery v{__version__}")

    # 1. Import tools
    importer = MCPImporter.from_server_config(tools_path)
    tools = importer.list_tools()
    print(f"Loaded {len(tools)} tools")

    # 2. Generate training data
    generator = ToolCallGenerator(tools=tools)
    dataset = generator.generate(num_examples=5000)

    report = validate_dataset(dataset, tools)
    print(f"Valid: {report.valid_count}/{report.total_count}")

    # 3. Train
    config = load_config("configs/phi3-tool-calling.yaml")
    trainer = AgentTrainer(config)
    result = trainer.train(dataset)
    print(f"Training loss: {result.final_loss:.4f}")

    # 4. Evaluate
    benchmark = AgentBenchmark(
        model_path=result.model_path,
        tools=tools,
        test_set=dataset[:500],
    )
    metrics = benchmark.run()
    print(metrics.to_markdown())


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "./my_tools.json"
    main(path)
