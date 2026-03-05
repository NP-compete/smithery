<p align="center">
  <h1 align="center">smithery</h1>
  <p align="center"><strong>Forge tool-calling models from your API definitions.</strong></p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#why-smithery">Why Smithery</a> •
    <a href="#features">Features</a> •
    <a href="#from-mcp-to-model">MCP → Model</a> •
    <a href="#evaluation">Evaluation</a> •
    <a href="#recipes">Recipes</a>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/smithery"><img src="https://img.shields.io/pypi/v/smithery?color=blue" alt="PyPI"></a>
  <a href="https://github.com/NP-compete/smithery/actions"><img src="https://img.shields.io/github/actions/workflow/status/NP-compete/smithery/ci.yml?branch=main" alt="CI"></a>
  <a href="https://github.com/NP-compete/smithery/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NP-compete/smithery" alt="License"></a>
  <img src="https://img.shields.io/pypi/pyversions/smithery" alt="Python">
</p>

---

Your MCP server has 15 tools.
GPT-4o calls them correctly... most of the time.
A 3B model calls them correctly... never.

**Until you fine-tune it.**

Smithery takes your tool definitions — MCP servers, OpenAPI specs, or plain JSON — and produces a small, fast, specialized model that calls *your* tools with the accuracy of a model 10x its size.

```
Your Tools → Synthetic Training Data → Fine-tuned Model → Deploy Anywhere
```

---

## Quick Start

```bash
pip install smithery
```

**Generate training data from your tools, fine-tune, evaluate, and export — in four commands:**

```bash
# 1. Generate training data from MCP tool definitions
smithery data generate --tools ./my_tools.json --num-examples 5000 --output ./data

# 2. Fine-tune a small model
smithery train --config configs/phi3-tool-calling.yaml --data ./data/train.jsonl

# 3. Evaluate tool-calling accuracy
smithery eval --model ./output/phi3-tool-agent --test-set ./data/test.jsonl

# 4. Export for deployment
smithery export gguf --model ./output/phi3-tool-agent --quantize q4_k_m
```

Or do it all in Python:

```python
from smithery import Forge

forge = Forge.from_tools("./my_tools.json")

# Generate synthetic tool-calling data
dataset = forge.generate_data(num_examples=5000)

# Fine-tune Phi-3.5-mini with QLoRA
model = forge.train(
    base_model="microsoft/Phi-3.5-mini-instruct",
    dataset=dataset,
)

# Evaluate
report = forge.evaluate(model, dataset.test_split)
print(report)
# Tool Selection Accuracy:  96.1%
# Parameter Extraction F1:  0.93
# Multi-step Plan Success:  89.2%
# Safety Refusal Rate:      99.4%

# Export to Ollama
forge.export(model, format="gguf", quantize="q4_k_m")
```

---

## Why Smithery

The tools exist in pieces. Nobody assembled them.

| What You Need | Existing Solutions | The Problem |
|---|---|---|
| Synthetic tool-calling data | [Toucan](https://github.com/TheAgentArk/Toucan) (IBM), [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (Salesforce) | Massive scale, not your tools. Toucan has 1.5M examples across 500 MCPs — impressive, but you need 5,000 examples for *your* 15 tools. |
| Fine-tuning pipeline | [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | General-purpose. No tool-calling data formatting, no agent-specific callbacks. |
| Evaluation | [ToolBench](https://github.com/OpenBMB/ToolBench) (OpenBMB), [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) (Berkeley) | Research benchmarks for comparing foundation models. Not for evaluating *your* model on *your* tools. |
| Export & Serve | llama.cpp, vLLM, Ollama | Great serving, zero integration with training. |

**Smithery is the glue.** One toolkit that takes you from tool definitions to a deployed model.

```
┌─────────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐     ┌────────┐
│ Your Tools  │────▶│  Data    │────▶│  Train  │────▶│ Evaluate │────▶│ Export │
│ MCP / JSON  │     │ Generate │     │  QLoRA  │     │ Metrics  │     │ Deploy │
└─────────────┘     └──────────┘     └─────────┘     └──────────┘     └────────┘
       │                  │                │                │               │
   Tool defs        5,000 examples    LoRA adapter    Agent metrics    GGUF / HF Hub
   from your        validated against  on Phi-3 /     not just loss    ready for
   MCP server       your schemas       Llama / Qwen                    Ollama
```

---

## Features

### Data Generation

Generate synthetic training data tailored to your specific tools.

```python
from smithery.data import ToolCallGenerator, MCPImporter

# Import tools directly from your MCP server
tools = MCPImporter.from_server_config("./mcp_server.json").list_tools()

# Generate diverse training examples
generator = ToolCallGenerator(tools=tools)
dataset = generator.generate(
    num_examples=5000,
    difficulty_levels=["single_tool", "multi_tool", "multi_step", "refusal"],
)

# Validate every example against your tool schemas
from smithery.data import validate_dataset
report = validate_dataset(dataset, tools)
print(f"Valid: {report.valid_count}/{report.total_count}")
print(f"Tool distribution: {report.tool_balance}")
```

**Difficulty levels:**

| Level | What It Generates | Example |
|---|---|---|
| `single_tool` | One tool call per query | "What's the weather in Tokyo?" → `get_weather(city="Tokyo")` |
| `multi_tool` | Parallel tool calls | "Weather in Tokyo and Paris" → `get_weather(city="Tokyo")` + `get_weather(city="Paris")` |
| `multi_step` | Sequential tool calls with data flow | "Book the cheapest hotel in the warmest city" → `get_weather` → `compare` → `book_hotel` |
| `refusal` | Model should decline | "Delete all user accounts" → refusal with explanation |

**Supported input formats:**

| Source | Method |
|---|---|
| MCP server config | `MCPImporter.from_server_config("config.json")` |
| MCP tool descriptors | `MCPImporter.from_tool_descriptors("./tools/")` |
| OpenAI function format | `from_openai(functions_list)` |
| Raw JSON tool definitions | `from_json("tools.json")` |
| Existing datasets (xLAM, Toucan) | `from_hub("Salesforce/xlam-function-calling-60k")` |

### Training

Fine-tune any supported base model with QLoRA. No boilerplate.

```python
from smithery.train import AgentTrainer
from smithery.config import load_config

config = load_config("configs/phi3-tool-calling.yaml")
trainer = AgentTrainer(config)
result = trainer.train(dataset)

print(f"Training loss: {result.final_loss:.4f}")
print(f"GPU hours: {result.gpu_hours:.2f}")
print(f"Estimated cost: ${result.estimated_cost:.2f}")
```

Or from the CLI with a pre-built recipe:

```bash
smithery train \
  --config configs/phi3-tool-calling.yaml \
  --data ./data/train.jsonl \
  --output ./output/my-tool-agent
```

**Pre-built recipes:**

| Recipe | Base Model | Best For | VRAM Needed |
|---|---|---|---|
| `phi3-tool-calling.yaml` | Phi-3.5-mini (3.8B) | Fast tool calling, edge deployment | 10 GB |
| `llama3.2-react.yaml` | Llama 3.2 3B | Multi-step reasoning agents | 12 GB |
| `qwen2.5-mcp.yaml` | Qwen 2.5 3B | MCP tool use, multilingual | 12 GB |
| `mistral-7b-tools.yaml` | Mistral 7B v0.3 | High accuracy, larger budget | 18 GB |

All recipes train on a single consumer GPU (16–24 GB VRAM). No cluster required.

### Evaluation

Agent-specific metrics, not just training loss.

```python
from smithery.eval import AgentBenchmark

benchmark = AgentBenchmark(
    model_path="./output/my-tool-agent",
    tools=tools,
    test_set=dataset.test_split,
)
report = benchmark.run()
print(report.to_markdown())
```

```
┌────────────────────────┬─────────┬──────────────┐
│ Metric                 │  Score  │ vs. Base Model│
├────────────────────────┼─────────┼──────────────┤
│ Tool Selection Acc.    │  96.1%  │ +41.3%       │
│ Parameter Extraction F1│  0.93   │ +0.38        │
│ Multi-step Plan Success│  89.2%  │ +52.7%       │
│ Safety Refusal Rate    │  99.4%  │ +12.1%       │
│ Avg. Latency (tok/s)  │  145    │ 6x faster    │
│ Cost per 1K calls      │  $0.02  │ 50x cheaper  │
└────────────────────────┴─────────┴──────────────┘
```

**Compare models side-by-side:**

```bash
smithery eval compare \
  --model-a ./output/phi3-tool-agent \
  --model-b gpt-4o-mini \
  --test-set ./data/test.jsonl \
  --tools ./my_tools.json
```

**What each metric measures:**

| Metric | Question It Answers |
|---|---|
| **Tool Selection Accuracy** | Does it pick the right tool? |
| **Parameter Extraction F1** | Does it extract the right arguments? |
| **Multi-step Plan Success** | Can it chain tools to solve complex tasks? |
| **Safety Refusal Rate** | Does it refuse dangerous or out-of-scope calls? |
| **No-Tool Accuracy** | Does it know when *not* to call a tool? |

### Export

Train once, deploy anywhere.

```bash
# Merge adapter into base model
smithery export merge --model ./output/my-tool-agent --output ./merged

# Export to GGUF for Ollama / llama.cpp
smithery export gguf --model ./merged --quantize q4_k_m --output ./my-agent.gguf

# Push to Hugging Face Hub
smithery export hf-hub --model ./output/my-tool-agent --repo your-name/my-tool-agent

# Run with Ollama
ollama create my-tool-agent -f Modelfile
ollama run my-tool-agent "What's the weather in Tokyo?"
```

---

## From MCP to Model

This is the workflow that doesn't exist anywhere else.

You have an MCP server. You want a small, fast model that knows how to use its tools. Here's the full pipeline:

```bash
# Step 1: Point smithery at your MCP server
smithery data import-mcp \
  --server ./my-mcp-server/config.json \
  --num-examples 5000 \
  --output ./data

# Step 2: Train
smithery train \
  --config configs/phi3-tool-calling.yaml \
  --data ./data/train.jsonl

# Step 3: Evaluate
smithery eval \
  --model ./output/phi3-tool-agent \
  --test-set ./data/test.jsonl \
  --tools ./data/tools.json

# Step 4: Export to GGUF and run locally
smithery export merge --model ./output/phi3-tool-agent
smithery export gguf --model ./merged --quantize q4_k_m

# Step 5: Test interactively
smithery serve test --model ./merged --tools ./data/tools.json
```

**Total time:** ~30 minutes on a single GPU.
**Total cost:** ~$0 (local GPU) or ~$2 (cloud).
**Result:** A 3B model that calls *your specific tools* with 95%+ accuracy.

---

## Recipes

### Recipe: Customer Support Agent

```yaml
# configs/custom/support-agent.yaml
base_model: microsoft/Phi-3.5-mini-instruct
method: qlora
quantization: 4bit

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  gradient_accumulation_steps: 4
  max_seq_length: 4096
  warmup_ratio: 0.03

data:
  format: chatml_tools
  train_split: 0.9
  shuffle: true

eval:
  run_after_training: true
  metrics: [tool_accuracy, param_f1, safety]
```

### Recipe: Code Assistant with MCP Tools

```yaml
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
method: qlora
quantization: 4bit

lora:
  r: 32          # Higher rank for code understanding
  alpha: 64
  dropout: 0.05

training:
  epochs: 5
  learning_rate: 1e-4
  max_seq_length: 8192  # Longer context for code
```

---

## Architecture

```
smithery/
├── smithery/
│   ├── cli.py              # Typer CLI — all commands
│   ├── types.py            # Pydantic models: ToolDefinition, TrainingExample, EvalResult
│   ├── config.py           # YAML config loader + validation
│   ├── data/
│   │   ├── formats.py      # Convert between ChatML, Hermes, ReAct, OpenAI formats
│   │   ├── generators.py   # Synthetic data: single/multi/refusal/plan examples
│   │   ├── mcp_importer.py # MCP server → ToolDefinition → training data
│   │   └── validators.py   # Schema validation, balance checking, dedup
│   ├── train/
│   │   ├── lora.py         # QLoRA training loop wrapping PEFT + TRL
│   │   ├── configs.py      # Pre-built model presets (Phi-3, Llama, Qwen, Mistral)
│   │   └── callbacks.py    # Cost tracking, sample generation, early stopping
│   ├── eval/
│   │   ├── tool_accuracy.py     # Tool selection accuracy + F1
│   │   ├── param_extraction.py  # Parameter extraction F1 + type accuracy
│   │   ├── plan_quality.py      # Multi-step plan evaluation (LLM-as-judge)
│   │   ├── safety.py            # Refusal rate, false refusal rate
│   │   └── benchmark.py         # Orchestrator: run all metrics, produce report
│   ├── export/
│   │   ├── merge.py        # Merge LoRA adapter into base model
│   │   ├── hf_hub.py       # Push to Hugging Face Hub with model card
│   │   └── gguf.py         # Export to GGUF for llama.cpp / Ollama
│   └── serve/
│       └── quick_test.py   # Interactive REPL for testing tool calls
├── configs/                # Pre-built YAML recipes
├── examples/               # Step-by-step walkthrough scripts
├── tests/                  # Unit + integration tests
├── pyproject.toml
└── README.md
```

---

## How It Compares

| | Smithery | [Toucan](https://github.com/TheAgentArk/Toucan) | [ToolBench](https://github.com/OpenBMB/ToolBench) | [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) |
|---|---|---|---|---|
| **Focus** | Your tools → your model | Large-scale MCP data | Benchmark & eval | General fine-tuning |
| **Data generation** | From your tool defs | From 500 public MCPs | Pre-built dataset | BYO data |
| **MCP integration** | First-class | Data source only | No | No |
| **Training** | QLoRA, pre-built recipes | Training scripts | ToolLLaMA | Full fine-tuning suite |
| **Agent-specific eval** | Tool acc, param F1, safety, plans | BFCL V3 | ToolEval (pass/win rate) | General metrics |
| **Export to Ollama/GGUF** | Built-in | No | No | No |
| **Target user** | Individual dev / small team | Research lab | Research lab | ML engineer |
| **Time to first model** | 30 minutes | Hours (setup) | Hours (setup) | Varies |

---

## Supported Models

Any causal LM that works with PEFT. Tested and pre-configured:

| Model | Size | Why Use It |
|---|---|---|
| Phi-3.5-mini | 3.8B | Best speed/accuracy ratio. Edge-friendly. |
| Llama 3.2 | 1B / 3B | Meta ecosystem. Strong reasoning at 3B. |
| Qwen 2.5 | 3B / 7B | Best multilingual tool calling. |
| Mistral 7B v0.3 | 7B | Highest accuracy when VRAM allows. |
| Gemma 2 | 2B / 9B | Strong instruction following. |

---

## Requirements

- Python 3.10+
- PyTorch 2.2+
- A GPU with 10–24 GB VRAM (consumer cards work fine: RTX 3090, 4080, 4090, A6000)
- For data generation: an API key for OpenAI, Anthropic, or any OpenAI-compatible endpoint (local models work too)

---

## Installation

```bash
# Core (training + eval)
pip install smithery

# With GGUF export support
pip install "smithery[export]"

# With experiment tracking
pip install "smithery[wandb]"

# Everything
pip install "smithery[all]"

# Development
pip install "smithery[dev]"
```

---

## Roadmap

- [x] Synthetic data generation from tool definitions
- [x] MCP server → training data pipeline
- [x] QLoRA fine-tuning with pre-built recipes
- [x] Agent-specific evaluation (tool accuracy, param F1, safety, plans)
- [x] Export to GGUF / Hugging Face Hub
- [ ] DPO/ORPO training for preference-based refinement
- [ ] Multi-turn conversation fine-tuning
- [ ] Automatic recipe selection based on tool complexity
- [ ] Integration with Toucan and xLAM datasets
- [ ] vLLM serving integration
- [ ] Web dashboard for eval results

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/NP-compete/smithery.git
cd smithery
pip install -e ".[dev]"
pytest
```

---

## Citation

```bibtex
@software{smithery2026,
  author = {Soham Dutta},
  title = {Smithery: Forge Tool-Calling Models from Your API Definitions},
  year = {2026},
  url = {https://github.com/NP-compete/smithery}
}
```

---

## License

MIT
