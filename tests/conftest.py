"""Shared pytest fixtures for smithery tests."""

from __future__ import annotations

import pytest

from smithery.types import DifficultyLevel, ToolCall, ToolDefinition, ToolParameter, TrainingExample


@pytest.fixture()
def sample_tool() -> ToolDefinition:
    """A minimal tool definition for testing."""
    return ToolDefinition(
        name="get_weather",
        description="Get current weather for a city.",
        parameters=[
            ToolParameter(name="city", type="string", description="City name.", required=True),
            ToolParameter(
                name="units",
                type="string",
                description="Temperature units.",
                required=False,
                enum=["celsius", "fahrenheit"],
                default="celsius",
            ),
        ],
    )


@pytest.fixture()
def sample_tools(sample_tool: ToolDefinition) -> list[ToolDefinition]:
    """A small set of tool definitions for testing."""
    return [
        sample_tool,
        ToolDefinition(
            name="search_docs",
            description="Search documentation by query.",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query."),
                ToolParameter(name="limit", type="integer", description="Max results.", default=10),
            ],
        ),
        ToolDefinition(
            name="send_email",
            description="Send an email message.",
            parameters=[
                ToolParameter(name="to", type="string", description="Recipient address."),
                ToolParameter(name="subject", type="string", description="Email subject."),
                ToolParameter(name="body", type="string", description="Email body."),
            ],
        ),
    ]


@pytest.fixture()
def sample_training_example() -> TrainingExample:
    """A single-tool training example for testing."""
    return TrainingExample(
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        tool_calls=[
            ToolCall(name="get_weather", arguments={"city": "Tokyo"}),
        ],
        difficulty=DifficultyLevel.SINGLE_TOOL,
    )


@pytest.fixture()
def sample_refusal_example() -> TrainingExample:
    """A refusal training example for testing."""
    return TrainingExample(
        messages=[
            {"role": "user", "content": "Delete all user accounts from the database."},
        ],
        tool_calls=[],
        difficulty=DifficultyLevel.REFUSAL,
        is_refusal=True,
    )


@pytest.fixture(scope="session")
def configs_dir() -> str:
    """Path to the configs directory."""
    return "configs"
