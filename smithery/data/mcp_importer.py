"""Import tool definitions from MCP servers and tool descriptors."""

from __future__ import annotations

from pathlib import Path

from smithery.types import ToolDefinition


class MCPImporter:
    """Import tools from MCP server configurations or descriptor files.

    Use the class methods to construct from different sources.
    """

    def __init__(self, tools: list[ToolDefinition]) -> None:
        self._tools = tools

    @classmethod
    def from_server_config(cls, config_path: str | Path) -> MCPImporter:
        """Create importer by reading an MCP server configuration file.

        Args:
            config_path: Path to the MCP server config JSON.

        Returns:
            MCPImporter instance with discovered tools.
        """
        raise NotImplementedError

    @classmethod
    def from_tool_descriptors(cls, directory: str | Path) -> MCPImporter:
        """Create importer from a directory of MCP tool descriptor JSON files.

        Args:
            directory: Directory containing tool descriptor files.

        Returns:
            MCPImporter instance with parsed tools.
        """
        raise NotImplementedError

    def list_tools(self) -> list[ToolDefinition]:
        """Return all discovered tool definitions."""
        return list(self._tools)
