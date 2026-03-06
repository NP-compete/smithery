"""Tests for CLI version flag and command."""

from typer.testing import CliRunner

from smithery import __version__
from smithery.cli import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "smithery" in result.stdout
    assert __version__ in result.stdout


def test_version_short_flag() -> None:
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "smithery" in result.stdout
    assert __version__ in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Python" in result.stdout
    assert "Platform:" in result.stdout


def test_version_command_shows_smithery_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
