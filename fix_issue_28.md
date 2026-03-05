# Pull Request: Implement `--version` Flag and `smithery version` Command

## Summary

This pull request adds a `--version` flag and a `smithery version` command to the CLI tool. The `--version` flag allows users to quickly check the version of `smithery`, while the `smithery version` command provides detailed information about the environment, including Python version and key dependencies. This enhancement improves the usability of the `smithery` CLI by making it easier for users to verify their setup.

## Implementation Details

### 1. Version Callback in `smithery/cli.py`

We have added a version callback function to handle the `--version` flag. The callback prints the version of the `smithery` application and exits.

```python
# smithery/cli.py
from typer import Typer, Option, Exit, Context
from rich.console import Console
from smithery import __version__

app = Typer()
console = Console()

def version_callback(value: bool) -> None:
    if value:
        console.print(f"smithery {__version__}")
        raise Exit()

@app.callback()
def main(
    version: Option(
        bool, 
        "--version", "-V", 
        callback=version_callback, 
        is_eager=True, 
        help="Show version."
    ) = None
) -> None:
    """Forge tool-calling models from your API definitions."""
    ...

@app.command()
def version() -> None:
    """Show smithery version and environment info."""
    import platform
    import sys
    
    console.print(f"smithery {__version__}")
    console.print(f"Python {sys.version.split()[0]}")
    
    try:
        import torch
        cuda = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU"
        console.print(f"PyTorch {torch.__version__} ({cuda})")
    except ImportError:
        console.print("PyTorch: not installed")
    
    try:
        import transformers
        console.print(f"Transformers {transformers.__version__}")
    except ImportError:
        console.print("Transformers: not installed")
    
    try:
        import peft
        console.print(f"PEFT {peft.__version__}")
    except ImportError:
        console.print("PEFT: not installed")

    console.print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
```

### 2. Test Cases in `tests/unit/test_cli.py`

We have added tests for both the `--version` flag and the `smithery version` command to ensure correct output and functionality.

```python
# tests/unit/test_cli.py
from typer.testing import CliRunner
from smithery.cli import app

runner = CliRunner()

def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "smithery" in result.stdout

def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "smithery" in result.stdout
    assert "Python" in result.stdout
    assert "Platform" in result.stdout
    # Additional assertions for dependencies can be added if they are installed
```

## Explanation of Changes

- Introduced a version callback function to process the `--version` flag.
- Added the `version` command that retrieves and displays comprehensive environment details.
- Extended test cases to encompass verification of the new functionality.

By implementing these changes, users of the `smithery` CLI will be able to quickly access version information, making it easier to troubleshoot and share environment setup details when needed.